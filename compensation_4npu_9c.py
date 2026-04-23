import argparse
from contextlib import nullcontext
import json
import os
import time
from pathlib import Path
import warnings

import numpy as np
import scipy.io as sio
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

try:
    import torch_npu  # noqa: F401
    HAS_TORCH_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except ImportError:
    HAS_TORCH_NPU = False

# torch_npu on some releases emits this deprecation warning during backward.
warnings.filterwarnings(
    "ignore",
    message=r".*AutoNonVariableTypeMode is deprecated.*",
    category=UserWarning,
)


torch.manual_seed(42)
np.random.seed(42)

IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 2
IMG_TOTAL = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS

COMPRESSION_RATES = {
    1 / 4: 512,
    1 / 16: 128,
    1 / 32: 64,
    1 / 64: 32,
}


class QuantumCompensationBlock(nn.Module):
    """
    可在 NPU 上端到端训练。
    """

    def __init__(self, n_qubits=16, n_layers=2, window_size=4, entanglement_shifts=None, chunk_size=32):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.window_size = window_size
        self.chunk_size = max(1, int(chunk_size))
        self.state_dim = 1 << n_qubits
        if n_qubits != window_size * window_size:
            raise ValueError("n_qubits must equal window_size*window_size for fold/unfold reconstruction.")

        self.weights_crz = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.weights_ry = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.input_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.backend = "torch_statevector_gates"

        # bit_table[q, basis] in {0.0, 1.0}: qubit q 在给定基态下的比特值。
        basis = torch.arange(self.state_dim, dtype=torch.long)
        qubits = torch.arange(n_qubits, dtype=torch.long).unsqueeze(1)
        bit_table = ((basis.unsqueeze(0) >> qubits) & 1).to(torch.float32)
        self.register_buffer("bit_table", bit_table, persistent=False)

        # Per-layer entanglement schedule.
        # Default for 2 layers: layer-1 uses 1-hop, layer-2 uses 2-hop.
        if entanglement_shifts is not None:
            parsed = tuple(int(s) for s in entanglement_shifts if int(s) > 0)
            if not parsed:
                parsed = (1,)
        else:
            if n_layers >= 2:
                parsed = (1, 2)
            else:
                parsed = (1,)

        # Convert user/default shifts into per-layer schedule.
        # Example: n_layers=2 and parsed=(1,2) -> (1,2)
        #          n_layers=3 and parsed=(1,2) -> (1,2,2)
        if len(parsed) >= n_layers:
            self.layer_entanglement_shifts = tuple(parsed[:n_layers])
        else:
            self.layer_entanglement_shifts = tuple(parsed) + (parsed[-1],) * (n_layers - len(parsed))

    @staticmethod
    def _split_pairs(state, qubit, n_qubits):
        high = 1 << (n_qubits - qubit - 1)
        low = 1 << qubit
        return state.view(state.shape[0], high, 2, low), high, low

    def _apply_h(self, real, imag, qubit):
        real_view, _, _ = self._split_pairs(real, qubit, self.n_qubits)
        imag_view, _, _ = self._split_pairs(imag, qubit, self.n_qubits)

        a0r = real_view[:, :, 0, :]
        a1r = real_view[:, :, 1, :]
        a0i = imag_view[:, :, 0, :]
        a1i = imag_view[:, :, 1, :]

        inv_sqrt2 = 2.0 ** -0.5
        out0r = (a0r + a1r) * inv_sqrt2
        out1r = (a0r - a1r) * inv_sqrt2
        out0i = (a0i + a1i) * inv_sqrt2
        out1i = (a0i - a1i) * inv_sqrt2

        real_out = torch.stack([out0r, out1r], dim=2).reshape(real.shape)
        imag_out = torch.stack([out0i, out1i], dim=2).reshape(imag.shape)
        return real_out, imag_out

    def _apply_ry(self, real, imag, qubit, theta):
        real_view, _, _ = self._split_pairs(real, qubit, self.n_qubits)
        imag_view, _, _ = self._split_pairs(imag, qubit, self.n_qubits)

        a0r = real_view[:, :, 0, :]
        a1r = real_view[:, :, 1, :]
        a0i = imag_view[:, :, 0, :]
        a1i = imag_view[:, :, 1, :]

        if theta.dim() == 0:
            c = torch.cos(theta * 0.5).view(1, 1, 1)
            s = torch.sin(theta * 0.5).view(1, 1, 1)
        else:
            c = torch.cos(theta * 0.5).view(-1, 1, 1)
            s = torch.sin(theta * 0.5).view(-1, 1, 1)

        out0r = c * a0r - s * a1r
        out1r = s * a0r + c * a1r
        out0i = c * a0i - s * a1i
        out1i = s * a0i + c * a1i

        real_out = torch.stack([out0r, out1r], dim=2).reshape(real.shape)
        imag_out = torch.stack([out0i, out1i], dim=2).reshape(imag.shape)
        return real_out, imag_out

    def _apply_crz(self, real, imag, control, target, theta):
        bit_control = self.bit_table[control]
        bit_target = self.bit_table[target]

        mask10 = bit_control * (1.0 - bit_target)
        mask11 = bit_control * bit_target

        c = torch.cos(theta * 0.5)
        s = torch.sin(theta * 0.5)

        phase_real = 1.0 + (c - 1.0) * (mask10 + mask11)
        phase_imag = (-s) * mask10 + s * mask11

        new_real = real * phase_real.unsqueeze(0) - imag * phase_imag.unsqueeze(0)
        new_imag = real * phase_imag.unsqueeze(0) + imag * phase_real.unsqueeze(0)
        return new_real, new_imag

    def _z_expectation(self, real, imag):
        probs = real * real + imag * imag
        z_sign = 1.0 - 2.0 * self.bit_table
        return probs @ z_sign.transpose(0, 1)

    def _torch_quantum_forward(self, inputs):
        # 真实门级状态向量模拟：编码(H+RY) -> 数据重传 + 强纠缠CRZ + RY -> 测量<Z>。
        batch = inputs.shape[0]
        device = inputs.device
        dtype = inputs.dtype

        basis0 = torch.zeros(batch, device=device, dtype=torch.long)
        real = F.one_hot(basis0, num_classes=self.state_dim).to(dtype=dtype)
        imag = torch.zeros_like(real)

        enc_angles = torch.tanh(inputs * self.input_scale) * np.pi
        for q in range(self.n_qubits):
            real, imag = self._apply_h(real, imag, q)
            real, imag = self._apply_ry(real, imag, q, enc_angles[:, q])

        for layer in range(self.n_layers):
            # Scheme A: keep the initial encoding pass and re-upload from the second variational layer.
            if layer > 0:
                for q in range(self.n_qubits):
                    real, imag = self._apply_ry(real, imag, q, enc_angles[:, q])

            # Per-layer entanglement: layer-1 uses 1-hop, layer-2 uses 2-hop by default.
            shift = self.layer_entanglement_shifts[layer]
            for q in range(self.n_qubits):
                target = (q + shift) % self.n_qubits
                theta = self.weights_crz[layer, q]
                real, imag = self._apply_crz(real, imag, q, target, theta)

            for q in range(self.n_qubits):
                real, imag = self._apply_ry(real, imag, q, self.weights_ry[layer, q])

        z_exp = self._z_expectation(real, imag)
        return torch.clamp(z_exp, -1.0, 1.0)

    def forward(self, x):
        # x: [batch, encoded_dim]
        if x.dim() != 2:
            raise ValueError(f"QuantumCompensationBlock expects 2D latent input [B, D], got shape={tuple(x.shape)}")

        batch, dim = x.shape
        latent_h = 4
        if dim % latent_h != 0:
            raise ValueError(f"encoded dim must be divisible by {latent_h}, got {dim}")
        latent_w = dim // latent_h
        if latent_w % self.window_size != 0:
            raise ValueError(f"latent width {latent_w} must be divisible by window_size {self.window_size}")

        original_device = x.device

        x_map = x.reshape(batch, 1, latent_h, latent_w)
        # 将输入移动到量子补偿参数所在设备。
        x_proc = x_map.to(self.weights_crz.device)

        unfold = nn.Unfold(kernel_size=self.window_size, stride=self.window_size).to(x_proc.device)
        patches = unfold(x_proc)  # [batch, 16, num_patches]
        num_patches = patches.shape[-1]

        # 对 encoded_dim=32 的典型 latent shape=(4,8)，4x4 unfold 应得到 2 个 patches。
        if latent_h == 4 and latent_w == 8 and num_patches != 2:
            raise RuntimeError(f"Expected 2 patches for latent shape (4,8) with 4x4 unfold, got {num_patches}")

        total_samples = batch * num_patches
        all_inputs = patches.permute(0, 2, 1).reshape(total_samples, self.n_qubits)

        all_outputs = []
        # 按块处理以控制显存/内存峰值。
        chunk_size = self.chunk_size
        for start in range(0, total_samples, chunk_size):
            end = min(total_samples, start + chunk_size)
            batch_inp = all_inputs[start:end].to(self.weights_crz.device)
            q_out = self._torch_quantum_forward(batch_inp)
            all_outputs.append(q_out)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_outputs = all_outputs.reshape(batch, num_patches, self.n_qubits).permute(0, 2, 1)

        fold = nn.Fold(output_size=(latent_h, latent_w), kernel_size=self.window_size, stride=self.window_size).to(x_proc.device)
        output = fold(all_outputs).float()
        output = output.reshape(batch, dim)
        return output.to(original_device)


class CsiNetEncoder(nn.Module):
    def __init__(self, encoded_dim, leaky_slope=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(IMG_CHANNELS, 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        # Match the classical CsiNet setting (Keras LeakyReLU default slope=0.3).
        self.leaky_slope = float(leaky_slope)
        self.lr1 = nn.LeakyReLU(negative_slope=self.leaky_slope)
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(IMG_TOTAL, encoded_dim)

    def forward(self, x):
        x = self.lr1(self.bn1(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc_encode(x)
        return x


class QuantumCompensatedDecoder(nn.Module):
    def __init__(
        self,
        encoded_dim,
        main_q_chunk_size=32,
        main_entanglement_shifts=None,
        leaky_slope=0.3,
    ):
        super().__init__()
        self.fc_decode = nn.Linear(encoded_dim * 2, IMG_TOTAL)
        self.quantum_comp = QuantumCompensationBlock(
            n_qubits=16,
            n_layers=2,
            window_size=4,
            entanglement_shifts=main_entanglement_shifts,
            chunk_size=main_q_chunk_size,
        )

        # Keep decoder residual design consistent with CsiNet_train.py.
        self.leaky_slope = float(leaky_slope)
        self.residual_blocks = nn.ModuleList([self._make_residual_block(IMG_CHANNELS, leaky_slope=self.leaky_slope) for _ in range(2)])

        self.output_conv = nn.Conv2d(IMG_CHANNELS, IMG_CHANNELS, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _make_residual_block(channels, leaky_slope=0.3):
        ls = float(leaky_slope)
        return nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=ls),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=ls),
            nn.Conv2d(16, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, s):
        batch_size = s.shape[0]
        q_latent = self.quantum_comp(s)
        s_con = torch.cat([s, q_latent], dim=1)
        x = self.fc_decode(s_con)
        x = x.reshape(batch_size, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

        residual = x
        for block in self.residual_blocks:
            residual = F.leaky_relu(residual + block(residual), negative_slope=self.leaky_slope)

        out = self.sigmoid(self.output_conv(residual))
        return out

class CsiNetQuantumCompensated(nn.Module):
    def __init__(
        self,
        encoded_dim,
        main_q_chunk_size=32,
        main_entanglement_shifts=None,
        leaky_slope=0.3,
    ):
        super().__init__()
        self.encoder = CsiNetEncoder(encoded_dim, leaky_slope=leaky_slope)
        self.decoder = QuantumCompensatedDecoder(
            encoded_dim=encoded_dim,
            main_q_chunk_size=main_q_chunk_size,
            main_entanglement_shifts=main_entanglement_shifts,
            leaky_slope=leaky_slope,
        )

    def forward(self, x):
        s = self.encoder(x)
        x_hat = self.decoder(s)
        return x_hat


def load_data(data_path="/root/work/luxian/csinet/data"):
    # 所有数据统一使用 outdoor 数据集
    x_train = sio.loadmat(f"{data_path}/DATA_Htrainout.mat")["HT"].astype(np.float32)
    x_val = sio.loadmat(f"{data_path}/DATA_Hvalout.mat")["HT"].astype(np.float32)
    x_test = sio.loadmat(f"{data_path}/DATA_Htestout.mat")["HT"].astype(np.float32)
    x_test_freq = sio.loadmat(f"{data_path}/DATA_HtestFout_all.mat")["HF_all"].astype(np.complex128)

    def preprocess(data):
        bs = data.shape[0]
        return data.reshape(bs, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    x_train = preprocess(x_train)
    x_val = preprocess(x_val)
    x_test = preprocess(x_test)
    x_test_freq = x_test_freq.reshape(-1, IMG_HEIGHT, 125)

    return x_train, x_val, x_test, x_test_freq


def calculate_nmse_rho(x_test, x_hat, x_test_freq):
    batch_size = x_test.shape[0]
    x_test_real = x_test[:, 0, :, :].reshape(batch_size, -1)
    x_test_imag = x_test[:, 1, :, :].reshape(batch_size, -1)
    x_test_c = (x_test_real - 0.5) + 1j * (x_test_imag - 0.5)

    x_hat_real = x_hat[:, 0, :, :].reshape(batch_size, -1)
    x_hat_imag = x_hat[:, 1, :, :].reshape(batch_size, -1)
    x_hat_c = (x_hat_real - 0.5) + 1j * (x_hat_imag - 0.5)

    x_hat_f = x_hat_c.reshape(batch_size, IMG_HEIGHT, IMG_WIDTH)
    x_hat_full = np.fft.fft(
        np.concatenate((x_hat_f, np.zeros((batch_size, IMG_HEIGHT, 257 - IMG_WIDTH))), axis=2),
        axis=2,
    )[:, :, 0:125]

    n1 = np.sqrt(np.sum(np.abs(x_test_freq) ** 2, axis=1))
    n2 = np.sqrt(np.sum(np.abs(x_hat_full) ** 2, axis=1))
    aa = np.abs(np.sum(np.conj(x_test_freq) * x_hat_full, axis=1))
    rho = np.mean(aa / (n1 * n2 + 1e-10), axis=1)

    power = np.sum(np.abs(x_test_c) ** 2, axis=1)
    mse = np.sum(np.abs(x_test_c - x_hat_c) ** 2, axis=1)
    nmse = 10 * np.log10(np.mean(mse / (power + 1e-10)))

    return float(nmse), float(np.mean(rho))


def _move_model_devices(model, device):
    # 兼容 DDP：先拿到底层模块再做设备迁移。
    base_model = _unwrap_model(model)
    base_model.to(device)
    if hasattr(base_model, "decoder") and hasattr(base_model.decoder, "quantum_comp"):
        base_model.decoder.quantum_comp = base_model.decoder.quantum_comp.to(device)
    return model


def _amp_context(device):
    # NPU-only runtime: keep full precision and avoid CUDA-specific AMP checks.
    return nullcontext()


def _inference_context():
    # Prefer the new user-facing API for inference-only workloads.
    if hasattr(torch, "inference_mode"):
        return torch.inference_mode()
    return torch.no_grad()


class _NoOpGradScaler:
    """Fallback scaler for environments without amp GradScaler support."""

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, _state):
        return None


def _build_grad_scaler(device):
    # NPU-only runtime: disable scaler to avoid CUDA/AMP dependency.
    return _NoOpGradScaler()


def _parse_shift_list(value, default):
    if value is None:
        return tuple(default)
    text = str(value).strip()
    if not text:
        return tuple(default)
    shifts = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        shifts.append(int(token))
    shifts = [s for s in shifts if s > 0]
    return tuple(shifts) if shifts else tuple(default)


def _setup_distributed():
    # torchrun 会注入这些环境变量；单卡运行时默认回退到 world_size=1。
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="hccl")
    return distributed, world_size, rank, local_rank


def _cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process(rank):
    return int(rank) == 0


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


_RHO_DFT_CACHE = {}


def _get_rho_dft_basis(width, keep_bins, device, dtype):
    key = (int(width), int(keep_bins), str(device), str(dtype))
    basis = _RHO_DFT_CACHE.get(key)
    if basis is not None:
        return basis

    positions = torch.arange(int(width), device=device, dtype=dtype).unsqueeze(1)
    bins = torch.arange(int(keep_bins), device=device, dtype=dtype).unsqueeze(0)
    angles = 2.0 * np.pi * positions * bins / float(257)
    cos_basis = torch.cos(angles)
    sin_basis = torch.sin(angles)

    basis = (cos_basis, sin_basis)
    _RHO_DFT_CACHE[key] = basis
    return basis


def _strict_rho_torch(x_ref, x_pred, eps=1e-8, fft_size=257, keep_bins=125):
    # Match evaluation rho definition using only real-valued ops.
    # This avoids complex dtype limitations on torch_npu while keeping the same math.
    if int(fft_size) != 257:
        raise ValueError(f"_strict_rho_torch currently expects fft_size=257, got {fft_size}")

    width = x_ref.shape[-1]
    dtype = torch.float32
    device = x_ref.device

    ref_real = (x_ref[:, 0, :, :] - 0.5).to(dtype)
    ref_imag = (x_ref[:, 1, :, :] - 0.5).to(dtype)
    pred_real = (x_pred[:, 0, :, :] - 0.5).to(dtype)
    pred_imag = (x_pred[:, 1, :, :] - 0.5).to(dtype)

    cos_basis, sin_basis = _get_rho_dft_basis(width=width, keep_bins=keep_bins, device=device, dtype=dtype)

    ref_f_real = torch.matmul(ref_real, cos_basis) + torch.matmul(ref_imag, sin_basis)
    ref_f_imag = torch.matmul(ref_imag, cos_basis) - torch.matmul(ref_real, sin_basis)
    pred_f_real = torch.matmul(pred_real, cos_basis) + torch.matmul(pred_imag, sin_basis)
    pred_f_imag = torch.matmul(pred_imag, cos_basis) - torch.matmul(pred_real, sin_basis)

    n1 = torch.sqrt(torch.sum(ref_f_real * ref_f_real + ref_f_imag * ref_f_imag, dim=1) + eps)
    n2 = torch.sqrt(torch.sum(pred_f_real * pred_f_real + pred_f_imag * pred_f_imag, dim=1) + eps)

    inner_real = torch.sum(ref_f_real * pred_f_real + ref_f_imag * pred_f_imag, dim=1)
    inner_imag = torch.sum(ref_f_real * pred_f_imag - ref_f_imag * pred_f_real, dim=1)
    aa = torch.sqrt(inner_real * inner_real + inner_imag * inner_imag + eps)

    rho_per_sample = torch.mean(aa / (n1 * n2 + eps), dim=1)
    return rho_per_sample.mean()


def _build_optimizer(model, lr, quantum_lr_scale=0.35):
    base_model = _unwrap_model(model)
    quantum_lr_scale = float(np.clip(quantum_lr_scale, 1e-3, 1.0))

    quantum_names = []
    base_params = []
    quantum_params = []

    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            name.startswith("decoder.quantum_comp")
        ):
            quantum_names.append(name)
            quantum_params.append(param)
        else:
            base_params.append(param)

    if not quantum_params or not base_params:
        optimizer = torch.optim.Adam(base_model.parameters(), lr=lr)
        optimizer_groups_info = {"base_lr": float(lr), "quantum_lr": float(lr), "quantum_params": quantum_names}
        return optimizer, optimizer_groups_info

    optimizer = torch.optim.Adam(
        [
            {"params": base_params, "lr": lr},
            {"params": quantum_params, "lr": lr * quantum_lr_scale},
        ]
    )
    optimizer_groups_info = {
        "base_lr": float(lr),
        "quantum_lr": float(lr * quantum_lr_scale),
        "quantum_params": quantum_names,
    }
    return optimizer, optimizer_groups_info

def _compute_rho_weight(epoch, base_weight=0.05, warmup_epochs=5):
    base_weight = float(max(0.0, base_weight))
    warmup_epochs = int(max(0, warmup_epochs))
    if base_weight <= 0.0:
        return 0.0
    if warmup_epochs <= 0:
        return base_weight
    progress = min(1.0, float(epoch + 1) / float(warmup_epochs))
    return float(base_weight * progress)


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    x_test_np,
    x_test_freq,
    epochs,
    lr,
    device,
    best_model_path,
    latest_checkpoint_path,
    best_checkpoint_path,
    save_dir=None,
    resume_from="",
    distributed=False,
    rank=0,
    train_sampler=None,
    lr_factor=0.5,
    lr_patience=6,
    min_lr=1e-5,
    early_stop_patience=15,
    early_stop_min_delta=1e-6,
    rho_loss_weight=0.05,
    rho_loss_warmup_epochs=5,
    quantum_lr_scale=0.35,
    leaky_slope=0.3,
    train_loss_path=None,
    val_loss_path=None,
    lr_path=None,
):
    model = _move_model_devices(model, device)
    main_process = _is_main_process(rank)

    optimizer, optimizer_groups_info = _build_optimizer(model, lr=lr, quantum_lr_scale=quantum_lr_scale)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=min_lr,
    )
    criterion = nn.MSELoss()
    scaler = _build_grad_scaler(device)

    if main_process:
        print(
            "Optimizer groups: "
            f"base_lr={optimizer_groups_info['base_lr']:.6g}, "
            f"quantum_lr={optimizer_groups_info['quantum_lr']:.6g}, "
            f"quantum_param_count={len(optimizer_groups_info['quantum_params'])}",
            flush=True,
        )

    best_val_loss = float("inf")
    no_improve_epochs = 0
    start_epoch = 0
    train_losses = []
    val_losses = []
    lr_history = []

    if resume_from and str(resume_from).strip():
        resume_path = Path(resume_from).expanduser().resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            _unwrap_model(model).load_state_dict(checkpoint["model_state"])
            if "optimizer_state" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                except ValueError:
                    if main_process:
                        print("[WARN] Skip optimizer state restore due to param-group mismatch.", flush=True)
            if "scheduler_state" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            if "scaler_state" in checkpoint and hasattr(scaler, "load_state_dict"):
                scaler.load_state_dict(checkpoint["scaler_state"])

            start_epoch = int(checkpoint.get("epoch", 0))
            best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
            no_improve_epochs = int(checkpoint.get("no_improve_epochs", no_improve_epochs))
            train_losses = list(checkpoint.get("train_losses", []))
            val_losses = list(checkpoint.get("val_losses", []))
            lr_history = list(checkpoint.get("lr_history", []))
            print(
                f"Resumed full checkpoint from {resume_path} at epoch={start_epoch}, best_val_loss={best_val_loss:.6f}",
                flush=True,
            )
        else:
            # 兼容旧格式：仅模型参数。
            _unwrap_model(model).load_state_dict(checkpoint)
            print(f"Loaded model weights only from {resume_path}", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        rho_weight_epoch = _compute_rho_weight(
            epoch,
            base_weight=rho_loss_weight,
            warmup_epochs=rho_loss_warmup_epochs,
        )

        train_loss = 0.0
        num_train_batches = 0

        for (data,) in train_loader:
            data = data.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with _amp_context(device):
                output = model(data)
                mse_loss = criterion(output, data)
                if rho_weight_epoch > 0.0:
                    rho = _strict_rho_torch(data, output)
                    loss = mse_loss + rho_weight_epoch * (1.0 - rho)
                else:
                    loss = mse_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            num_train_batches += 1

        if distributed:
            loss_stats = torch.tensor([train_loss, float(num_train_batches)], device=device, dtype=torch.float32)
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
            total_loss = loss_stats[0].item()
            total_batches = max(1.0, loss_stats[1].item())
            avg_train_loss = total_loss / total_batches
        else:
            avg_train_loss = train_loss / max(1, len(train_loader))

        train_losses.append(avg_train_loss)

        avg_val_loss = float("nan")
        epoch_metrics = {}

        # 只在主进程进行验证与测试指标计算，避免每个 rank 重复评估。
        if main_process:
            model.eval()
            val_loss = 0.0
            with _inference_context():
                for (data,) in val_loader:
                    data = data.to(device, non_blocking=True)
                    with _amp_context(device):
                        output = model(data)
                        loss = criterion(output, data)
                    val_loss += loss.item()

            avg_val_loss = val_loss / max(1, len(val_loader))

            # Compute test metrics each epoch (NMSE & rho if frequency data available)
            test_outputs = []
            with _inference_context():
                for (data,) in test_loader:
                    data = data.to(device, non_blocking=True)
                    with _amp_context(device):
                        out = model(data)
                    test_outputs.append(out.float().cpu().numpy())
            x_hat = np.concatenate(test_outputs, axis=0)

            if x_test_freq is not None:
                try:
                    nmse, rho = calculate_nmse_rho(x_test_np, x_hat, x_test_freq)
                    epoch_metrics["nmse_db"] = nmse
                    epoch_metrics["rho"] = rho
                except Exception:
                    epoch_metrics["nmse_db"] = None
                    epoch_metrics["rho"] = None
            else:
                epoch_metrics["test_mse"] = float(np.mean((x_hat - x_test_np) ** 2))

        val_losses.append(avg_val_loss)

        # 每个 epoch 结束时都打印 NMSE 和 rho（无频域标签时显示 N/A）
        nmse_value = epoch_metrics.get("nmse_db", None) if main_process else None
        rho_value = epoch_metrics.get("rho", None) if main_process else None

        nmse_str = f"{nmse_value:.2f} dB" if isinstance(nmse_value, (int, float)) else "N/A"
        rho_str = f"{rho_value:.4f}" if isinstance(rho_value, (int, float)) else "N/A"

        # Log epoch summary to console and to snapshot file if provided
        current_lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]

        metrics_str = f"NMSE: {nmse_str}, Rho: {rho_str}"
        if "test_mse" in epoch_metrics:
            metrics_str += f", Test MSE: {epoch_metrics['test_mse']:.6f}"

        summary_line = (
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.6f} "
            f"Val Loss: {avg_val_loss:.6f} "
            f"LR: {current_lrs} "
            f"RhoW: {rho_weight_epoch:.4f} "
            f"{metrics_str}"
        )
        if main_process:
            print(summary_line, flush=True)
        if main_process and save_dir is not None:
            try:
                # 断点续训时避免覆盖历史日志。
                write_mode = "w" if (start_epoch == 0 and epoch == 0) else "a"
                with open(Path(save_dir) / "log_snapshot.txt", write_mode, encoding="utf-8") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                    f.write(summary_line + "\n")
                    f.write("----\n")
            except Exception:
                pass

        # Keep LR scheduler state consistent across ranks by broadcasting val loss.
        if distributed:
            if main_process:
                val_for_scheduler = torch.tensor(avg_val_loss, device=device, dtype=torch.float32)
            else:
                val_for_scheduler = torch.zeros(1, device=device, dtype=torch.float32).squeeze(0)
            dist.broadcast(val_for_scheduler, src=0)
            scheduler_metric = float(val_for_scheduler.item())
        else:
            scheduler_metric = float(avg_val_loss)

        scheduler.step(scheduler_metric)

        current_lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]
        lr_history.append(current_lrs)

        if main_process:
            try:
                if train_loss_path is not None:
                    np.savetxt(train_loss_path, np.array(train_losses, dtype=np.float32), delimiter=",")
                if val_loss_path is not None:
                    np.savetxt(val_loss_path, np.array(val_losses, dtype=np.float32), delimiter=",")
                if lr_path is not None:
                    np.savetxt(lr_path, np.array(lr_history, dtype=np.float32), delimiter=",")
            except Exception:
                pass

        stop_now = False
        if main_process:
            improved = avg_val_loss < (best_val_loss - early_stop_min_delta)
            if improved:
                best_val_loss = avg_val_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
                stop_now = True

        if main_process and avg_val_loss < (best_val_loss + 1e-12):
            unwrapped = _unwrap_model(model)
            torch.save(unwrapped.state_dict(), best_model_path)
            best_checkpoint = {
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "no_improve_epochs": no_improve_epochs,
                "model_state": unwrapped.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if hasattr(scaler, "state_dict") else {},
                "train_losses": train_losses,
                "val_losses": val_losses,
                "lr_history": lr_history,
            }
            torch.save(best_checkpoint, best_checkpoint_path)
            print(f"Saved best model (epoch {epoch+1}): {best_model_path}", flush=True)

        if main_process:
            unwrapped = _unwrap_model(model)
            # Save per-epoch model in the same directory as best model: model_epoch_x.y
            file_ext = best_model_path.suffix.lstrip(".") or "pth"
            epoch_model_path = best_model_path.parent / f"model_epoch_{epoch + 1}.{file_ext}"
            torch.save(unwrapped.state_dict(), epoch_model_path)
            latest_checkpoint = {
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "no_improve_epochs": no_improve_epochs,
                "model_state": unwrapped.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if hasattr(scaler, "state_dict") else {},
                "train_losses": train_losses,
                "val_losses": val_losses,
                "lr_history": lr_history,
            }
            torch.save(latest_checkpoint, latest_checkpoint_path)
            print(f"Saved epoch model (epoch {epoch+1}): {epoch_model_path}", flush=True)

            if stop_now:
                print(
                    f"Early stopping at epoch {epoch + 1}: no val-loss improvement for {no_improve_epochs} epochs.",
                    flush=True,
                )

        if distributed:
            stop_tensor = torch.tensor(1 if (main_process and stop_now) else 0, device=device, dtype=torch.int32)
            dist.broadcast(stop_tensor, src=0)
            stop_now = bool(stop_tensor.item())

        if stop_now:
            break

    return train_losses, val_losses, lr_history


def make_sanity_loaders(batch_size=32, train_samples=64, val_samples=32, test_samples=32):
    x_train = torch.rand(train_samples, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    x_val = torch.rand(val_samples, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    x_test = torch.rand(test_samples, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    train_loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=batch_size, shuffle=False, pin_memory=False)
    test_loader = DataLoader(TensorDataset(x_test), batch_size=batch_size, shuffle=False, pin_memory=False)
    return train_loader, val_loader, test_loader, x_test.numpy(), None


def run(args):
    # 强制使用 NPU，否则报错。
    if not HAS_TORCH_NPU:
        raise RuntimeError(
            "NPU is not available. Install torch_npu and ensure Ascend runtime is configured correctly."
        )

    distributed, world_size, rank, local_rank = _setup_distributed()
    main_process = _is_main_process(rank)

    

    # 默认不强制 world size，避免 python 直接启动时报错；可通过 --strict-world-size 开启严格校验。
    if world_size != args.expected_world_size:
        msg = (
            f"Expected WORLD_SIZE={args.expected_world_size}, but got WORLD_SIZE={world_size}. "
            f"Use torchrun --nproc_per_node={args.expected_world_size} for multi-NPU training."
        )
        if args.strict_world_size:
            raise RuntimeError(msg)
        if main_process:
            print(f"[WARN] {msg} Falling back to current WORLD_SIZE={world_size}.", flush=True)

    if hasattr(torch.npu, "set_device"):
        torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    print(f"Using device: {device}, rank={rank}, world_size={world_size}")

    model = CsiNetQuantumCompensated(
        encoded_dim=args.encoded_dim,
        main_q_chunk_size=args.q_main_chunk_size,
        main_entanglement_shifts=_parse_shift_list(args.q_main_shifts, default=(1, 2)),
        leaky_slope=args.leaky_slope,
    ).to(device)
    print(f"Quantum backend: {model.decoder.quantum_comp.backend}")

    if distributed:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    if args.sanity:
        train_loader, val_loader_tmp, test_loader_tmp, x_test_np_tmp, x_test_freq_tmp = make_sanity_loaders(
            batch_size=args.batch_size,
            train_samples=args.sanity_train_samples,
            val_samples=args.sanity_val_samples,
            test_samples=args.sanity_test_samples,
        )
        # sanity 场景不走分布式切分，保持简单。
        train_sampler = None
        val_loader = val_loader_tmp if main_process else None
        test_loader = test_loader_tmp if main_process else None
        x_test_np = x_test_np_tmp if main_process else None
        x_test_freq = x_test_freq_tmp if main_process else None
    else:
        x_train, x_val, x_test, x_test_freq_full = load_data(args.data_path)
        # Optionally subset the real datasets to requested sizes
        if getattr(args, "train_samples", 0) and args.train_samples > 0:
            x_train = x_train[: args.train_samples]
        if getattr(args, "val_samples", 0) and args.val_samples > 0:
            x_val = x_val[: args.val_samples]
        if getattr(args, "test_samples", 0) and args.test_samples > 0:
            x_test = x_test[: args.test_samples]
            x_test_freq_full = x_test_freq_full[: args.test_samples]

        x_train_t = torch.FloatTensor(x_train)
        x_val_t = torch.FloatTensor(x_val)
        x_test_t = torch.FloatTensor(x_test)

        train_dataset = TensorDataset(x_train_t)
        if distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=train_sampler,
                pin_memory=False,
            )
        else:
            train_sampler = None
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=False,
            )

        # 验证/测试只在主进程执行，避免重复评估开销。
        if main_process:
            val_loader = DataLoader(TensorDataset(x_val_t), batch_size=args.batch_size, shuffle=False, pin_memory=False)
            test_loader = DataLoader(TensorDataset(x_test_t), batch_size=args.batch_size, shuffle=False, pin_memory=False)
            x_test_np = x_test
            x_test_freq = x_test_freq_full
        else:
            val_loader = None
            test_loader = None
            x_test_np = None
            x_test_freq = None

    # Determine save directory: priority --outputdir, then deprecated --output-dir,
    # otherwise default out_100k_9c.
    out_arg = getattr(args, "outputdir", "") or getattr(args, "output_dir", "")
    if out_arg and str(out_arg).strip():
        save_dir = Path(out_arg).expanduser().resolve()
    else:
        save_dir = Path(__file__).resolve().parent / "out_100k_9c"
    save_dir.mkdir(parents=True, exist_ok=True)

    run_tag = args.run_tag.strip() if args.run_tag else ""
    if not run_tag:
        run_tag = time.strftime("%Y%m%d_%H%M%S")

    suffix = f"{args.envir}_dim{args.encoded_dim}_{run_tag}"
    best_model_path = save_dir / f"best_model_quantum_npu_{suffix}.pth"
    latest_checkpoint_path = save_dir / f"latest_checkpoint_quantum_npu_{suffix}.pth"
    best_checkpoint_path = save_dir / f"best_checkpoint_quantum_npu_{suffix}.pth"
    train_loss_path = save_dir / f"train_loss_quantum_npu_{suffix}.csv"
    val_loss_path = save_dir / f"val_loss_quantum_npu_{suffix}.csv"
    lr_path = save_dir / f"lr_history_quantum_npu_{suffix}.csv"

    try:
        start = time.time()
        train_losses, val_losses, lr_history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            x_test_np=x_test_np,
            x_test_freq=x_test_freq,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            best_model_path=best_model_path,
            latest_checkpoint_path=latest_checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            save_dir=save_dir,
            resume_from=args.resume_from,
            distributed=distributed,
            rank=rank,
            train_sampler=train_sampler,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            min_lr=args.min_lr,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            rho_loss_weight=args.rho_loss_weight,
            rho_loss_warmup_epochs=args.rho_loss_warmup_epochs,
            quantum_lr_scale=args.quantum_lr_scale,
            train_loss_path=train_loss_path,
            val_loss_path=val_loss_path,
            lr_path=lr_path,
        )
        train_time = time.time() - start
        if main_process:
            print(f"Training time: {train_time:.2f}s")

        if distributed:
            dist.barrier()

        if main_process:
            model_to_eval = _unwrap_model(model)
            if best_model_path.exists():
                model_to_eval.load_state_dict(torch.load(best_model_path, map_location=device))
            else:
                print("Best model file not found after training; using latest in-memory model for evaluation.", flush=True)
            model_to_eval = _move_model_devices(model_to_eval, device)
            model_to_eval.eval()

            outputs = []
            infer_start = time.time()
            with _inference_context():
                for (data,) in test_loader:
                    data = data.to(device, non_blocking=True)
                    output = model_to_eval(data)
                    outputs.append(output.float().cpu().numpy())
            infer_end = time.time()

            x_hat = np.concatenate(outputs, axis=0)
            inference_time_per_sample = (infer_end - infer_start) / x_hat.shape[0]
            print(f"Inference time per sample: {inference_time_per_sample:.6f}s")

            metrics = {}
            if x_test_freq is not None:
                nmse, rho = calculate_nmse_rho(x_test_np, x_hat, x_test_freq)
                metrics["nmse_db"] = nmse
                metrics["cosine_similarity"] = rho
                print(f"NMSE: {nmse:.2f} dB")
                print(f"Cosine similarity: {rho:.4f}")
            else:
                sanity_mse = float(np.mean((x_hat - x_test_np) ** 2))
                metrics["sanity_mse"] = sanity_mse
                print(f"Sanity MSE: {sanity_mse:.6f}")

            final_model_path = save_dir / f"csinet_quantum_npu_{suffix}.pth"
            torch.save(model_to_eval.state_dict(), final_model_path)
            np.savetxt(train_loss_path, train_losses, delimiter=",")
            np.savetxt(val_loss_path, val_losses, delimiter=",")
            np.savetxt(lr_path, np.array(lr_history), delimiter=",")

            summary = {
                "args": vars(args),
                "device": str(device),
                "rank": int(rank),
                "world_size": int(world_size),
                "quantum_backend": model_to_eval.decoder.quantum_comp.backend,
                "train_time_sec": float(train_time),
                "inference_time_per_sample_sec": float(inference_time_per_sample),
                "train_samples": int(len(train_loader.dataset)),
                "val_samples": int(len(val_loader.dataset)),
                "test_samples": int(len(test_loader.dataset)),
                "best_model_path": str(best_model_path),
                "latest_checkpoint_path": str(latest_checkpoint_path),
                "best_checkpoint_path": str(best_checkpoint_path),
                "final_model_path": str(final_model_path),
                "train_loss_csv": str(train_loss_path),
                "val_loss_csv": str(val_loss_path),
                "lr_history_csv": str(lr_path),
                "metrics": metrics,
            }

            summary_path = save_dir / f"run_summary_quantum_npu_{suffix}.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print(f"Saved best model: {best_model_path}")
            print(f"Saved latest checkpoint: {latest_checkpoint_path}")
            print(f"Saved best checkpoint: {best_checkpoint_path}")
            print(f"Saved final model: {final_model_path}")
            print(f"Saved train loss: {train_loss_path}")
            print(f"Saved val loss: {val_loss_path}")
            print(f"Saved lr history: {lr_path}")
            print(f"Saved run summary: {summary_path}")
    finally:
        _cleanup_distributed(distributed)


def build_parser():
    parser = argparse.ArgumentParser(description="CsiNet quantum-classical hybrid (NPU-oriented)")
    parser.add_argument("--envir", type=str, default="outdoor", choices=["outdoor"], help="data environment (outdoor only)")
    parser.add_argument("--data-path", type=str, default="/root/work/luxian/csinet/data")
    parser.add_argument("--encoded-dim", type=int, default=32, choices=sorted(set(COMPRESSION_RATES.values())))
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--expected-world-size", type=int, default=4, help="expected number of NPU processes (default: 4)")
    parser.add_argument(
        "--strict-world-size",
        action="store_true",
        help="if set, mismatch between WORLD_SIZE and --expected-world-size will raise an error",
    )
    parser.add_argument("--output-dir", type=str, default="", help="(deprecated) output directory; prefer --outputdir")
    parser.add_argument("--outputdir", type=str, default="out_100k_9c", help="output directory for saved artifacts (default: out_100k_9c)")
    parser.add_argument("--resume-from", type=str, default="", help="path to a checkpoint (.pth) for resuming training")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    parser.add_argument("--lr-patience", type=int, default=6, help="epochs with no val improvement before LR is reduced")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="minimum learning rate for scheduler")
    parser.add_argument("--quantum-lr-scale", type=float, default=0.35, help="quantum branch lr = lr * quantum-lr-scale")
    parser.add_argument("--early-stop-patience", type=int, default=15, help="epochs with no val improvement before early stop; <=0 disables")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-6, help="minimum val loss improvement to reset early-stop counter")
    parser.add_argument("--rho-loss-weight", type=float, default=0.05, help="weight for (1-rho) loss term")
    parser.add_argument("--rho-loss-warmup-epochs", type=int, default=5, help="epochs to linearly warm up rho loss weight")
    parser.add_argument("--q-main-chunk-size", type=int, default=32, help="chunk size for main quantum compensation block")
    parser.add_argument(
        "--q-main-shifts",
        type=str,
        default="1,2",
        help="comma-separated per-layer entanglement shifts (e.g., '1,2' means layer1=1-hop, layer2=2-hop)",
    )
    parser.add_argument("--train-samples", type=int, default=100000, help="number of training samples to use (default: 100000; <=0 means all)")
    parser.add_argument("--val-samples", type=int, default=30000, help="number of validation samples to use (default: 30000; <=0 means all)")
    parser.add_argument("--test-samples", type=int, default=20000, help="number of test samples to use (default: 20000; <=0 means all)")

    # 快速验证入口，避免完整数据训练耗时。
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--sanity-train-samples", type=int, default=64)
    parser.add_argument("--sanity-val-samples", type=int, default=32)
    parser.add_argument("--sanity-test-samples", type=int, default=32)
    parser.add_argument("--leaky-slope", type=float, default=0.3, help="negative slope for LeakyReLU activations (default: 0.3)")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
