OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;

cry(1.0471975511966) q[2],q[1];
ry(pi/2) q[2];
ry(1.0471975511966) q[1];
cry(1.0471975511966) q[0],q[2];
cry(1.23095941734077) q[1],q[2];
cry(pi/2) q[0],q[2];
cx q[2],q[0];
x q[0];
cx q[1],q[2];
