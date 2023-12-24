import numpy as np
import scipy.linalg as la

################################################################################
## Create 9-qubit QCNN circuit from Cong et al.
################################################################################

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister, QuantumCircuit, ParameterVector, Parameter, ParameterVector, Gate
from qiskit.circuit.library import RYGate, RZGate

# Define unitary operator (L and R) into gate
def two_qubit_unitary_U(ParamVector):
    qc = QuantumCircuit(2)
    qc.u(theta=ParamVector[0], phi=ParamVector[1], lam=ParamVector[2], qubit=0)
    qc.u(theta=ParamVector[3], phi=ParamVector[4], lam=ParamVector[5], qubit=1)
    qc.cx(control_qubit=0, target_qubit=1)
    qc.ry(theta=ParamVector[6], qubit=0)
    qc.ry(theta=ParamVector[7], qubit=1)
    qc.cx(control_qubit=0, target_qubit=1)
    qc.u(theta=ParamVector[8], phi=ParamVector[9], lam=ParamVector[10], qubit=0)
    qc.u(theta=ParamVector[11], phi=ParamVector[12], lam=ParamVector[13], qubit=1)
    qc = qc.to_gate(label='2-qubit Unitary')
    return qc

# Define unitary operator (L and R) into gate
def two_qubit_unitary_U_endpoint(ParamVector):
    qc = QuantumCircuit(2)
    qc.u(theta=ParamVector[0], phi=ParamVector[1], lam=ParamVector[2], qubit=0)
    qc.u(theta=ParamVector[3], phi=ParamVector[4], lam=ParamVector[5], qubit=1)
    qc.cx(control_qubit=0, target_qubit=1)
    qc.ry(theta=ParamVector[6], qubit=0)
    qc.ry(theta=ParamVector[7], qubit=1)
    qc.cx(control_qubit=1, target_qubit=0)
    qc.ry(theta=ParamVector[8], qubit=1)
    qc.cx(control_qubit=0, target_qubit=1)
    qc.u(theta=ParamVector[9], phi=ParamVector[10], lam=ParamVector[11], qubit=0)
    qc.u(theta=ParamVector[12], phi=ParamVector[13], lam=ParamVector[14], qubit=1)
    qc = qc.to_gate(label='2-qubit Unitary endpoint')
    return qc

# Define unitary operator (L and R) into gate
def uniformly_control_ry(ParamVector):    
    qc = QuantumCircuit(3)
    MC_RYgate_00 = RYGate(theta=ParamVector[0]).control(num_ctrl_qubits=2, label="00", ctrl_state='00')
    MC_RYgate_01 = RYGate(theta=ParamVector[1]).control(num_ctrl_qubits=2, label="01", ctrl_state='01')
    MC_RYgate_10 = RYGate(theta=ParamVector[2]).control(num_ctrl_qubits=2, label="10", ctrl_state='10')
    MC_RYgate_11 = RYGate(theta=ParamVector[3]).control(num_ctrl_qubits=2, label="11", ctrl_state='11')
    qc.append(MC_RYgate_00, [1,2,0])
    qc.append(MC_RYgate_01, [1,2,0])
    qc.append(MC_RYgate_10, [1,2,0])
    qc.append(MC_RYgate_11, [1,2,0])
    qc = qc.to_gate(label='Uniformly Controlled RY')
    return qc

# Define unitary operator (L and R) into gate
def uniformly_control_rz(ParamVector):    
    qc = QuantumCircuit(3)
    MC_RZgate_00 = RZGate(phi=ParamVector[0]).control(num_ctrl_qubits=2, label="00", ctrl_state='00')
    MC_RZgate_01 = RZGate(phi=ParamVector[1]).control(num_ctrl_qubits=2, label="01", ctrl_state='01')
    MC_RZgate_10 = RZGate(phi=ParamVector[2]).control(num_ctrl_qubits=2, label="10", ctrl_state='10')
    MC_RZgate_11 = RZGate(phi=ParamVector[3]).control(num_ctrl_qubits=2, label="11", ctrl_state='11')
    qc.append(MC_RZgate_00, [1,2,0])
    qc.append(MC_RZgate_01, [1,2,0])
    qc.append(MC_RZgate_10, [1,2,0])
    qc.append(MC_RZgate_11, [1,2,0])
    qc = qc.to_gate(label='Uniformly Controlled RZ')
    return qc

# Define unitary operator (L and R) into gate
def three_qubit_unitary_decomposition(ParamVector):    
    qc = QuantumCircuit(3)
    qc.append(two_qubit_unitary_U(ParamVector[0:14]), qargs = [1,2])
    qc.append(uniformly_control_rz(ParamVector[14:18]), qargs = [0,1,2])
    qc.append(two_qubit_unitary_U(ParamVector[18:32]), qargs = [1,2])
    qc.append(uniformly_control_ry(ParamVector[32:36]), qargs = [0,1,2])
    qc.append(two_qubit_unitary_U(ParamVector[36:50]), qargs = [1,2])
    qc.append(uniformly_control_rz(ParamVector[50:54]), qargs = [0,1,2])
    qc.append(two_qubit_unitary_U_endpoint(ParamVector[54:69]), qargs = [1,2])

    qc = qc.to_gate(label='3-qubit Unitary')
    return qc

# Convolutional filter - SU(4)
def conv_SU4(ParamVector):
    qc = QuantumCircuit(2)
    qc.u(theta=ParamVector[0], phi=ParamVector[1], lam=ParamVector[2], qubit=0)
    qc.u(theta=ParamVector[3], phi=ParamVector[4], lam=ParamVector[5], qubit=1)
    qc.cx(control_qubit=0, target_qubit=1)
    qc.ry(theta=ParamVector[6], qubit=0)
    qc.rz(phi=ParamVector[7], qubit=1)
    qc.cx(control_qubit=1, target_qubit=0)
    qc.ry(theta=ParamVector[8], qubit=0)
    qc.cx(control_qubit=0, target_qubit=1)
    qc.u(theta=ParamVector[9], phi=ParamVector[10], lam=ParamVector[11], qubit=0)
    qc.u(theta=ParamVector[12], phi=ParamVector[13], lam=ParamVector[14], qubit=1)
    qc = qc.to_gate(label="conv_SU4")
    return qc

# Convolution block for layer: U1
def conv_U1_block(ParamVector):
    qc = QuantumCircuit(4)
    qc.append(conv_SU4(ParamVector[0:15]), qargs = [0,1])
    qc.append(conv_SU4(ParamVector[0:15]), qargs = [2,3])
    qc.append(conv_SU4(ParamVector[0:15]), qargs = [0,3])
    qc.append(conv_SU4(ParamVector[0:15]), qargs = [0,2])
    qc.append(conv_SU4(ParamVector[0:15]), qargs = [1,3])
    qc.append(conv_SU4(ParamVector[0:15]), qargs = [1,2])
    qc = qc.to_gate(label='U1')
    return qc

# Convolution block for layer: U2
def conv_U2_block(ParamVector):
    qc = QuantumCircuit(3)
    qc.append(three_qubit_unitary_decomposition(ParamVector[0:69]), qargs = [0,1,2])
    qc = qc.to_gate(label='U2')
    return qc

# Convolution block for layer: U3
def conv_U3_block(ParamVector):
    qc = QuantumCircuit(3)
    qc.append(three_qubit_unitary_decomposition(ParamVector[0:69]), qargs = [0,1,2])
    qc = qc.to_gate(label='U3')
    return qc

# Convolution block for layer: U4
def conv_U4_block(ParamVector):
    qc = QuantumCircuit(3)
    qc.append(three_qubit_unitary_decomposition(ParamVector[0:69]), qargs = [0,1,2])
    qc = qc.to_gate(label='U4')
    return qc

# Pooling control-gate for layer: Pooling
def Pooling_gate(ParamVector):
    qc = QuantumCircuit(1)
    qc.u(theta=ParamVector[0], phi=ParamVector[1], lam=ParamVector[2], qubit=0)
    qc = qc.to_gate(label='P1').control(num_ctrl_qubits=1, label=None, ctrl_state=None)
    return qc

# Pooling block for layer: Pooling
def Pooling_block(ParamVector):
    qc = QuantumCircuit(3)
    qc.h([0,2])
    qc.append(Pooling_gate(ParamVector[0:3]), qargs = [0,1])
    qc.append(Pooling_gate(ParamVector[3:6]), qargs = [2,1])
    qc = qc.to_gate(label='Pooling')
    return qc

# Fully connected block for layer: FC
def Fully_Connected_block(ParamVector, qubits):
    if qubits==3:
        qc = QuantumCircuit(qubits)
        # Same as 'three_qubit_unitary_decomposition'
        qc.append(two_qubit_unitary_U(ParamVector[0:14]), qargs = [1,2])
        qc.append(uniformly_control_rz(ParamVector[14:18]), qargs = [0,1,2])
        qc.append(two_qubit_unitary_U(ParamVector[18:32]), qargs = [1,2])
        qc.append(uniformly_control_ry(ParamVector[32:36]), qargs = [0,1,2])
        qc.append(two_qubit_unitary_U(ParamVector[36:50]), qargs = [1,2])
        qc.append(uniformly_control_rz(ParamVector[50:54]), qargs = [0,1,2])
        qc.append(two_qubit_unitary_U_endpoint(ParamVector[54:69]), qargs = [1,2])
        qc = qc.to_gate(label='FC-3qubit')
    elif qubits==2:
        qc = QuantumCircuit(qubits)
        # Same as 'two_qubit_unitary_U'
        qc.u(theta=ParamVector[0], phi=ParamVector[1], lam=ParamVector[2], qubit=0)
        qc.u(theta=ParamVector[3], phi=ParamVector[4], lam=ParamVector[5], qubit=1)
        qc.cx(control_qubit=0, target_qubit=1)
        qc.ry(theta=ParamVector[6], qubit=0)
        qc.ry(theta=ParamVector[7], qubit=1)
        qc.cx(control_qubit=0, target_qubit=1)
        qc.u(theta=ParamVector[8], phi=ParamVector[9], lam=ParamVector[10], qubit=0)
        qc.u(theta=ParamVector[11], phi=ParamVector[12], lam=ParamVector[13], qubit=1)
        qc = qc.to_gate(label='FC-2qubit')
    return qc

# Assign display styles of the gates in a circuit.draw
circuit_draw_style = style={'displaycolor': {'conv_SU4': ('#0196fe', '#000000'),
                                             'U1': ('#0196fe', '#000000'),
                                             'U2': ('#0196fe', '#000000'),
                                             'U3': ('#0196fe', '#000000'),
                                             'U4': ('#0196fe', '#000000'),
                                             'P1': ('#fe7e78', '#000000'),
                                             'Pooling': ('#fe7e78', '#000000'),
                                             'FC': ('#b587f7', '#000000')},
                            'creglinestyle': 'dashdot'}

# Build QCNN_Cong_ansatz
def QCNN_Cong_qubit_ansatz(num_qubits, traditional_qcnn=False):
    if num_qubits==9:
        # Create classical and quantum registers
        if traditional_qcnn:
            creg = ClassicalRegister(size=1, name='c')
        else:
            creg = ClassicalRegister(size=num_qubits, name='c')

        qreg = QuantumRegister(size=num_qubits, name='q')

        # Create circuit parameter vectors
        params_c1 = ParameterVector(name='θc1', length=2**(2**2)-1)
        params_c2 = ParameterVector(name='θc2', length=69)
        params_c3 = ParameterVector(name='θc3', length=69)
        params_c4 = ParameterVector(name='θc4', length=69)
        params_p = ParameterVector(name='θp', length=(2**(2**1)-1)*2)
        params_f = ParameterVector(name='θf', length=69)

        # Build QCNN_Cong circuit -- 9-qubit case
        ansatz = QuantumCircuit(qreg, creg)

        # Convolution: C1 layer
        ansatz.append(conv_U1_block(params_c1), qargs=[0,1,2,3])
        ansatz.append(conv_U1_block(params_c1), qargs=[2,3,4,5])
        ansatz.append(conv_U1_block(params_c1), qargs=[5,6,7,8])
        ansatz.barrier()
        # Convolution: C2 layer
        ansatz.append(conv_U2_block(params_c2), qargs=[0,1,2])
        ansatz.append(conv_U2_block(params_c2), qargs=[3,4,5])
        ansatz.append(conv_U2_block(params_c2), qargs=[6,7,8])
        ansatz.barrier()
        # Convolution: C3 layer
        ansatz.append(conv_U3_block(params_c3), qargs=[2,3,4])
        ansatz.append(conv_U3_block(params_c3), qargs=[5,6,7])
        ansatz.barrier()
        # Convolution: C4 layer
        ansatz.append(conv_U4_block(params_c4), qargs=[1,2,3])
        ansatz.append(conv_U4_block(params_c4), qargs=[4,5,6])
        ansatz.barrier()
        # Pooling: P layer
        ansatz.append(Pooling_block(params_p), qargs=[0,1,2])
        ansatz.append(Pooling_block(params_p), qargs=[3,4,5])
        ansatz.append(Pooling_block(params_p), qargs=[6,7,8])
        ansatz.barrier()
        
        # Fully Connected: FC layer
        ansatz.append(Fully_Connected_block(params_f, qubits=3), qargs=[1,4,7])
        
        # # Measure the remaining, trash qubits
        # if traditional_qcnn:
        #     ansatz.barrier()
        # else:
        #     ansatz.h([0,2,3,5,6,8])
        #     ansatz.barrier()
                
        # # Measure
        # if traditional_qcnn:
        #     ansatz.h(4)
        #     ansatz.barrier()
                        
        # else:
        #     ansatz.h([1,4,7])
        #     ansatz.barrier()
    
    if num_qubits==8:
        # Create classical and quantum registers
        if traditional_qcnn:
            creg = ClassicalRegister(size=1, name='c')
        else:
            creg = ClassicalRegister(size=num_qubits, name='c')

        qreg = QuantumRegister(size=num_qubits, name='q')

        # Create circuit parameter vectors
        params_c1 = ParameterVector(name='θc1', length=2**(2**2)-1)
        params_c2 = ParameterVector(name='θc2', length=69)
        params_c3 = ParameterVector(name='θc3', length=69)
        params_c4 = ParameterVector(name='θc4', length=69)
        params_p = ParameterVector(name='θp', length=(2**(2**1)-1)*2)
        params_f = ParameterVector(name='θf', length=14)

        # Build QCNN_Cong circuit -- 8-qubit case
        ansatz = QuantumCircuit(qreg, creg)

        # Convolution: C1 layer
        ansatz.append(conv_U1_block(params_c1), qargs=[2,3,4,5])
        ansatz.append(conv_U1_block(params_c1), qargs=[4,5,6,7])
        ansatz.barrier()
        # Convolution: C2 layer
        ansatz.append(conv_U2_block(params_c2), qargs=[2,3,4])
        ansatz.append(conv_U2_block(params_c2), qargs=[5,6,7])
        ansatz.barrier()
        # Convolution: C3 layer
        ansatz.append(conv_U3_block(params_c3), qargs=[1,2,3])
        ansatz.append(conv_U3_block(params_c3), qargs=[4,5,6])
        ansatz.barrier()
        # Convolution: C4 layer
        ansatz.append(conv_U4_block(params_c4), qargs=[0,1,2])
        ansatz.append(conv_U4_block(params_c4), qargs=[3,4,5])
        ansatz.barrier()
        # Pooling: P layer
        ansatz.append(Pooling_block(params_p), qargs=[2,3,4])
        ansatz.append(Pooling_block(params_p), qargs=[5,6,7])
        ansatz.barrier()
        
        # Fully Connected: FC layer
        ansatz.append(Fully_Connected_block(params_f, qubits=2), qargs=[3,6])

        # # Measure the remaining, trash qubits
        # if traditional_qcnn:
        #     ansatz.barrier()
        # else:
        #     ansatz.h([0,1,2,4,5,7])
        #     ansatz.barrier()
                
        # # Measure
        # if traditional_qcnn:
        #     ansatz.h(6)
        #     ansatz.barrier()
                        
        # else:
        #     ansatz.h([3,6])
        #     ansatz.barrier()
    
    return ansatz
    
    