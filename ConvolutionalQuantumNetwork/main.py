"""

We need to now implement a data bank of randomly initialized states parameterized by an initial value of:
Number of states
Number of qubits to initialize these states
Then we have a dictionary of states, and then each state is comprised of a dictionary of parameters to apply to an RY-CRY-RY gate grouping on each qubit. Could just use a list i suppose as the order is the same.
Then this is loaded onto a circuit along side the unitary data.

random states for qubits 1-4 initialized between 0->pi
swap test: 0 ancilla compares phi, psi where 1/5, 2/6, 3/7, 4/8


e.g:
Qubit 0 is an anicilla qubit for the swap test,
Qubit 1,2,3,4 is the random state with RY-CRY-RY gates
Qubit 5,6,7,8 is the data loaded
apply H gate to Qubit 0, then CSWAP Qubit 1-5 2-6 3-7 4-8 controlled on Qubit 0, then H gate on qubit 0
Measure Qubit 0 and return Expected value of Qubit 0


"""
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import qiskit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from data_preprocessing import ToConvolutionalMatrices
import torch
import torch.nn
import tqdm
import torchvision
import sys
import time
from QuantumEncoder import QuantumEncoder,QuantumEncoderBank
from QuantumState import QuantumConvolutionalStates,QuantumConvolutionalState
from unitary_data_encoder import EncodeUnitaryMatrix
from qiskit.providers.aer import AerError
from qiskit import Aer
from NeuralNet import NeuralNet
stride = 1
batch_size = 1
num_workers = 1
ori_img_size = 28
SUBSAMPLE_SIZE = 200
n_layers = 3
q_count = 4
simulator = Aer.get_backend('statevector_simulator')
try:
    simulator_gpu = Aer.get_backend('qasm_simulator')
    simulator_gpu.set_options(device='GPU')
except AerError as e:
    print(e)
print(simulator_gpu)

transform = transforms.Compose([transforms.Resize((ori_img_size, ori_img_size)),
                                transforms.ToTensor()])
# Path to MNIST Dataset
# (train_data,train_labels),(test_data,test_labels) = tf.keras.datasets.mnist.load_data()
# the data is not normalized, and needs to be converted to a np array...
train_data = MNIST('data', train=True, download=False)
train_labels = train_data.targets

# converting to np array and normalizing by val/max(arr)
train_data = train_data.data.numpy()
train_data = train_data / np.max(train_data)
# grab first 100 images
train_labels = train_labels.data.numpy()
valid_labels = np.logical_or(train_labels==3,train_labels==6)
train_data = train_data[valid_labels][:SUBSAMPLE_SIZE]
train_labels = train_labels[valid_labels][:SUBSAMPLE_SIZE]
train_labels = train_labels == 3 # S
print(train_data.shape)
print(train_labels)

#---- BASIC IMPLEMENTATION OF SMALL NEURAL NETWORK ENDING -----

temp_net = NeuralNet(49,1)
alpha = 0.1
# -------------------------------------------------------------

if __name__ == '__main__':
    gradients = np.empty((n_layers,q_count),dtype=list)
    costs = []
    ConvStates = QuantumConvolutionalStates(1, n_layers, q_count)
    for i,data_point in enumerate(train_data):
        data = ToConvolutionalMatrices(data_point, stride=4, filter_shape=(4, 4))
        conv_data_matrix, conv_data_state_vector = data.get_unitary_convolutional_data()
        #FEEDFORWARD OF NETWORK
        basic_output = QuantumEncoderBank(conv_data_matrix, ConvStates.states[0].get_states()).induceLayer(simulator_gpu,7500)
        basic_output = basic_output.reshape(1,basic_output.shape[0]*basic_output.shape[1])
        #BACKPROPAGATE THROUGH NETWORK
        grads,cost = temp_net.backpropagate(basic_output,train_labels[i])
        costs.append(cost)
        start = time.time()
        #GRADS WILL APPLY THE GRADIENTS TO THE WEIGHTS, THEN RETURN A VECTOR OF dCost/dA(l-1) FOR US TO APPLY TO THE QUANTUM STATES
        for layer in range(n_layers):
            for gate in range(q_count):
                fwd,bck = ConvStates.states[0].differentiate(layer,gate)
                fwd_circuits = QuantumEncoderBank(conv_data_matrix, fwd, simulator_gpu)
                bck_circuits = QuantumEncoderBank(conv_data_matrix, bck, simulator_gpu)
                fwd_prob = fwd_circuits.induceLayer(simulator_gpu,7500)
                bck_prob  = bck_circuits.induceLayer(simulator_gpu,7500)
                gradient = np.mean(fwd_prob-bck_prob)
                target = np.mean(grads)
                update = -alpha*target*gradient
                ConvStates.states[0].update_state(layer,gate,update)
                if gradients[layer,gate] is None:
                    gradients[layer,gate] = []
                gradients[layer,gate].append(gradient)
                #print(f"Gradient of neuron: {target} \nGradient of Gate {layer}-{gate}: {gradient}\nUpdated Theta by {update}")
        print(f"Data Point {i} Took {time.time()-start}s to Process")
        #print(costs)

