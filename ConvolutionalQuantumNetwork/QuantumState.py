import numpy as np
import copy

class QuantumConvolutionalStates():
    def __init__(self,n,layer_count,qubit_count):
        self.n_states = n
        self.qubit_count = qubit_count
        self.layer_count = layer_count
        self.states = None
        self.initialize_states()

    def initialize_states(self):
        self.states = [QuantumConvolutionalState(self.layer_count,self.qubit_count) for _ in range(self.n_states)]

    def return_states(self):
        return self.states


class QuantumConvolutionalState():
    def __init__(self, layer_count, qubit_count):
        self.state = None
        self.layer_count = layer_count
        self.qubit_count = qubit_count
        self.initialize_random_state()

    def initialize_random_state(self):
        self.state = [[np.random.rand()*np.pi for _ in range(self.qubit_count)] for _ in range(self.layer_count)]

    def get_states(self):
        return self.state

    def update_state(self,layer_index,gate_index,update_value):
        self.state[layer_index][gate_index] += update_value

    def fwd_diff(self,layer_index,gate_index):
        temporary_list = copy.deepcopy(self.state)
        temporary_list[layer_index][gate_index] += 0.5*np.pi
        return temporary_list

    def bck_diff(self,layer_index,gate_index):
        temporary_list = copy.deepcopy(self.state)
        temporary_list[layer_index][gate_index] -= 0.5*np.pi
        return temporary_list

    def differentiate(self, layer_index, gate_index):
        return self.bck_diff(layer_index, gate_index), self.fwd_diff(layer_index, gate_index)
