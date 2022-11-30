import numpy as np

class QuantumConvolutionalState():
    def __init__(self):
        self.states = None
        self.initialize_states()


    def initialize_states(self):
        if self.states is None:
            self.states = [[np.random.rand()*np.pi for _ in range(int(self.qubits//2))]]
        else:
            self.states.append([np.random.rand()*np.pi for _ in range(int(self.qubits//2))])


