import torch
import numpy as np


class EncodeUnitaryMatrix():
    def __init__(self,data):
        self.data = data

    # Check if a matrix is unitary using this function
    def is_unitary(self,m):
        return np.allclose(np.eye(len(m)), m.dot(m.T.conj()))

    def to_unitary_matrix(self):
        qm = self.toQuantumMatrix(self.data)
        qv =self.toQuantumData(self.data)
        self.data_matrix = np.array(qm)
        self.data_vector = np.array(qv)

    def toQuantumData(self,data):
        input_vec = data.copy().ravel()
        vec_len = input_vec.shape[0]
        input_matrix = np.zeros((vec_len, vec_len))
        input_matrix[0] = input_vec
        input_matrix = np.float64(input_matrix.transpose(0, 1))
        u, s, v = np.linalg.svd(input_matrix)
        output_matrix = np.dot(u, v)
        output_matrix = output_matrix
        output_data = output_matrix[0, :]
        print(output_data)
        return output_data

    def toQuantumMatrix(self,data):
        input_vec = data.flatten()
        vec_len = input_vec.shape[0]
        input_matrix = np.zeros((vec_len,vec_len))
        input_matrix[0] = input_vec
        input_matrix = input_matrix.T
        u, s, v = np.linalg.svd(input_matrix)
        output_matrix = np.dot(u, v)
        return output_matrix

    def get_data(self):
        return self.data

    def get_unitary_gate(self):
        self.to_unitary_matrix()
        return self.data_matrix

    def get_quantum_data(self):
        self.to_unitary_matrix()
        return self.data_vector

