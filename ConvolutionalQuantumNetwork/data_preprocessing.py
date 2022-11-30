import torch
import numpy as np

class ToConvolutionalMatrices():
    def __init__(self,data,stride,filter_shape):
        output_shape = ((data.shape[0]-filter_shape[0])/stride)+1
        output_shape = int(output_shape)
        mat_sq = np.zeros([output_shape,output_shape, filter_shape[0], filter_shape[1]])
        y_index = 0
        for i in range(0, data.shape[1]-filter_shape[1], stride):
            x_index = 0
            for j in range(0, data.shape[1]-filter_shape[1], stride):
                cube = data[i:i + 4, j:j + 4]
                mat_sq[x_index, y_index] = cube
                x_index+=1
            y_index+=1
        self.conv_data = mat_sq
        self.conv_data = self.conv_data/np.max(self.conv_data)
        self.filter_shape = filter_shape
        self.conv_data_unitary_matrix = np.zeros([output_shape,output_shape, 2**filter_shape[0],2**filter_shape[1]])
        self.conv_data_unitary_data = np.zeros([output_shape,output_shape, filter_shape[0],filter_shape[1]])

    def is_unitary(self,m):
        return np.allclose(np.eye(len(m)), m.dot(m.T.conj()))

    def to_unitary_matrix(self):
        for i in range(self.conv_data.shape[0]):
            for j in range(self.conv_data.shape[1]):
                temporary_convolutional_matrix = self.conv_data[i,j].view(type=np.matrix)
                temporary_convolutional_matrix = temporary_convolutional_matrix
                qm = self.toQuantumMatrix(temporary_convolutional_matrix)
                qv =self.toQuantumData(temporary_convolutional_matrix)
                self.conv_data_unitary_matrix[i,j] = np.array(qm)
                self.conv_data_unitary_data[i,j] = np.array(qv)

    def toQuantumData(self,data):
        input_vec = data.copy().ravel()
        vec_len = input_vec.shape[1]
        input_matrix = np.zeros((vec_len, vec_len))
        input_matrix[0] = input_vec
        input_matrix = np.float64(input_matrix.transpose(0, 1))
        u, s, v = np.linalg.svd(input_matrix)
        output_matrix = np.dot(u, v)
        output_matrix = output_matrix
        output_data = output_matrix[0, :].reshape(1,self.filter_shape[0],self.filter_shape[1] )
        return output_data

    def toQuantumMatrix(self,data):
        input_vec = data.flatten()
        vec_len = input_vec.shape[1]
        input_matrix = np.zeros((vec_len,vec_len))
        input_matrix[0] = input_vec
        input_matrix = input_matrix.T
        u, s, v = np.linalg.svd(input_matrix)
        output_matrix = np.dot(u, v)
        #output_matrix = output_matrix[:,0]
        #output_matrix = output_matrix.reshape((self.filter_shape[0],self.filter_shape[1]))
        return output_matrix

    def get_convolutional_data(self):
        return self.conv_data

    def get_unitary_convolutional_data(self):
        self.to_unitary_matrix()
        return (self.conv_data_unitary_matrix,self.conv_data_unitary_data)


