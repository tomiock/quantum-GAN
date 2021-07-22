"""Classical Discriminator."""

import numpy as np
from typing import Optional, List, Union, Dict, Any, Callable, cast, Tuple

class DiscriminatorV2:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self._ret: Dict[str, Any] = {"loss": []}

        self.architecture = [
            {"input_dim": self.in_features, "output_dim": 64, "activation": "relu"},
            {"input_dim": 64, "output_dim": 8, "activation": "relu"},
            {"input_dim": 8, "output_dim": self.out_features, "activation": "sigmoid"},
        ]
        self.params_values = self.init_layers()

    def init_layers(self, seed=99):

        # random seed initiation
        # np.random.seed(seed)
        # number of layers in our neural network
        # parameters storage initiation
        params_values = {}

        # iteration over network layers
        for idx, layer in enumerate(self.architecture):
            # we number network layers from 1
            layer_idx = idx + 1

            # extracting the number of units in layers
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            # initiating the values of the W matrix
            # and vector b for subsequent layers
            params_values['W' + str(layer_idx)] = np.random.randn(
                layer_output_size, layer_input_size) * .1
            params_values['b' + str(layer_idx)] = np.random.randn(
                layer_output_size, 1) * .1

        return params_values

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def leaky_relu(self, z, slope=0.2):
        return np.maximum(np.zeros(np.shape(z)), z) + slope * np.minimum(
            np.zeros(np.shape(z)), z
        )

    def sigmoid_backward(self, dA, Z):
        sig = self.sigmoid(Z)
        return dA * sig * (1 - sig)

    def single_layer_forward_propagation(self, A_prev, W_curr, b_curr, activation="leaky_relu"):
        # calculation of the input value for the activation function
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        # selection of activation function
        if activation == "relu":
            activation_func = self.relu
        elif activation == "leaky_relu":
            activation_func = self.leaky_relu
        elif activation == "sigmoid":
            activation_func = self.sigmoid
        else:
            raise Exception('Non-supported activation function')

        # return of calculated activation A and the intermediate Z matrix
        return activation_func(Z_curr), Z_curr

    def forward(self, image, parameters):
        # creating a temporary memory to store the information needed for a backward step
        memory = {}
        # X vector is the activation for layer 0 
        A_curr = image.reshape(image.shape[0], 1)

        # iteration over network layers
        for idx, layer in enumerate(self.architecture):
            # we number network layers from 1
            layer_idx = idx + 1
            # transfer the activation from the previous iteration
            A_prev = A_curr

            # extraction of the activation function for the current layer
            activ_function_curr = layer["activation"]
            # extraction of W for the current layer
            W_curr = parameters["W" + str(layer_idx)]
            # extraction of b for the current layer
            b_curr = parameters["b" + str(layer_idx)]
            # calculation of activation for the current layer
            A_curr, Z_curr = self.single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

            # saving calculated values in the memory
            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr

        # return of prediction vector and a dictionary containing intermediate values
        return A_curr, memory

    def get_label(self, image, params):
        return self.forward(image, params)

    def BCE(self, prediction, target):
        return target*np.log10(prediction) + (1 - target)*np.log10(1 - prediction)

    def minimax(self, real_prediction, fake_prediction):
        return np.log10(real_prediction) + np.log10(1 - fake_prediction)

    def minimax_backward(self, real_prediction, fake_prediction):
        return (1/(real_prediction*np.log(10)) + 1/((fake_prediction - 1)*np.log(10)))
    # an auxiliary function that converts probability into class


    def update(self, params_values, grads_values, learning_rate):

        # iteration over network layers
        for layer_idx, layer in enumerate(self.architecture, 1):
            params_values["W" + str(layer_idx)] += learning_rate * grads_values["dW" + str(layer_idx)]
            params_values["b" + str(layer_idx)] += learning_rate * grads_values["db" + str(layer_idx)]

        return params_values

    def single_layer_backward_propagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
        # number of examples
        #m = A_prev.shape[1]

        # selection of activation function
        if activation == "relu":
            backward_activation_func = self.relu_backward
        elif activation == "sigmoid":
            backward_activation_func = self.sigmoid_backward
        else:
            raise Exception('Non-supported activation function')

        # calculation of the activation function derivative
        dZ_curr = backward_activation_func(dA_curr, Z_curr)

        # derivative of the matrix W
        dW_curr = np.dot(dZ_curr, A_prev.T) #/ m
        # derivative of the vector b
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) #/ m
        # derivative of the matrix A_prev
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def backward(self, real_prediction, fake_prediction, memory, params_values):
        grads_values = {}

        ## number of examples
        #target = np.array(target)
        ##m = Y.shape[1] (not on use)
        #Y = Y.reshape(target.shape)
#
        ## initiation of gradient descent algorithm
        dA_prev = self.minimax_backward(real_prediction, fake_prediction)
        #dA_prev = self.loss_backward(prediction_real, prediction_fake)

        for layer_idx_prev, layer in reversed(list(enumerate(self.architecture))):
            # we number network layers from 1
            layer_idx_curr = layer_idx_prev + 1
            # extraction of the activation function for the current layer
            activation_function_curr = layer["activation"]

            dA_curr = dA_prev

            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]

            W_curr = params_values["W" + str(layer_idx_curr)]
            b_curr = params_values["b" + str(layer_idx_curr)]

            dA_prev, dW_curr, db_curr = self.single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activation_function_curr)

            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr

        return grads_values

    def step(self, real_image, fake_image, learning_rate):

        # initiation of lists storing the history
        # of metrics calculated during the learning process
        self.cost_history = []

        # step forward
        prediction_real, cache = self.forward(real_image, self.params_values)
        prediction_fake, cache = self.forward(fake_image, self.params_values)

        print(prediction_real, prediction_fake)

        # calculating metrics and saving them in history

        # step backward - calculating gradient
        grads_values = self.backward(prediction_real, prediction_fake, cache, self.params_values)
        #grads_values_loss = self.loss_backward(prediction_real, prediction_fake, grads_values)
        # updating model stat
        self.params_values = self.update(self.params_values, grads_values, learning_rate)
        loss_final = self.minimax(prediction_real, prediction_fake)

        self._ret["loss"].append(loss_final.flatten())
        self._ret["params"] = self.params_values

        return self._ret
