import math
from nn.matrix import Matrix
from time import sleep

class NeuralNetwork(object):

    def __init__(self, i, h, o, lr):
        self.input_nodes = i
        self.hidden_nodes = h
        self.output_nodes = o
        self.learning_rate = lr

        # Create our weight matrix, input as columns, hidden as rows
        self.weights_ih = Matrix(h, i)

        # Create our second weight matrix, hidden as columns, output as rows
        self.weights_ho = Matrix(o, h)

        # Get some random weights from -1 to 1
        self.weights_ih.randomize_range(-1, 1)
        self.weights_ho.randomize_range(-1, 1)

        # Create our biases
        self.bias_h = Matrix(h, 1)
        self.bias_o = Matrix(o, 1)
        self.bias_h.randomize_range(-1, 1) # need to check the original randomize function
        self.bias_o.randomize_range(-1, 1)
    
    def __str__(self):
        s = "NeuralNetwork: input count {}, hidden count {}, output count {}, learning rate {}".format(self.input_nodes, self.hidden_nodes, self.output_nodes, self.learning_rate)
        s += "\n weights_ih {}".format(self.weights_ih)
        s += "\n weights_ho {}".format(self.weights_ho)
        s += "\n bias_h {}".format(self.bias_h)
        s += "\n bias_o {}".format(self.bias_o)
        return s

    def feedforward(self, input_matrix):
        # Calculate the dot product of the weights and inputs
        hidden_layer_output = Matrix.dot_product(self.weights_ih, input_matrix)

        # Element wise add the bias
        hidden_layer_output.elementwise_add(self.bias_h)

        # Map an activation function
        hidden_layer_output.map(NeuralNetwork.sigmoid)

        # Repeat steps for output layer
        output_layer_output = Matrix.dot_product(self.weights_ho, hidden_layer_output)
        output_layer_output.elementwise_add(self.bias_o)
        output_layer_output.map(NeuralNetwork.sigmoid)

        return output_layer_output

    def train(self, input_matrix, target_matrix):
        # RECALCULATE FEEDFORWARD TO GET OUTPUTS
        # Like we did with feedforward, we need to calculate our outputs
        # Calculate hidden layer first
        hidden_layer_output = Matrix.dot_product(self.weights_ih, input_matrix)
        hidden_layer_output.elementwise_add(self.bias_h)
        hidden_layer_output.map(NeuralNetwork.sigmoid)
        # Repeat steps for output layer
        output_layer_output = Matrix.dot_product(self.weights_ho, hidden_layer_output)
        output_layer_output.elementwise_add(self.bias_o)
        output_layer_output.map(NeuralNetwork.sigmoid)

        # OUTPUT ERROR ADJUSTMENT TO HIDDEN

        # Calculate the error from output layer
        # error = targets - outputs
        output_errors = Matrix.clone(target_matrix)
        output_errors.elementwise_subtract(output_layer_output)

        # Delta of Weights H->0 = learningRate * outputErrors * ( outputs * ( 1 - outputs ) ) * transposeHidden
        # Calculate our output gradient (Gradient Descent Algorithm (need notes on this))
        #                                                       ^^^^^^^^^ gradient ^^^^^^^^^^
        # gradient = outputs * (1 - outputs)
        # let gradient = outputLayerOutput * (1 - outputLayerOutput); // assuming sigmoid has set output between 0 and 1
        output_gradients = Matrix.clone(output_layer_output)
        output_gradients.map(NeuralNetwork.derivative) # we would use derivative_sigmoid but outputs are already sigmoid from above
        output_gradients.elementwise_multiply(output_errors)
        output_gradients.scalar_multiply(self.learning_rate)

        # Calculate our hidden -> output deltas
        transpose_hiddens = hidden_layer_output.transpose()
        weights_ho_deltas = Matrix.dot_product(output_gradients, transpose_hiddens)

        # Finally, add our error delta for our weights
        self.weights_ho.elementwise_add(weights_ho_deltas)

        # Calculate output layer bias adjustment which is just the gradient adjustment
        self.bias_o.elementwise_add(output_gradients)

        # HIDDEN ERROR ADJUSTMENT TO INPUT

        # Calculate the error from output layer
        # error = targets - hidden
        transpose_weights_ho = self.weights_ho.transpose() # this confuses me
        hidden_errors = Matrix.dot_product(transpose_weights_ho, output_errors)

        # Calculate the hidden gradient
        # Delta of Weights I->H = learningRate * hiddenErrors * ( hiddens * ( 1 - hiddens ) ) * transposeInput
        #                                                       ^^^^^^^^^ gradient ^^^^^^^^^^
        hidden_gradients = Matrix.clone(hidden_layer_output)
        hidden_gradients.map(NeuralNetwork.derivative) # already sigmoid
        hidden_gradients.elementwise_multiply(hidden_errors)
        hidden_gradients.scalar_multiply(self.learning_rate)

        # Calculate our input -> hidden deltas
        transpose_inputs = input_matrix.transpose()
        weights_ih_deltas = Matrix.dot_product(hidden_gradients, transpose_inputs)

        # Finally add our error delta for our weights
        self.weights_ih.elementwise_add(weights_ih_deltas)

        # Calculate hidden layer bias adjustment which is just the gradient adjustment
        self.bias_h.elementwise_add(hidden_gradients)
    

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(0 - x)) # need to check e^-x is math.exp

    @staticmethod
    def derivative_sigmoid(x):
        return NeuralNetwork.derivative(NeuralNetwork.sigmoid(x))
    
    @staticmethod
    def derivative(x):
        return x * (1 - x)
