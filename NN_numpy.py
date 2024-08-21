# before I start learning NN in tensorflor/pytorch, I wanna see exactly what kind of math magic is happening in NN,
# in this project I must use NN to solve iris classification dataset and the only rule is that the only library 
# I can use is Numpy (sklearn for dataset)
# this project can be useful for someone like me, who starts with NNs and wants to see how they work under the hood

from sklearn.datasets import load_iris
import numpy as np

data_iris = load_iris()
X = data_iris.data
y = data_iris.target

# dataset has 4 variables and 3 outcomes, so my neural network will have 4 input neurons, 2 hidden layers, 
# each with 5 neurons and 3 outcome neurons

class NeuralNetwork:
    def __init__(self, data, outcome):
        self.data = data
        self.outcome = outcome

        self.weights1 = np.random.rand(4, 5)  # I initialize random weights and biases
        self.biases1 = np.random.rand(5)      # weights are MxN matrices with random numbers, biases are vectors, 1 for each vector
        self.weights2 = np.random.rand(5, 5)  # each row is for a neuron so when going from 4 neurons to 5 neurons, I have 4x5 matrix as in each row
        self.biases2 = np.random.rand(5)      # there are 5 weights for each neuron from the next layer
        self.weights3 = np.random.rand(5, 3)        

    # firstly I define activation functions, I choose ReLU, with an option to do derivative for backprop, same for Softmax
    def ReLU(self, x, deriv=False):
        if deriv ==  True:
           return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

    def Softmax(self, x, deriv=False):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        softmax = exps / np.sum(exps, axis=-1, keepdims=True)
        if deriv:
            return softmax * (1 - softmax)
        return softmax

    def forward_pass_init(self, inputs):
        # now we can take 4 inputs as a vector 1x4 and weights as a 4x5 matrix and do matrix multiplication and we add bias
        # on other words each row in weights matrix are weights for 5 neurons in second layer, by this multiplication, 
        # we do another vector which has values of second layer, then we repeat the process with 1x5 multiplied by 5x5 and finally into output layer 
        self.hidden1 = self.ReLU(np.dot(inputs, self.weights1) + self.biases1)
        self.hidden2 = self.ReLU(np.dot(self.hidden1, self.weights2) + self.biases2)
        self.output = self.Softmax(np.dot(self.hidden2, self.weights3))
        return self.output
    
    def backward_pass(self, inputs, expected_output, learning_rate):
        # with backprop firstly we calculate the error of our predicted outputs, the purpose of the derivative is to measure
        # which neuron had the biggest impact and to "repair" that particular neuron more than other 
        output_error = expected_output - self.output
        output_delta = output_error * self.Softmax(self.output, deriv = True)

        # now output_delta is a vector 1x3 and in NN we go backward, so we have to transpose matrices such as self.weights3 
        # to multiply vector 1x3 and 3x5 matrix, what helps me remember which matrix to transpose is that we always go "into"
        # the weights matrix from the left (rows) and exit matrix on top (columns), such as go to 3 and exit to 5 neurons in second hidden layer
        hidden2_error = np.dot(output_delta, self.weights3.T)
        hidden2_delta = hidden2_error * self.ReLU(self.hidden2, deriv = True)

        hidden1_error = np.dot(hidden2_delta, self.weights2.T)
        hidden1_delta = hidden1_error * self.ReLU(self.hidden1, deriv = True)

        # now we update weights with the errors, as output delta is vector with errors 1x3, we have to transpose 5x3 to 3x5
        
        self.weights3 += learning_rate * np.dot(self.hidden2.T, output_delta)
        self.weights2 += learning_rate * np.dot(self.hidden1.T, hidden2_delta)
        self.biases2 += learning_rate * np.sum(hidden2_delta, axis=0)
        self.weights1 += learning_rate * np.dot(inputs.T, hidden1_delta)
        self.biases1 += learning_rate * np.sum(hidden1_delta, axis=0)

        # to put all pieces together and train the network we make a train function, iterate over the dataset
    def train(self, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(len(self.data)):
                self.forward_pass_init(self.data[i].reshape(1, -1))
                
                target = np.zeros((1, 3))
                target[0, self.outcome[i]] = 1
                
                self.backward_pass(self.data[i].reshape(1, -1), target, learning_rate)
            # to look at our progress in training we print accuracy of our model each 100th iteration
            if epoch % 100 == 0:
                predictions = np.argmax(self.forward_pass_init(self.data), axis=1)
                loss = round(np.mean(np.square(self.outcome - predictions)), 3)
                accuracy = round(np.mean(predictions == self.outcome), 3)
                print(f'Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}')

# Instantiate and train the neural network
nn = NeuralNetwork(X, y)
nn.train(epochs=1000, learning_rate=0.01)

# i can get to around 0.96 - 0.98 accuracy, the number often oscilated around 0.967 which is are 5 wrongly 
# classified samples out of 150, it seems that those 5 have some unique hard to recognize numbers
