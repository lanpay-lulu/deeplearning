"""
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
# Third-party libraries
import numpy as np
import time

import layer
from activation import CrossEntropyCost, QuadraticCost, SoftmaxCost
from activation import TanhActivation, SigmoidActivation


random.seed(10)
np.random.seed(10)

class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost, act=SigmoidActivation):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
        self.activ_fn = act
        # line zip is to generate a full connected network weights
        
        self.layers = [layer.Layer(w, b, act) for w, b in zip(self.weights[:-1], self.biases[:-1])]
        
        ' output can be configured by output config string '
        #self.output_layer = layer.OutputLayer(self.weights[-1], self.biases[-1], TanhActivation, SoftmaxCost)
        self.output_layer = layer.OutputLayer(self.weights[-1], self.biases[-1], None, SoftmaxCost)
        

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])] 
        
    # Xavier initialization
    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1)/2 for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) / 2
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])] 

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for lay in self.layers:
            a = lay.forward(a)
        a = self.output_layer.forward(a)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data = None,
            monitor_evaluation_cost = False,
            monitor_evaluation_accuracy = False,
            monitor_training_cost = False,
            monitor_training_accuracy = False):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):
            t1 = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                        mini_batch, eta, lmbda, len(training_data))
            print "Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print "Accuracy on training data: {} / {}".format(accuracy,n)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data: {} / {}".format(
                        self.accuracy(evaluation_data), n_data)

            t2 = time.time() - t1
            print 'time =', t2

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy 

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # wsh note: nabla_w has the same structure as self.weights, representing gradient for each node. So does nabla_b.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        nabla_b1 = [np.zeros(b.shape) for b in self.biases]
        nabla_w1 = [np.zeros(w.shape) for w in self.weights]
        
        delta_nabla_b, delta_nabla_w = self.full_backprop(mini_batch, eta, lmbda, n)


    def full_backprop(self, mini_batch, eta, lmbda, total_n):
        """mx and my are vectors.
        Return a tuple of (nabla_a, nabla_w).
        """
        n = len(mini_batch) # number of samples in this batch
        xsize = mini_batch[0][0].shape[0] # x feature size
        ysize = mini_batch[0][1].shape[0] # y feature size
        mx = np.zeros((xsize,n))
        my = np.zeros((ysize,n))
        i = 0
        
        # cut batch to two matrix, matrix x and matrix y
        for x, y in mini_batch:
            mx[:,i] = x[:,0]
            my[:,i] = y[:,0]
            i += 1
        
        #print(np.sum(mx>0))
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # feed forward
        a = self.feedforward(mx)

        # backward pass
        nabla_b, nabla_w, delta = self.output_layer.backward(my)  
        self.output_layer.update(nabla_b, nabla_w, eta, lmbda, n, total_n)

        for l in xrange(1, len(self.layers)+1):
            lay = self.layers[-l]
            nabla_b, nabla_w, delta = lay.backward(delta)
            lay.update(nabla_b, nabla_w, eta, lmbda, n, total_n)
            
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        if convert:
            result = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x,y) in data]
        else:
            result = [(np.argmax(self.feedforward(x)), y)
                        for (x,y) in data]
        return sum(int(x==y) for (x,y) in result)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x,y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.cost(a, y)
        cost /= len(data)
        cost += 0.5*(lmbda/len(data))*sum(
                np.linalg.norm(w)**2 for w in self.weights)
        return cost


    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def vectorized_result(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def test():
    print 'test'
    net = Network([3, 30, 10], 
            cost=SoftmaxCost, 
            act=TanhActivation
            #act=SigmoidActivation
            )
    x = np.array([[1,2,3], [2,3,4]]).transpose()
    a1 = net.feedforward(x)
   
if __name__ == '__main__':
    test()
