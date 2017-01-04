import mnist_loader

import network
import activation

# add softmax to the output layer

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 20, 10], 
        cost=activation.CrossEntropyCost,
        #cost=activation.QuadraticCost,
        #cost=activation.SoftmaxCost,
        #act=activation.TanhActivation
        act=activation.SigmoidActivation
        )

eta = 0.2  # learning rate
batch_size = 21
epochs = 10
net.SGD(training_data, epochs, batch_size, eta,
        lmbda = 2,
        evaluation_data = validation_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=True,
        monitor_training_accuracy=True,
        monitor_training_cost=True)


