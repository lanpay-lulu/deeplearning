import mnist_loader

import BNnetwork
import activation

# add softmax to the output layer

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = BNnetwork.Network([784, 30, 10], 
        cost=activation.CrossEntropyCost,
        #cost=activation.QuadraticCost,
        #cost=activation.SoftmaxCost,
        #act=activation.TanhActivation
        act=activation.SigmoidActivation
        )

eta = 0.1  # learning rate
batch_size = 11
epochs = 10
net.SGD(training_data, epochs, batch_size, eta,
        lmbda = 2,
        evaluation_data = validation_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=True,
        monitor_training_accuracy=True,
        monitor_training_cost=True)


