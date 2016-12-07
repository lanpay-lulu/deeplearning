import mnist_loader

#import mynetwork2
import mynetwork22 as network2


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784, 30, 10], 
        #cost=network2.CrossEntropyCost,
        cost=network2.QuadraticCost)
        #act=network2.TanhActivation)
#net.large_weight_initializer()
net.SGD(training_data, 10, 10, 0.1,
        lmbda = 5.0,
        evaluation_data = validation_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=True,
        monitor_training_accuracy=True,
        monitor_training_cost=True)


