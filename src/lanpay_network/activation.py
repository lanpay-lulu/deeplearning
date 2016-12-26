# coding=utf8

import numpy as np


class TanhActivation(object):
    @staticmethod
    def fn(z):
        return np.tanh(z)

    @staticmethod
    def prime(z):
        fn = TanhActivation.fn
        return 1/(np.cosh(z))**2


class SigmoidActivation(object):
    @staticmethod
    def fn(z):
        return 1.0/(1.0+np.exp(-z))

    """Derivative of the sigmoid function."""
    @staticmethod
    def prime(z):
        fn = SigmoidActivation.fn
        return fn(z)*(1-fn(z))



'''
    有两种Cost类:
        (1) 一种是只包含Cost，如QuadraticCost;
        (2) 另一种聚合了Activation和Cost，如SoftmaxCost，对这种cost，其求导是dL/dz，一般这种写法是中间有能约掉的项，因此聚合到一起使得表达式更简洁，计算更快速。
    对外的接口设计:
        (1) 前向后向都是针对a；
        (2) 前向后向都是针对z (因为如何计算a只有它自己知道);
'''
    
' Softmax activation + maximum likelihood cost. '
class SoftmaxCost(object):
    @staticmethod
    def combined():
        return True


    @staticmethod
    def fn(z):
        e = np.exp(z)
        s = np.sum(e, axis=0)
        return e/s

    @staticmethod
    def cost(a, y):
        p = np.sum(a[y>0.5])
        if p < 0.000001:
            p = 0.000001
        return -np.log(p)

    @staticmethod
    def cost_z(z, y):
        fn = SoftmaxCost.fn
        a = fn(z)
        p = np.sum(a[y>0.5])
        if p < 0.000001:
            p = 0.000001
        return -np.log(p)
    
    @staticmethod
    def delta_z(z, y):
        fn = SoftmaxCost.fn
        return fn(z) - y
        #return a - y

' Just the Quadratic cost. '
class QuadraticCost(object):
    @staticmethod
    def combined():
        return False

    @staticmethod
    def cost(a, y):
        return 0.5*np.linalg.norm(a-y)**2
    
    @staticmethod
    def delta(a, y):
        return a - y


class CrossEntropyCost(object):
    @staticmethod
    def combined():
        return False

    @staticmethod
    def cost(a, y):
        return -np.sum(np.nan_to_num(y*np.log(a)+(1-y)*np.log(1-a)))
    
    @staticmethod
    def delta(a, y):
        return -(y/a - (1-y)/(1-a))


