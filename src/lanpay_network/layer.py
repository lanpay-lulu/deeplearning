import numpy as np


'''
--- Note: activation function class should have such method:
        - fn(z)  # activation function
        - prime(z)  # prime function
'''

class Layer(object):
    def __init__(self, w, b, act_fn):
        self.act_fn = act_fn
        self.w = w
        self.b = b
        self.x = None # input
        self.z = None # middle result
        self.a = None # activation

    def forward(self, x):  # input is x, shape is (f, n); f is feature size, n is sample number.
        self.x = x
        self.z = np.dot(self.w, x) + self.b
        self.a = self.act_fn.fn(self.z)
        return self.a

    def backward(self, delta):
        prime = self.act_fn.prime(self.z)
        delta = delta * prime
        nabla_b = np.zeros(self.b.shape)
        nabla_b[:,0] = np.sum(delta, axis=1)
        nabla_w = np.dot(delta, self.x.transpose())
        delta = np.dot(self.w.transpose(), delta)
        return (nabla_b, nabla_w, delta)
        
    ''' 
        eta: learning rate
        lmbda: regularization rate
        n: batch size
    ''' 
    def update(self, nabla_b, nabla_w, eta, lmbda, batch_n, total_n):
        self.b = self.b - eta/batch_n * nabla_b
        self.w = (1-eta*lmbda/total_n) * self.w - eta/batch_n * nabla_w

'''
Output layer includes more than a normal layer.
Besides a normal layer, it also include a cost function.
Becauese (cost + normal) layer has a simpler delta form, we combine them together. 
'''
class OutputLayer(Layer):
    def __init__(self, w, b, act_fn, cost_fn): # if act_fn is contained in cost_fn, then act_fn should be None
        self.combined = (act_fn == None)
        a_fn = act_fn if act_fn!=None else cost_fn
        Layer.__init__(self, w, b, a_fn)
        self.cost_fn = cost_fn
        #self.act_fn = act_fn

    # forward is the same, so just call the super method

    
    def backward(self, y): # delta starts from here
        if self.combined:
            delta = (self.cost_fn).delta_z(self.z, y)  # dL / dz
        else:
            delta_a = (self.cost_fn).delta(self.a, y)
            delta = (self.act_fn).prime(z) * delta_a 

        nabla_b = np.zeros(self.b.shape)
        #nabla_w = np.zeros(w.shape)
        nabla_b[:,0] = np.sum(delta, axis=1)
        nabla_w = np.dot(delta, self.x.transpose())
        delta = np.dot(self.w.transpose(), delta)
        return (nabla_b, nabla_w, delta)



