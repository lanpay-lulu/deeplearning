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

    def forward(self, x):  # previous layer size is n1, this layer size is n2. batch size is m.
        self.x = x  # shape (n1, m)
        self.z = np.dot(self.w, x) + self.b  # w shape (n2, n1), z shape is (n2, m)
        self.a = self.act_fn.fn(self.z) # a shape (n2, m)
        return self.a

    def backward(self, delta):
        prime = self.act_fn.prime(self.z)
        delta = delta * prime
        nabla_b = np.zeros(self.b.shape)
        nabla_b[:,0] = np.sum(delta, axis=1)
        nabla_w = np.dot(delta, self.x.transpose())
        delta = np.dot(self.w.transpose(), delta)  # w involved
        return (nabla_b, nabla_w, delta)
        
    ''' 
        eta: learning rate
        lmbda: regularization rate
        n: batch size
    ''' 
    def update(self, nabla_b, nabla_w, eta, lmbda, batch_n, total_n):
        self.b = self.b - eta/batch_n * nabla_b
        self.w = (1-eta*lmbda/total_n) * self.w - eta/batch_n * nabla_w



class BNLayer(Layer):
    def __init__(self, w, b, act_fn):
        Layer.__init__(self, w, b, act_fn)
        self.eps = 0.0001
        self.gm = np.ones(b.shape) # gamma, same shape as self.b
        self.middle = None   
        self.nabla_gm = np.zeros(b.shape)
        self.mean = None # for test
        self.std = None # for test
        self.mean_list = [] # store middle mean
        self.var_list = [] # store middle var

    def forward(self, x, train=True): # previous layer size is n1, current layer size is n2
        # test
        if not train:
            if self.mean is None:
                self.calc_test_param()
            xx = np.dot(self.w, x)
            zz = (xx - self.mean) / self.std
            z = self.gm * zz + self.b
            a = self.act_fn.fn(xx)
            return a
        
        # train
        self.x = x      
        xx = np.dot(self.w, x)  # w is (n2, n1), x is (n1, m), xx is (n2, m)
        n, m = xx.shape # m is batch size, n is n2.
        mean = xx.mean(axis=1).reshape([n, 1])
        var = xx.var(axis=1).reshape([n, 1]) + self.eps
        std = np.sqrt(var)
        zz = (xx - mean) / std # normalize. zz is (n2, m) -- x hat
        z = self.gm * zz + self.b # transform. z is (n2, m)
        self.z = z
        self.a = self.act_fn.fn(xx) # (n2, m)
        self.middle = [xx, mean, std, zz]
        self.mean_list.append(mean[:,0])
        self.var_list.append(var[:,0])
        return self.a

    def backward(self, delta):
        xx, mean, std, zz = self.middle
        n, m = xx.shape
        prime = self.act_fn.prime(self.z) # dL / dz
        delta = delta * prime  # (n, m)
        nabla_b = np.zeros(self.b.shape)
        nabla_b[:,0] = np.sum(delta, axis=1)
        self.nabla_gm[:,0] = np.sum(delta * zz, axis=1) 
        
        # dz/dw = dz/dzz * dzz/xx * dxx/dw
        dzz = delta * self.gm # (n, m)
        # formular comes from [https://kevinzakka.github.io/2016/09/14/batch_normalization/]
        dxx = (1. / m) / std * (m*dzz - np.sum(dzz, axis=1).reshape([n,1]) - zz*np.sum(zz*dzz, axis=1).reshape([n,1])) 
        delta = dxx
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
        self.gm = self.gm - eta/batch_n * self.nabla_gm
        self.w = (1-eta*lmbda/total_n) * self.w - eta/batch_n * nabla_w


    def calc_test_param(self):
        me = np.array(self.mean_list) # (m, n), m is list length, n is layer size.
        m, n = me.shape
        va = np.array(self.var_list)
        self.mean = me.mean(axis=0).reshape([n, 1])
        var = va.mean(axis=0).reshape([n, 1]) * m / (m-1) + self.eps
        self.std = np.sqrt(var)
        self.mean_list = []
        self.var_list = []

'''
Output layer includes more than a normal layer.
Besides a normal layer, it also include a cost function.
Becauese (cost + normal) layer has a simpler delta form, we combine them together. 
'''
class OutputLayer(Layer):
    def __init__(self, w, b, act_fn, cost_fn): # if act_fn is contained in cost_fn, then act_fn should be None
        self.combined = cost_fn.combined()
        a_fn = act_fn if not self.combined else cost_fn
        Layer.__init__(self, w, b, a_fn)
        self.cost_fn = cost_fn
        print '[info] init OutputLayer...'
        print 'act_fn = ', act_fn
        print 'cost_fn = ', cost_fn
        #self.act_fn = act_fn

    # forward is the same, so just call the super method

    
    def backward(self, y): # delta starts from here
        if self.combined:
            delta = (self.cost_fn).delta_z(self.z, y)  # dL / dz
        else:
            delta_a = (self.cost_fn).delta(self.a, y)
            delta = (self.act_fn).prime(self.z) * delta_a 

        nabla_b = np.zeros(self.b.shape)
        #nabla_w = np.zeros(w.shape)
        nabla_b[:,0] = np.sum(delta, axis=1)
        nabla_w = np.dot(delta, self.x.transpose())
        delta = np.dot(self.w.transpose(), delta)
        return (nabla_b, nabla_w, delta) # nabla_w contains no batch size



