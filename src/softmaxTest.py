

import numpy as np


data = np.array([1, 1, 0, 0, 1])

w = np.random.randn(5) / 5
b = np.random.randn(1)

y = np.array([0, 0, 1, 0, 0])

#print data.shape, w.shape

z = np.dot(w, data.transpose()) + b

#print z

y=np.array(([1,2,3],[4,5,6]))
def change(a):
    a = np.exp(a)
    return a

yy = change(y)
print "y=",y
print "yy=",yy

