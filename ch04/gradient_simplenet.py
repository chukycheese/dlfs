import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, y):
        z = self.predict(x)
        pred_y = softmax(z)
        loss = cross_entropy_error(pred_y, y)

        return loss

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

np.argmax(p)

y = np.array([0, 0, 1])
net.loss(x, y)

def f(W):
    return net.loss(x, y)

dW = numerical_gradient(f, net.W)
dW

f = lambda w: net.loss(x, y)
dW = numerical_gradient(f, net.W)
dW