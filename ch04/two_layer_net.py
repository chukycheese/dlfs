import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)

        y = softmax(z2)

        return y

    def loss(self, x, y):
        pred_y = self.predict(x)

        return cross_entropy_error(pred_y, y)
    
    def accuracy(self, x, y):
        pred_y = self.predict(x)
        pred_y = np.argmax(pred_y, axis = 1)
        y = np.argmax(y, axis = 1)

        accuracy = np.sum(pred_y == y) / float(x.shape[0])
        
        return accuracy

    def numerical_gradient(self, x, y):
        loss_W = lambda W: self.loss(x, y)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

net = TwoLayerNet(input_size = 784, hidden_size = 100, output_size = 10)
net.params['W1'].shape
net.params['b1'].shape
net.params['W2'].shape
net.params['b2'].shape

x = np.random.randn(100, 784)
pred_y = net.predict(x)

x = np.random.rand(100, 784)
y = np.random.rand(100, 10)

grads = net.numerical_gradient(x, y)

grads['W1'].shape
grads['b1'].shape
grads['W2'].shape
grads['b2'].shape
        

