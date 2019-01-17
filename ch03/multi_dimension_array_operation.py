import numpy as np

A = np.array([1, 2, 3, 4])
print(A)

np.ndim(A)
A.shape
A.shape[0]

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)

np.ndim(B)
B.shape
B.shape[0]

# Matrix multiplication (Inner product)
A = np.array([[1, 2], [3, 4]])
A.shape
B = np.array([[5, 6], [7, 8]])
B.shape

np.dot(A, B)

A = np.array([[1, 2, 3], [4, 5, 6]])
A.shape
B = np.array([[1, 2], [3, 4], [5, 6]])
B.shape

np.dot(A, B)
np.dot(B, A)

C = np.array([[1, 2], [3, 4]])
C.shape
np.dot(A, C) # Do not run
np.dot(C, A)

A = np.array([[1, 2], [3, 4], [5, 6]])
A.shape
B = np.array([7, 8])
B.shape
if A.shape[1] == B.shape[0]:
    np.dot(A, B)

# Dot product of neural network
X = np.array([1, 2])
X.shape
W = np.array([[1, 3, 4], [2, 4, 5]])
W.shape
Y = np.dot(X, W)
print(Y)

# Implementing NN with matrix multiplication
X = np.array([1.0, 5.0])
# First layer
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
print(A1)

Z1 = sigmoid(A1)
print(Z1)

# Second layer
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(A2, '\n', Z2)

# Third layer
def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
print(Y)


# Summary of implementation
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)

