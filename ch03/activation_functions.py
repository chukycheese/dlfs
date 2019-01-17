def step_function_1(x):
    if x > 0:
        return 1
    else:
        return 0

step_function_1(0.5)
step_function_1(-0.234)

import numpy as np
step_function_1(np.array([0.1, 1.2])) # Do not run

def step_funtion_2(x):
    y = x > 0
    return y.astype(np.int)

x = np.array([0.1, -1.0, 2.0])
y = x > 0
y
y.astype(np.int)
step_funtion_2(np.array([0.1, -1, 3.0]))

# Graph of step function
import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, dtype = np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
sigmoid(x)

t = np.array([1.0, 2.0, 3.0])
1.0 + t
1.0 / t

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.plot(x, step_function(x), linestyle = '--')
plt.ylim(-0.1, 1.1)
plt.show()

# ReLU: Rectified Linear Unit
def ReLU(x):
    return np.maximum(0, x)