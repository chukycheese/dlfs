import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
    h = 1e-04
    return (f(x + h) - f(x - h)) / (2 * h)

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

# plt.xlabel('x')
# plt.ylabel('y')
# plt.plot(x, y)
# plt.show()

numerical_diff(function_1, 5)
numerical_diff(function_1, 10)

# Partial differentiation
def function_2(x):
    return np.sum(x ** 2, axis = 0)

# Numerical Gradient
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        
        # Calculate f(x + h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # Calculate f(x - h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val # Restore the original value
    
    return grad

numerical_gradient(function_2, np.array([3.0, 4.0]))
numerical_gradient(function_2, np.array([0.0, 2.0]))
numerical_gradient(function_2, np.array([3.0, 0.0]))
