import numpy as np

# Softmax function
a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

def softmax_1(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# Caution points on softmax function: Values can explode
a = np.array([1010, 1000, 990])
np.exp(a) / np.sum(np.exp(a)) # nan occurred

c = np.max(a)
a - c
np.exp(a - c) / np.sum(np.exp(a - c)) # Works

# Softmax function with adjustment
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# Features of softmax function
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y, np.sum(y))