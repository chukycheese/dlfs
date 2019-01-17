import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, y_train), (x_test, y_test) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(y_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
y_batch = y_train[batch_mask]

def cross_entropy_error(y_pred, y):
    if pred_y.ndim == 1:
        y = y.reshape(1, y.size)
        pred_y = pred_y.reshape(1, pred_y.size)

    batch_size = pred_y.shape[0]
    return -np.sum(np.log(pred_y[np.arange(batch_size), y])) / batch_size