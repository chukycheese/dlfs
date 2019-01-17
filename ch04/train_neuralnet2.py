import numpy as np
from dataset.mnist import load_mnist
from ch04.two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize = True, one_hot_label = True)

# Hyperparameters
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# Numer of iteration per epoch
iter_per_epoch = max(train_size / batch_size, 1)


for i in range(iters_num):

    # Get mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # Calcuate gradient
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch) # Improved version

    # Update hyperparameters
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # Log the learning progress
    loss = network.loss(x_train, t_train)
    train_loss_list.append(loss)

# Calculate accuracy for each epoch
if i % iter_per_epoch ==0:
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)

    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    print('train acc, test acc |' 
          + str(train_acc) + ", " + str(test_acc))