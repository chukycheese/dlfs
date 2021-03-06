import numpy as np
from neuralnet_mnist import get_data, init_network, predict

x, y = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    pred_y_batch = predict(network, x_batch)
    p = np.argmax(pred_y_batch, axis = 1)
    accuracy_cnt += np.sum(p == y[i:i+batch_size])

print('Accuracy: ' + str(float(accuracy_cnt) / len(x)))