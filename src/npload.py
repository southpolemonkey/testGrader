from keras import utils
import numpy as np
import matplotlib.pyplot as plt

path='mnist.npz'
path = utils.data_utils.get_file(path,
                origin='https://s3.amazonaws.com/img-datasets/mnist.npz',
                file_hash='8a61469f7ea1b51cbae51d4f78837e45')
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']

# print(x_train.shape)  #(60000, 28, 28)
# print(x_test.shape)  #(10000, 28, 28)

for i in range(11):
    plt.imshow(x_train[i])
    plt.show()