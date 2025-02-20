import sys
sys.path.append('../')
from hamilton_neural_network import (
    TrainTestData,
    HamiltonianNeuralNetwork,
    LatentHamiltonianNeuralNetwork,
)
from hamilton_system import HamiltonianSystem
from pdf_models import IndepedentGaussians, OneDimGaussianMixtureDensity
import tensorflow as tf
import matplotlib.pyplot as plt

expU = OneDimGaussianMixtureDensity()
expK = IndepedentGaussians(tf.constant([0.0]), tf.constant([1.0]))
q0 = tf.constant([0.0])
p0 = tf.constant([1.0])
T = 20.0
leap_frog_per_unit = 20
num_samples = 50
num_train = 40

file = tf.io.read_file("./exps/train_test_data.txt")
train_test_data = tf.io.parse_tensor(file, out_type=tf.float32)
train_data = train_test_data[:num_train, :]
test_data = train_test_data[num_train:, :]
hnn = HamiltonianNeuralNetwork(2, 16, train_data, test_data)
train_hist, test_hist = hnn.train(10000, 40)
ax, fig = plt.subplots()
fig.plot(train_hist, label="train", color="red")
fig.plot(test_hist, label="test", color="blue")
fig.legend()
plt.show()