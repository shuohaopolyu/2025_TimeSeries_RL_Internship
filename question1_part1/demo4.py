import sys

from hamilton_neural_network import (
    TrainTestData,
    LatentHamiltonianNeuralNetwork,
)
from hamilton_system import HamiltonianSystem
from pdf_models import NegLogIndepedentGaussians
import tensorflow as tf
import matplotlib.pyplot as plt
from no_u_turn.nuts import NoUTurnSampling

tf.random.set_seed(0)

a = tf.math.sqrt(0.1)
b = tf.math.sqrt(10.0)
U = NegLogIndepedentGaussians(
    tf.constant([0.0, 0.0, 0.0, 0.0, 0.0]),
    tf.constant([0.1, a.numpy(), 1.0, b.numpy(), 10.0]),
)
K = NegLogIndepedentGaussians(
    tf.constant([0.0, 0.0, 0.0, 0.0, 0.0]), tf.constant([1.0, 1.0, 1.0, 1.0, 1.0])
)
T = 120.0
leap_frog_per_unit = 40
num_samples = 40
num_train = int(0.9 * num_samples * leap_frog_per_unit * T)

# file = tf.io.read_file("./exps/demo4_train_test_data.txt")
# train_test_data = tf.io.parse_tensor(file, out_type=tf.float32)
# train_test_data = tf.random.shuffle(train_test_data)
# train_data = train_test_data[:num_train, :]
# test_data = train_test_data[num_train:, :]
# plt.plot(train_data[:100, 0])
# plt.show()
# lhnn = LatentHamiltonianNeuralNetwork(3, 64, 5)
# lhnn.build(input_shape=(1, 10))
# train_hist, test_hist = lhnn.train(
#     5000, 1000, 4e-5, train_data, test_data, save_dir="./exps/demo4_lhnn.weights.h5"
# )

lhnn = LatentHamiltonianNeuralNetwork(3, 64, 5)
lhnn.build(input_shape=(1, 10))
lhnn.load_weights("./exps/demo4_lhnn.weights.h5")
original_hamiltonian = HamiltonianSystem(U=U, K=K)

q0 = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0]])
nuts = NoUTurnSampling(
    num_samples=25000,
    q0=q0,
    dt=0.025,
    lhnn=lhnn,
    Hamiltonian=HamiltonianSystem(U=U, K=K),
    Delta_lf=1000.0,
    Delta_lhnn=10.0,
    num_lf_steps=20,
    j_max=12,
)
nuts(print_every=100)
print(nuts.lhnn_call)
print(nuts.Hamiltonian_gradient_call)
q_hist = tf.concat(nuts.q_hist, axis=0)
tf.io.write_file("./exps/demo4_q_hist.txt", tf.io.serialize_tensor(q_hist))