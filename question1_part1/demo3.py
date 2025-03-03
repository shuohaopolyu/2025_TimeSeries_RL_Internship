import sys

sys.path.append("../")
from hamilton_neural_network import (
    TrainTestData,
    LatentHamiltonianNeuralNetwork,
)
from hamilton_system import HamiltonianSystem
from pdf_models import NegLogIndepedentGaussians, NegLogNealFunnel
import tensorflow as tf
import matplotlib.pyplot as plt
from no_u_turn.nuts import NoUTurnSampling

tf.random.set_seed(0)

U = NegLogNealFunnel()
K = NegLogIndepedentGaussians(tf.constant([0.0, 0.0]), tf.constant([1.0, 1.0]))
q0 = tf.constant([[0.0, 0.0]])
p0 = tf.random.normal(q0.shape)
T = 250.0
leap_frog_per_unit = 40
num_samples = 40
num_train = int(0.9 * num_samples * leap_frog_per_unit * T)

# train_test_data = TrainTestData(num_samples, T, leap_frog_per_unit, q0, p0, U=U, K=K)
# samples = train_test_data()
# tf.io.write_file("./exps/demo3_train_test_data.txt", tf.io.serialize_tensor(samples))

file = tf.io.read_file("./exps/demo3_train_test_data.txt")
train_test_data = tf.io.parse_tensor(file, out_type=tf.float32)
train_test_data = tf.random.shuffle(train_test_data)
train_data = train_test_data[:num_train, :]
test_data = train_test_data[num_train:, :]
print(train_data.shape, test_data.shape)
lhnn = LatentHamiltonianNeuralNetwork(3, 100, 2)
lhnn.build(input_shape=(1, 4))
train_hist, test_hist = lhnn.train(
    701, 1000, 1e-4, train_data, test_data, save_dir="./exps/demo3_lhnn.weights.h5", print_every=100
)

lhnn = LatentHamiltonianNeuralNetwork(3, 100, 2)
lhnn.build(input_shape=(1, 4))
lhnn.load_weights("./exps/demo3_lhnn.weights.h5")
q0 = tf.constant([[0.0, 0.0]])
nuts = NoUTurnSampling(
    num_samples=25000,
    q0=q0,
    dt=0.025,
    lhnn=lhnn,
    Hamiltonian=HamiltonianSystem(U=U, K=K),
    Delta_lf=1000.0,
    Delta_lhnn=10.0,
    num_lf_steps=20,
    j_max=12
)
nuts(print_every=5000)
print(nuts.lhnn_call)
print(nuts.Hamiltonian_gradient_call)
q_hist = tf.concat(nuts.q_hist, axis=0)
tf.io.write_file("./exps/demo3_q_hist.txt", tf.io.serialize_tensor(q_hist))
