import tensorflow as tf
from no_u_turn.nuts import NoUTurnSampling
from hamilton_neural_network import LatentHamiltonianNeuralNetwork
from pdf_models import NegLogThreeDimRosenbrock, NegLogIndepedentGaussians
import matplotlib.pyplot as plt
import h5py
from hamilton_system import HamiltonianSystem


# lhnn = LatentHamiltonianNeuralNetwork(3, 16, 4)
# lhnn.build((1, 2))
# # lhnn.summary()
# file_name = "./exps/demo_1_lhnn.weights.h5"
# with h5py.File(file_name, "r") as f:
#     print(list(f.keys()))
#     for key in f.keys():
#         print(key, f[key])
# lhnn.load_weights("./exps/demo_1_lhnn.weights.h5")

# num_samples = 300
# q0 = tf.constant([[0.0]])
# p0 = tf.constant([[1.0]])
# dt = 0.05
# nuts = NoUTurnSampling(num_samples=num_samples, q0=q0, dt=dt, lhnn=lhnn)
# q_prime, p_prime = nuts.leapfrog(q0, p0, 1)
# hist = []
# for i in range(num_samples):
#     hist.append(q_prime)
#     q_prime, p_prime = nuts.leapfrog(q_prime, p_prime, 1)
# hist = tf.concat(hist, axis=0)
# plt.plot(hist.numpy().flatten())
# plt.show()

U = NegLogThreeDimRosenbrock()
K = NegLogIndepedentGaussians(tf.constant([0.0, 0.0, 0.0]), tf.constant([1.0, 1.0, 1.0]))
q0 = tf.constant([[0.0, 2.0, 10.0]])
p0 = tf.constant([[0.0, 0.0, 0.0]])
T = 40.0
leap_frog_per_unit = 40
num_samples =1
num_train = int(0.9 * num_samples * leap_frog_per_unit * T)
H_sys = HamiltonianSystem(U=U, K=K)
print(H_sys.H(q0, p0))

# file = tf.io.read_file("./exps/demo2_train_test_data.txt")
# train_test_data = tf.io.parse_tensor(file, out_type=tf.float32)
# print(train_test_data.shape)
hist = H_sys.symplectic_integrate(q0, p0, 0.025, 1000)
plt.plot(hist[:, 0])
plt.plot(hist[:, 1])
plt.plot(hist[:, 2])
plt.show()