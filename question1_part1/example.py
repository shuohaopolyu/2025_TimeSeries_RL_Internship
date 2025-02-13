from prob_models import IndepedentGaussians, OneDimGaussianMixtureDensity
import tensorflow as tf
import matplotlib.pyplot as plt
from hamilton_system import HamiltonianSystem

# q = tf.linspace(-2.4, 2.4, 1000)[:, None]
# expU = OneDimGaussianMixtureDensity()
# pdf = expU.f(q)
# plt.plot(q, pdf)
# plt.show()

expU = OneDimGaussianMixtureDensity()
expK = IndepedentGaussians(tf.constant([0.0]), tf.constant([1.0]))
H_system = HamiltonianSystem(expU, expK)

hist = H_system.symplectic_integrate(tf.constant([1.0]), tf.constant([3.0]), 0.05, 400)
hist = tf.constant(hist)

plt.plot(hist[:, 0], hist[:, 1])
plt.show()

