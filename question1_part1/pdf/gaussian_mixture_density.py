import tensorflow as tf


class OneDimGaussianMixtureDensity:

    def __init__(self, sigma_1=0.35, sigma_2=0.35, mu_1=1, mu_2=-1):
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.mu_1 = mu_1
        self.mu_2 = mu_2

    def f(self, q):
        return 0.5 * tf.exp(
            (q - self.mu_1) ** 2 / (2 * self.sigma_1**2)
        ) + 0.5 * tf.exp((q - self.mu_2) ** 2 / (2 * self.sigma_2**2))
