import tensorflow as tf
import numpy as np
import abc


class PDFModel(abc.ABC):

    @abc.abstractmethod
    def f(self, q: tf.Tensor):
        pass


class OneDimGaussianMixtureDensity(PDFModel):

    def __init__(self, mu_1=1, mu_2=-1, sigma_1=0.35, sigma_2=0.35):
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.mu_1 = mu_1
        self.mu_2 = mu_2

    def f(self, q: tf.Tensor):
        assert q.shape[-1] == 1, "the last dimension of q must be 1"
        return tf.reduce_sum(
            0.5 * tf.exp(-((q - self.mu_1) ** 2) / (2 * self.sigma_1**2))
            + 0.5 * tf.exp(-((q - self.mu_2) ** 2) / (2 * self.sigma_2**2)),
            axis=-1,
        )


class IndepedentGaussians(PDFModel):

    def __init__(self, mus: tf.Tensor, sigmas: tf.Tensor):
        self.mus = mus
        self.sigmas = sigmas
        assert (
            len(mus.shape) == 1 and len(sigmas.shape) == 1
        ), "mus and sigmas must be 1D tensors"
        assert (
            mus.shape[0] == sigmas.shape[0]
        ), "mus and sigmas must have the same length"

    def f(self, q: tf.Tensor):
        assert q.shape[-1] == self.mus.shape[0], "q must have the same dimension as mus"
        return tf.reduce_prod(
            tf.exp(-0.5 * (q - self.mus) ** 2 / self.sigmas**2)
            / tf.sqrt(2 * np.pi * self.sigmas**2),
            axis=-1,
        )


class ThreeDimRosenbrock(PDFModel):

    def f(self, q: tf.Tensor):
        assert q.shape[-1] == 3, "q must have dimension 3"
        if len(q.shape) == 1:
            q = q[tf.newaxis, :]
            return (
                tf.exp(
                    -(
                        (100 * (q[:, 1] - q[:, 0] ** 2) ** 2 + (1 - q[:, 0]) ** 2)
                        + (100 * (q[:, 2] - q[:, 1] ** 2) ** 2 + (1 - q[:, 1]) ** 2)
                    )
                    / 20.0
                )
            )[0]
        else:
            return tf.exp(
                -(
                    (100 * (q[:, 1] - q[:, 0] ** 2) ** 2 + (1 - q[:, 0]) ** 2)
                    + (100 * (q[:, 2] - q[:, 1] ** 2) ** 2 + (1 - q[:, 1]) ** 2)
                )
                / 20.0
            )
