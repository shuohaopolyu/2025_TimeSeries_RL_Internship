import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class ConditionalMixGaussian():
    def __init__(self, p_psi: callable, theta: tuple):
        """
        Initializes the Conditional GMM model: 
        p(y | x) = Sum_{k=1}^{num_mix} pi_k * N(y | A * tanh(Bx) + mu_k, sigma_k)

        Parameters:
            p_psi: The trained Gaussian Mixture Model for residuals.
            theta: Tuple containing ([A, B]).
        """
        self.p_psi = p_psi
        self.A = theta[0]
        self.B = theta[1]

    def p_y_given_x(self, x: tf.Tensor):
        """
        Defines the conditional distribution p(y | x).

        Parameters:
            x: Tensor of shape [batch_size], input features.

        Returns:
            gmm: A MixtureSameFamily distribution representing p(y | x).
        """
        # Compute f(x, theta) = A * tanh(Bx)
        fx = self.A * tf.tanh(self.B * x)
        # Expand fx to [batch_size, 1] to enable broadcasting with [num_mix]
        fx_expanded = tf.expand_dims(fx, axis=-1)  # Shape: [batch_size, 1]
        mixture_means = fx_expanded + tf.expand_dims(self.p_psi.components_distribution.loc, axis=0)  # Shape: [batch_size, num_mix]
        mixture_stddevs = self.p_psi.components_distribution.scale  # Shape: [num_mix]

        # Define the mixture model for each x
        gmm = tfd.MixtureSameFamily(
            mixture_distribution=self.p_psi.mixture_distribution,
            components_distribution=tfd.Normal(loc=mixture_means, scale=mixture_stddevs),
        )
        return gmm

    def log_prob(self, y, x):
        """
        Computes log p(y | x).

        Parameters:
            y: Tensor of shape [batch_size], observed y values.
            x: Tensor of shape [batch_size], input features.

        Returns:
            log_probs: Tensor of shape [batch_size], log-probabilities.
        """
        gmm = self.p_y_given_x(x)
        return gmm.log_prob(y)
    
    def prob(self, y, x):
        """
        Computes p(y | x).

        Parameters:
            y: Tensor of shape [batch_size], observed y values.
            x: Tensor of shape [batch_size], input features.

        Returns:
            probs: Tensor of shape [batch_size], probabilities.
        """
        gmm = self.p_y_given_x(x)
        return gmm.prob(y)

    def sample(self, x, num_samples=1):
        """
        Samples y values from p(y | x).

        Parameters:
            x: Tensor of shape [batch_size], input features.
            num_samples: Integer, number of samples per x.

        Returns:
            samples: Tensor of shape [batch_size, num_samples], sampled y values.
        """
        gmm = self.p_y_given_x(x)
        return gmm.sample(num_samples)