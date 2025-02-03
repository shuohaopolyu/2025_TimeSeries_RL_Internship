import tensorflow as tf
import tensorflow_probability as tfp
from utils.conditional_dist import ConditionalMixGaussian


tfd = tfp.distributions


def build_mix_gaussian_variable(
    y: tf.Tensor,
    num_mix: int,
    learning_rate: float,
    max_training_step: int,
    debug_mode: bool = False,
):
    """Mixture Gaussian variable with learnable parameters.

    The function constructs a mixture Gaussian variable with learnable parameters.
    Specifically, the weights of the mixture components are treated as learnable parameters.
    The means and standard deviations of these components are initialized to uniform and
    constant values, respectively, and remain fixed throughout the training process.
    The weights are optimized using the Adam optimizer to minimize the negative log-likelihood
    loss corresponding to the observational data.

    """
    # Initialize means and standard deviations
    ini_means = tf.linspace(tf.reduce_min(y), tf.reduce_max(y), num_mix)
    ini_std = tf.Variable((tf.reduce_max(y) - tf.reduce_min(y)) / (3 * num_mix), name="std")

    # Initialize weights uniformly between 0 and 1
    ini_weights = tf.random.uniform((num_mix,), minval=0.0, maxval=1.0)
    weights = tf.Variable(ini_weights, name="weights")

    # Define the optimizer
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.99)

    @tf.function
    def optimize():
        with tf.GradientTape() as tape:
            ini_stds = tf.ones(num_mix) * ini_std
            # Compute softmax to ensure weights sum to 1
            mixture_probs = tf.nn.softmax(weights)

            # Construct the mixture model inside the GradientTape context
            mixture_model = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=mixture_probs),
                components_distribution=tfd.Normal(loc=ini_means, scale=ini_stds),
            )

            # Compute negative log-likelihood loss
            loss = -tf.reduce_mean(mixture_model.log_prob(y))

        # Compute gradients with respect to weights and standard deviation
        grads = tape.gradient(loss, [weights, ini_std])

        # Apply gradients if they are not None
        if grads is not None:
            optimizer.apply_gradients(zip(grads, [weights, ini_std]))


        return loss

    best_loss = float("inf")
    patience_counter = 0

    print("Start optimization for mixture Gaussian variable.")
    for i in range(max_training_step):
        loss = optimize()

        if i % 100 == 0 and debug_mode:
            print(f"Iteration {i}, Loss: {loss.numpy():.4f}")

        # Early stopping based on patience
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > 20 and debug_mode:
            print("Early stopping due to no improvement.")
            break
    print("Optimization finished.")

    # After optimization, construct the final mixture model with learned weights
    final_mixture = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=tf.nn.softmax(weights)),
        components_distribution=tfd.Normal(loc=ini_means, scale=tf.ones(num_mix) * ini_std),
    )

    return final_mixture


def build_mix_gaussian_function(
    x: tf.Tensor,
    y: tf.Tensor,
    num_mix: int,
    learning_rate: float,
    max_training_step: int,
    debug_mode: bool = False,
):
    A = tf.Variable(1.0, name="A")
    B = tf.Variable(1.0, name="B")

    # Initialize means and standard deviations
    ini_means = tfd.Uniform(tf.reduce_min(y)/2, tf.reduce_max(y)/2).sample((num_mix,))

    ini_stds = tfd.Uniform((tf.reduce_max(y) - tf.reduce_min(y)) / 50, (tf.reduce_max(y) - tf.reduce_min(y)) / 10).sample((num_mix,))

    # Initialize weights uniformly between 0 and 1
    ini_weights = tfd.Uniform(0.0, 1.0).sample((num_mix,))
    weights = tf.Variable(ini_weights, name="weights")

    def f_x(theta, x):
        _A, _B = theta
        return _A * tf.tanh(_B * x)

    # Define the optimizer
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.99)

    @tf.function
    def optimize():
        with tf.GradientTape() as tape:
            # Compute softmax to ensure weights sum to 1
            mixture_probs = tf.nn.softmax(weights)

            # Construct the mixture model inside the GradientTape context
            mixture_model = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=mixture_probs),
                components_distribution=tfd.Normal(loc=ini_means, scale=ini_stds),
            )

            # Compute negative log-likelihood loss
            loss = -tf.reduce_mean(mixture_model.log_prob(y - f_x((A, B), x)))

        # Compute gradients with respect to weights
        grads = tape.gradient(loss, [weights, A, B])

        # Apply gradients if they are not None
        if grads is not None:
            optimizer.apply_gradients(zip(grads, [weights, A, B]))

        return loss

    best_loss = float("inf")
    patience_counter = 0

    print("Start optimization for mixture Gaussian function.")
    for i in range(max_training_step):
        loss = optimize()

        if i % 100 == 0 and debug_mode:
            print(f"Iteration {i}, Loss: {loss.numpy():.4f}")

        # Early stopping based on patience
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > 20 and debug_mode:
            print("Early stopping due to no improvement.")
            break

    print("Optimization finished.")
    # After optimization, construct the final mixture model with learned weights
    p_psi = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=tf.nn.softmax(weights)),
        components_distribution=tfd.Normal(loc=ini_means, scale=ini_stds),
    )

    conditional_gmm = ConditionalMixGaussian(p_psi, (A, B))

    return conditional_gmm
