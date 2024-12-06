import tensorflow as tf
import tensorflow_probability as tfp


def build_gprm(
    index_x: tf.Tensor,
    x: tf.Tensor,
    y: tf.Tensor,
    amplitude_factor: float = 1.0,
    length_scale_factor: float = 1.0,
    obs_noise_factor: float = 0.1,
    max_training_step: int = 2000,
    patience: int = 100,
    debug_mode=False,
):
    assert len(index_x.shape) == 2, "Variable index_x should be 2D tensor."
    assert len(x.shape) == 2, "Variable x should be 2D tensor."
    assert len(y.shape) == 1, "Variable y should be 1D tensor."

    # Initialize trainable variables, including amplitude, length_scale, for the RBF kernel and observation_noise_variance
    amplitude = tfp.util.TransformedVariable(
        initial_value=amplitude_factor,
        bijector=tfp.bijectors.Exp(),
        name="amplitude",
    )
    length_scale = tfp.util.TransformedVariable(
        initial_value=length_scale_factor,
        bijector=tfp.bijectors.Exp(),
        name="length_scale",
    )
    observation_noise_variance = tfp.util.TransformedVariable(
        initial_value=obs_noise_factor,
        bijector=tfp.bijectors.Exp(),
        name="observation_noise_variance",
    )
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude=amplitude, length_scale=length_scale
    )

    # Utilize an unconditioned Gaussian Process to train the kernel parameters
    gp = tfp.distributions.GaussianProcess(
        kernel=kernel,
        index_points=x,
        observation_noise_variance=observation_noise_variance,
    )

    # Define the optimizer and optimization loop
    optimizer = tf.optimizers.Adam(learning_rate=0.01, beta_1=0.5, beta_2=0.99)

    @tf.function
    def optimize():
        with tf.GradientTape() as tape:
            loss = -gp.log_prob(y)
        grads = tape.gradient(loss, gp.trainable_variables)
        optimizer.apply_gradients(zip(grads, gp.trainable_variables))
        return loss

    best_loss = float("inf")
    patience_counter = 0
    losses = []
    is_early_stopping = False

    for step in range(max_training_step):
        loss_value = optimize()
        losses.append(loss_value)

        if debug_mode:
            if step % 100 == 0:
                print(f"Step {step}, Loss {loss_value}")

        if loss_value < best_loss:
            best_loss = loss_value
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > patience:
            print(f"Early stopping at step {step}")
            is_early_stopping = True
            break

    if debug_mode:
        print(f"Optimization finished at step {step}, Loss {loss_value}")

    if not is_early_stopping:
        print("Warning: optimization might not converge")

    gprm = tfp.distributions.GaussianProcessRegressionModel(
        kernel=kernel,
        index_points=index_x,
        observation_index_points=x,
        observations=y,
        observation_noise_variance=observation_noise_variance,
    )

    return gprm, losses, is_early_stopping