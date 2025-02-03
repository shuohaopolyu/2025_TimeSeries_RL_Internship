import tensorflow as tf
import tensorflow_probability as tfp


def build_gprm(
    index_x: tf.Tensor,
    x: tf.Tensor,
    y: tf.Tensor,
    amplitude_factor: float = 1.0,
    length_scale_factor: float = 1.0,
    obs_noise_factor: float = 1.0,
    max_training_step: int = 20000,
    learning_rate: float = 2e-4,
    patience: int = 20,
    debug_mode=False,
):
    """ Gaussian Process Regression Model (GPRM) with Exponentiated Quadratic Kernel.
    
    A key limitation of the method BIOCD is that it requires the specification of structural
    equation's link functions when approximating the m_1 model. Quoting the authors of the paper:
    "Future work could explore approaches that require less specific prior knowledge, potentially 
    employing non-parametric techniques." 

    This function implements a non-parametric technique to approximate the m_1 model using a Gaussian
    Process Regression Model (GPRM) with an Exponentiated Quadratic Kernel. The GPRM is trained using
    the observed data D_obs and the index points x. The GPRM is then used to predict the mean and standard
    deviation of the m_1 model at the index points x.
    """
    # assert len(index_x.shape) == 2, "Variable index_x should be 2D tensor."
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

    gp = tfp.distributions.GaussianProcess(
        kernel=kernel,
        index_points=x,
        observation_noise_variance=observation_noise_variance,
    )

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.99)

    @tf.function
    def optimize():
        with tf.GradientTape() as tape:
            loss = -gp.log_prob(y)
        grads = tape.gradient(loss, gp.trainable_variables)
        optimizer.apply_gradients(zip(grads, gp.trainable_variables))
        return loss
    
    best_loss = float("inf")
    patience_counter = 0

    for i in range(max_training_step):
        loss = optimize()
        if i % 100 == 0 and debug_mode:
            print(f"iteration {i}, loss: {loss}")
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter > patience:
            break

    optimized_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude=amplitude, length_scale=length_scale
    )

    gprm = tfp.distributions.GaussianProcessRegressionModel(
        kernel=optimized_kernel,
        index_points=index_x,
        observation_index_points=x,
        observations=y,
        observation_noise_variance=observation_noise_variance,
    )

    return gprm
    
