import tensorflow as tf
import tensorflow_probability as tfp
from collections import OrderedDict
from utils.causal_kernel import CausalKernel



def build_gprm(
    index_x: tf.Tensor,
    x: tf.Tensor,
    y: tf.Tensor,
    amplitude_factor: float = 1.0,
    length_scale_factor: float = 1.0,
    obs_noise_factor: float = 0.1,
    max_training_step: int = 20000,
    learning_rate: float = 2e-4,
    patience: int = 20,
    mean_fn=None,
    causal_std_fn=None,
    debug_mode=False,
):
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

    if causal_std_fn is None:
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=amplitude, length_scale=length_scale
        )
    else:
        kernel = CausalKernel(
            causal_std_fn=causal_std_fn, amplitude=amplitude, length_scale=length_scale
        )
        # observation_noise_variance = (
        #     0.0  # CausalKernel does not have explicit noise defined.
        # )

        assert (
            mean_fn is not None
        ), "Please provide a mean function for the CausalKernel."

    # Utilize an unconditioned Gaussian Process to train the kernel parameters
    gp = tfp.distributions.GaussianProcess(
        kernel=kernel,
        index_points=x,
        observation_noise_variance=observation_noise_variance,
        mean_fn=mean_fn,
    )

    # Define the optimizer and optimization loop
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
        kernel=gp.kernel,
        index_points=index_x,
        observation_index_points=x,
        observations=y,
        observation_noise_variance=gp.observation_noise_variance,
        mean_fn=gp.mean_fn,
    )

    return gprm, losses, is_early_stopping


def build_gaussian_variable(observation_data: tf.Tensor) -> callable:
    assert (
        len(observation_data.shape) == 1
    ), "Variable observation_data should be 1D tensor."
    assert (
        observation_data.shape[0] > 1
    ), "Variable observation_data should have more than 1 element."
    mean_obs = tf.reduce_mean(observation_data)
    std_obs = tf.math.reduce_std(observation_data)

    def gaussian_variable(sample: OrderedDict, e_num: int = 1):
        return tfp.distributions.Normal(loc=mean_obs, scale=std_obs).sample((e_num, 1))

    return gaussian_variable


def build_gaussian_process(gprm, predecessors: list[str]) -> callable:

    def gaussian_process(sample: OrderedDict, e_num: int = 1):
        index_x = []
        for i, parent in enumerate(predecessors):
            parent_name, parent_index = parent.split("_")
            ipt_of_this_parent = sample[parent_name][:, int(parent_index)]
            reshaped_parent_input = ipt_of_this_parent[:, tf.newaxis]
            if i == 0:
                index_x = reshaped_parent_input
            else:
                index_x = tf.concat([index_x, reshaped_parent_input], axis=1)
        assert len(index_x.shape) == 2, "Variable index_x should be 2D tensor."
        assert index_x.shape[1] == len(
            predecessors
        ), "Variable index_x should have the same length as the predecessors."
        # sample from the marginal distribution of the Gaussian Process Regression Model
        # https://github.com/tensorflow/probability/issues/837
        samples = (gprm.get_marginal_distribution(index_x).sample())[:, tf.newaxis]
        return samples

    return gaussian_process
