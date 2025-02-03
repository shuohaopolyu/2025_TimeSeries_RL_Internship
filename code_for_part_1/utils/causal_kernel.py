import tensorflow as tf
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import (
    positive_semidefinite_kernel as psd_kernel,
)
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.math.psd_kernels.internal import util
from tensorflow_probability.python.internal import assert_util


__all__ = ["CausalKernel"]


class CausalKernel(psd_kernel.AutoCompositeTensorPsdKernel):
    """A custom kernel that applies the callable causal_std_fn to the base kernel.

    Args:
        causal_std_fn: A callable function that takes a tensor with shape [B, E, F] or [E, F], where B is the batch size,
        E is the number of examples, and F is the number of features,
        the function returns a tensor with shape [B, E] or [E] that represents the standard deviation of the causal kernel.

        feature_ndims: The number of feature dimensions. Default is 1.
    """

    def __init__(
        self,
        causal_std_fn: callable,
        amplitude=None,
        length_scale=None,
        inverse_length_scale=None,
        feature_ndims=1,
        validate_args=False,
        dtype=tf.float32,
        name="CausalKernel",
        jitter=1e-6,
    ):
        parameters = dict(locals())
        self.jitter = jitter
        if (length_scale is not None) and (inverse_length_scale is not None):
            raise ValueError(
                "Must specify at most one of `length_scale` and "
                "`inverse_length_scale`."
            )
        with tf.name_scope(name):
            self._amplitude = tensor_util.convert_nonref_to_tensor(
                amplitude, name="amplitude", dtype=dtype
            )
            self._length_scale = tensor_util.convert_nonref_to_tensor(
                length_scale, name="length_scale", dtype=dtype
            )
            self._inverse_length_scale = tensor_util.convert_nonref_to_tensor(
                inverse_length_scale, name="inverse_length_scale", dtype=dtype
            )
            super().__init__(
                feature_ndims,
                dtype=dtype,
                name=name,
                validate_args=validate_args,
                parameters=parameters,
            )
        self.causal_std_fn = causal_std_fn

    @property
    def amplitude(self):
        """Amplitude parameter."""
        return self._amplitude

    @property
    def length_scale(self):
        """Length scale parameter."""
        return self._length_scale

    @property
    def inverse_length_scale(self):
        """Inverse length scale parameter."""
        return self._inverse_length_scale

    def _inverse_length_scale_parameter(self):
        if self.inverse_length_scale is None:
            if self.length_scale is not None:
                return tf.math.reciprocal(self.length_scale)
            else:
                return None
        else:
            return tf.convert_to_tensor(self.inverse_length_scale)

    @classmethod
    def _parameter_properties(cls, dtype):
        from tensorflow_probability.python.bijectors import (
            softplus,
        )  # pylint:disable=g-import-not-at-top

        return dict(
            amplitude=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus.Softplus(low=dtype_util.eps(dtype))
                )
            ),
            length_scale=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus.Softplus(low=dtype_util.eps(dtype))
                )
            ),
            inverse_length_scale=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=softplus.Softplus
            ),
        )

    def _apply_with_distance(self, x1, x2, example_ndims=0):
        pairwise_square_distance = util.sum_rightmost_ndims_preserving_shape(
            tf.math.squared_difference(x1, x2), self.feature_ndims
        )
        exponent = -0.5 * pairwise_square_distance
        inverse_length_scale = self._inverse_length_scale_parameter()

        if inverse_length_scale is not None:
            inverse_length_scale = util.pad_shape_with_ones(
                inverse_length_scale, example_ndims
            )
            exponent = exponent * tf.math.square(inverse_length_scale)

        if self.amplitude is not None:
            amplitude = tf.convert_to_tensor(self.amplitude)
            amplitude = util.pad_shape_with_ones(amplitude, example_ndims)
            exponent = exponent + 2.0 * tf.math.log(amplitude)
        return tf.exp(exponent)

    def _apply_causal_variance(self, x1, x2):
        if x1 is x2:
            return self.causal_std_fn(x1)**2
        k_x1 = tf.expand_dims(self.causal_std_fn(x1), -1)
        k_x2 = tf.expand_dims(self.causal_std_fn(x2), -2)
        k_x12 = tf.matmul(k_x1, k_x2) + self.jitter
        return k_x12

    def _apply(self, x1, x2, example_ndims=0):
        return self._apply_with_distance(
            x1, x2, example_ndims=example_ndims
        ) + self._apply_causal_variance(x1, x2)
    
    def _parameter_control_dependencies(self, is_init):
        if not self.validate_args:
            return []
        assertions = []
        if self._inverse_length_scale is not None and is_init != tensor_util.is_ref(
            self._inverse_length_scale
        ):
            assertions.append(
                assert_util.assert_non_negative(
                    self._inverse_length_scale,
                    message="`inverse_length_scale` must be non-negative.",
                )
            )
        for arg_name, arg in dict(
            amplitude=self.amplitude, length_scale=self._length_scale
        ).items():
            if arg is not None and is_init != tensor_util.is_ref(arg):
                assertions.append(
                    assert_util.assert_positive(
                        arg, message=f"{arg_name} must be positive."
                    )
                )
        return assertions
