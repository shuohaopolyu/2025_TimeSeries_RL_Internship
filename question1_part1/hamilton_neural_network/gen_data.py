import tensorflow as tf
from hamilton_system import HamiltonianSystem
import matplotlib.pyplot as plt

class TrainTestData:
    def __init__(
        self,
        num_samples: int,
        T: float,
        leap_frog_per_unit: int,
        q0_for_first_sample: tf.Tensor,
        p0_for_first_sample: tf.Tensor,
        expU: callable=None,
        expK: callable=None,
        U: callable = None,
        K: callable = None,
        mass: tf.Tensor = None,
    ):
        self.num_samples = num_samples
        self.dt = 1.0 / leap_frog_per_unit
        self.n_steps = int(T / self.dt)
        self.q0_for_first_sample = q0_for_first_sample
        self.p0_for_first_sample = p0_for_first_sample
        self.H_system = HamiltonianSystem(expU, expK, U, K, mass)
        self.n_dof = self.q0_for_first_sample.shape[-1]
        if expK is not None:
            assert self.n_dof == expK.sigmas.shape[-1]
        if K is not None:
            assert self.n_dof == K.sigmas.shape[-1]

    def __call__(self) -> tf.Tensor:
        print("Generating samples...")
        samples = []
        q = self.q0_for_first_sample
        p = self.p0_for_first_sample
        for i in range(self.num_samples):
            hist = self.H_system.symplectic_integrate(
                q,
                p,
                self.dt,
                self.n_steps,
            )
            samples.append(hist)
            q = hist[-1, 0:self.n_dof]
            p = tf.random.normal((self.n_dof,))
        print("Finished generating samples.")
        return tf.concat(samples, axis=0)
