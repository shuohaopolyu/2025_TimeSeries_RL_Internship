import tensorflow as tf
from hamilton_system import HamiltonianSystem
import matplotlib.pyplot as plt

class TrainTestData:
    def __init__(
        self,
        num_samples: int,
        expU: callable,
        expK: callable,
        T: float,
        leap_frog_per_unit: int,
        q0_for_first_sample: tf.Tensor,
        p0_for_first_sample: tf.Tensor,
    ):
        self.num_samples = num_samples
        self.expU = expU
        self.expK = expK
        self.dt = 1.0 / leap_frog_per_unit
        self.n_steps = int(T / self.dt)
        self.q0_for_first_sample = q0_for_first_sample
        self.p0_for_first_sample = p0_for_first_sample
        self.H_system = HamiltonianSystem(self.expU, self.expK)
        self.n_dof = self.H_system.mass.shape[0]

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
