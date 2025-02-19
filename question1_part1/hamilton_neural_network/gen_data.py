import tensorflow as tf
from hamilton_system import HamiltonianSystem


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

    def __call__(self) -> tf.Tensor:
        print("Generating samples...")
        H_system = HamiltonianSystem(self.expU, self.expK)
        n_dof = H_system.mass.shape[0]
        samples = []
        q = self.q0_for_first_sample
        p = self.p0_for_first_sample
        for i in range(self.num_samples):
            hist = H_system.symplectic_integrate(
                q,
                p,
                self.dt,
                self.n_steps,
            )
            samples.append(hist[-1, :])
            q = hist[-1, 0:n_dof]
            p = tf.random.normal((n_dof,))
            # print when the progress first greater than 10%, 20%, ..., 100%
            # if self.num_samples >= 10 and (i + 1) % (int(self.num_samples // 10)) == 0:
            #     print(f"{(i + 1) / self.num_samples * 100:.0f}%" + " done")
        print("Finished generating samples.")
        return tf.stack(samples)
