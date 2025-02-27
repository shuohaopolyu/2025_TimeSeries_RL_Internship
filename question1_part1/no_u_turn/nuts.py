import tensorflow as tf
from hamilton_neural_network import LatentHamiltonianNeuralNetwork
from hamilton_system import HamiltonianSystem


class NoUTurnSampling:
    """
    No-U-Turn Sampling (NUTS) built from scratch based on the original paper:
    Hoffman, M. D., and Gelman, A. (2014). The No-U-Turn Sampler: Adaptively
    Setting Path Lengths in Hamiltonian Monte Carlo, and
    Dhulipalaa et al. (2019). Efficient Bayesian Inference with Latent
    Hamiltonian Neural Networks in No-U-Turn Sampling.
    """

    def __init__(
        self,
        num_samples: int,
        q0: tf.tensor,
        dt: float,
        lhnn: LatentHamiltonianNeuralNetwork,
        Hamiltonian: HamiltonianSystem,
        Delta_lf: float,
        Delta_lhnn: float,
        num_lf_steps: int,
    ):
        self.num_samples = num_samples
        self.dt = dt
        self.lhnn = lhnn
        self.q_hist = []
        self.q_hist.append(q0)
        self.Hamiltonian = Hamiltonian
        self.Delta_lf = Delta_lf
        self.Delta_lhnn = Delta_lhnn
        self.num_lf_steps = num_lf_steps

    def __call__(self):
        indicator_lf = 0
        n_lf = 0
        for i in range(self.num_samples):
            q_0 = self.q_hist[-1]
            p_0 = tf.random.normal(q_0.shape)
            H = self.Hamiltonian.H(q_0, p_0)
            u = tf.random.uniform(1, 0, tf.exp(-H))
            q_minus, p_minus = q_0, p_0
            q_plus, p_plus = q_0, p_0
            j = 0
            q_star = q_0
            n = 1
            s = 1
            if indicator_lf == 1:
                n_lf += 1
            if n_lf == self.num_lf_steps:
                indicator_lf = 0
                n_lf = 0
            while s == 1:
                v = tf.cast(tf.random.uniform(1, 0, 1) < 0.5, tf.int32)
                if v == -1:
                    (
                        q_minus,
                        p_minus,
                        _,
                        _,
                        q_prime,
                        p_prime,
                        n_prime,
                        s_prime,
                        indicator_lf,
                    ) = self.buildtree(q_minus, p_minus, u, v, j, indicator_lf)
                else:
                    (
                        _,
                        _,
                        q_plus,
                        p_plus,
                        q_prime,
                        p_prime,
                        n_prime,
                        s_prime,
                        indicator_lf,
                    ) = self.buildtree(q_plus, p_plus, u, v, j, indicator_lf)
                if s_prime == 1:
                    # With probability min{1, n′/n }, set {qi,pi} ← {q′,p′}
                    if tf.random.uniform(1, 0, 1) < n_prime / n:
                        q_star = q_prime
                n = n + n_prime
                s = (
                    s_prime
                    * tf.cast(
                        tf.reduce_sum((q_plus - q_minus) * p_minus) >= 0, tf.float32
                    )
                    * tf.cast(
                        tf.reduce_sum((q_plus - q_minus) * p_plus) >= 0, tf.float32
                    )
                )
                j += 1
            self.q_hist.append(q_star)
        return self.q_hist

    def buildtree(self, q0, p0, u, v, j, indicator_lf) -> tf.Tensor:
        if j == 0:
            q_prime, p_prime = self.leapfrog(q0, p0, v)
            H = self.Hamiltonian.H(q_prime, p_prime)
            if (H + tf.math.log(u) - self.Delta_lhnn > 0) or (indicator_lf == 1):
                indicator_lf = 1
            else:
                indicator_lf = 0
            s_prime = tf.cast(H + tf.math.log(u) - self.Delta_lhnn <= 0, tf.float32)

            if indicator_lf == 1:
                q_prime, p_prime = self.leapfrog(q0, p0, v)
                s_prime = tf.cast(H + tf.math.log(u) - self.Delta_lf <= 0, tf.float32)

            n_prime = tf.cast(u - tf.exp(-H) <= 0, tf.float32)
            return (
                q_prime,
                p_prime,
                q_prime,
                p_prime,
                q_prime,
                p_prime,
                n_prime,
                s_prime,
                indicator_lf,
            )
        else:
            #Recursion to build left and right sub-trees (follows from Algorithm 3 in [5], with 1lf additionall 
            #passed to and retrieved from every BuildTree evaluation)
            pass

    def leapfrog(self, q0, p0, v) -> tuple:
        assert q0.shape == p0.shape, "q0 and p0 must have the same shape."
        if len(q0.shape) == 1:
            q0 = q0[tf.newaxis, :]
            p0 = p0[tf.newaxis, :]
        q = q0
        p = p0
        p_half = p - 0.5 * self.dt * self.lhnn.dHdq(q, p) * v
        q_prime = q + self.dt * p_half
        p_prime = p_half - 0.5 * self.dt * self.lhnn.dHdq(q_prime, p_half) * v
        return q_prime, p_prime
