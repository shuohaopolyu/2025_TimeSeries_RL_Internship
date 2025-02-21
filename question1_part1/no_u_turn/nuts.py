import tensorflow as tf

class NoUTurnSampling:
    """
    No-U-Turn Sampling (NUTS) built from scratch based on the original paper:
    Hoffman, M. D., and Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting 
    Path Lengths in Hamiltonian Monte Carlo.
    """

    def __init__(self, num_samples, q0, dt, hnn=None, lhnn=None):
        self.num_samples = num_samples
        self.dt = dt
        assert (
            hnn is not None or lhnn is not None
        ), "Either hnn or lhnn must be provided"
        self.hnn = hnn if hnn is not None else lhnn
        self.q_hist = []
        self.q_hist.append(q0)

    def __call__(self):
        for i in range(self.num_samples):
            print(f"Sampling {i+1}/{self.num_samples}")
            p0 = tf.random.normal(self.q_hist[-1].shape)
            H = self.hnn.forward(self.q_hist[i], p0)
            u = tf.random.uniform([], 0, tf.exp(-H))
            q_minus, q_plus = self.q_hist[i], self.q_hist[i]
            p_minus, p_plus = p0, p0
            j = 0
            C = [(self.q_hist[i], p0)]
            s = 1
            while s == 1:
                _v = tf.random.uniform([], -1.0, 1.0)
                v_j = tf.sign(_v)
                print(f"v_j: {v_j}")
                if v_j == -1:
                    q_minus, p_minus, _, _, C_prime, s_prime = self.buildtree(
                        q_minus, p_minus, u, v_j, j
                    )
                else:
                    _, _, q_plus, p_plus, C_prime, s_prime = self.buildtree(
                        q_plus, p_plus, u, v_j, j
                    )
                if s_prime == 1:
                    C += C_prime
                position_moved = q_plus - q_minus
                s = (
                    s_prime
                    * tf.sign(tf.matmul(position_moved, p_minus))
                    * tf.sign(tf.matmul(position_moved, p_plus))
                )
                j += 1
            # sample q and p randomly from C
            idx = tf.random.uniform([], 0, len(C), dtype=tf.int32)
            self.q_hist.append(C[idx][0])
            print(f"q_hist: {self.q_hist}")

    def buildtree(self, q0, p0, u, v_j, j, Delta_max=1000) -> tf.Tensor:
        if j == 0:
            q_prime, p_prime = self.leapfrog(q0, p0, v_j)
            H = self.hnn.forward(q_prime, p_prime)
            if u <= tf.exp(-H):
                C_prime = [(q_prime, p_prime)]
            else:
                C_prime = []
            s_prime = tf.sign(-H - tf.math.log(u) + Delta_max)
            return q_prime, p_prime, q_prime, p_prime, C_prime, s_prime
        else:
            q_minus, p_minus, q_plus, p_plus, C_prime, s_prime = self.buildtree(
                q0, p0, u, v_j, j - 1
            )
            if v_j == -1:
                q_minus, p_minus, _, _, C_prime_prime, s_prime_prime = self.buildtree(
                    q_minus, p_minus, u, v_j, j - 1
                )
            else:
                _, _, q_plus, p_plus, C_prime_prime, s_prime_prime = self.buildtree(
                    q_plus, p_plus, u, v_j, j - 1
                )
            position_moved = q_plus - q_minus
            s_prime = (
                s_prime
                * s_prime_prime
                * tf.sign(tf.matmul(position_moved, p_minus))
                * tf.sign(tf.matmul(position_moved, p_plus))
            )
            C_prime += C_prime_prime
            return q_minus, p_minus, q_plus, p_plus, C_prime, s_prime

    def leapfrog(self, q0, p0, v_j) -> tf.Tensor:
        q = tf.identity(q0)
        p = tf.identity(p0)
        p_half = p - 0.5 * self.dt * self.hnn.dHdq(q, p) * v_j
        q_prime = q + self.dt * p_half
        p_prime = p_half - 0.5 * self.dt * self.hnn.dHdq(q_prime, p_half) * v_j
        return q_prime, p_prime

class EfficientNoUTurnSampling:
    def __init__(self, num_samples, q0, p0, dt, NN):
        pass

    def buildtree(self, q0, p0, n_steps, direction) -> tf.Tensor:
        pass
