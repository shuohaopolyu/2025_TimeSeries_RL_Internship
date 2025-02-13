import tensorflow as tf

class HamiltonianSystem:

    def __init__(self, expU, expK):
        self.expU = expU
        self.expK = expK

    def H(self, q, p):
        expH = self.expU.f(q) * self.expK.f(p)
        return -tf.math.log(expH)
    
    def dHdp(self, q, p):
        with tf.GradientTape() as tape:
            tape.watch(p)
            H = self.H(q, p)
        return tape.gradient(H, p)

    def dHdq(self, q, p):
        with tf.GradientTape() as tape:
            tape.watch(q)
            H = self.H(q, p)
        return tape.gradient(H, q)

    def symplectic_integrate(self, q0, p0, dt, n_steps):
        hist = []
        q = q0
        p = p0
        for _ in range(n_steps):
            hist.append([q.numpy(), p.numpy()])
            # q_forward = q + dt / self.mass * p - dt ** 2 / (2 * self.mass) * self.dHdq(q, p)
            q_forward = q + dt / 1.0 * p - dt ** 2 / (2 * 1.0) * self.dHdq(q, p)
            p_forward = p - dt / 2 * (self.dHdq(q, p) + self.dHdq(q_forward, p))
            q = q_forward
            p = p_forward
        return hist