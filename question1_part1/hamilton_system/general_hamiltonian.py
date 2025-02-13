import tensorflow as tf

class HamiltonianSystem:

    def __init__(self, expU, expK):
        self.expU = expU
        self.expK = expK
        self.mass = expK.sigmas

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
        q = q0
        p = p0
        hist = tf.concat([q, p], axis=-1)[tf.newaxis, :]
        for _ in range(n_steps):
            q_forward = q + dt / self.mass * p - dt ** 2 / (2 * self.mass) * self.dHdq(q, p)
            p_forward = p - dt / 2 * (self.dHdq(q, p) + self.dHdq(q_forward, p))
            q = q_forward
            p = p_forward
            qp = tf.concat([q, p], axis=-1)[tf.newaxis, :]
            hist = tf.concat([hist, qp], axis=0)
        return tf.constant(hist)