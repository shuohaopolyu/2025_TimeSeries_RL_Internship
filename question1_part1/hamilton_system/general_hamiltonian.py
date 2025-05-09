import tensorflow as tf

class HamiltonianSystem:

    def __init__(self, expU=None, expK=None, U=None, K=None, mass=None):
        # expU: exponential of negative potential energy
        # expK: exponential of negative kinetic energy
        if U is None and K is None:
            self.expU = expU
            self.expK = expK
            self.U = None
            self.K = None
        else:
            self.U = U
            self.K = K
            self.expU = None
            self.expK = None
        if mass is None:
            self.mass = 1.0
        else:
            self.mass = mass

    def H(self, q, p) -> tf.Tensor:
        # q: shape (n_dof,), p: shape (n_dof,)
        # returns: Hamiltonian, shape ()
        # for batched q and p, returns shape (batch_size,)
        if self.U is not None and self.K is not None:
            return self.U.f(q) + self.K.f(p)
        else:
            expH = self.expU.f(q) * self.expK.f(p)
            return -tf.math.log(expH)
    
    def dHdp(self, q, p) -> tf.Tensor:
        # q: shape (n_dof,), p: shape (n_dof,)
        # returns: gradient of H with respect to p, shape (n_dof,)
        # for batched q and p, returns shape (batch_size, n_dof)
        with tf.GradientTape() as tape:
            tape.watch(p)
            H = self.H(q, p)
        return tape.gradient(H, p)

    def dHdq(self, q, p) -> tf.Tensor:
        # q: shape (n_dof,), p: shape (n_dof,)
        # returns: gradient of H with respect to q, shape (n_dof,)
        # for batched q and p, returns shape (batch_size, n_dof)
        with tf.GradientTape() as tape:
            tape.watch(q)
            H = self.H(q, p)
        return tape.gradient(H, q)

    def symplectic_integrate(self, q0, p0, dt, n_steps) -> tf.Tensor:
        # q0, p0: initial position and momentum, each of shape (n_dof,)
        # dt: time step
        # n_steps: number of steps
        # returns: history of q and p, shape (n_steps+1, 4 * n_dof)
        q = q0
        p = p0
        dqdt = self.dHdp(q, p)
        dpdt = -self.dHdq(q, p)
        if len(q.shape) == 1:
            hist = tf.concat([q, p, dqdt, dpdt], axis=-1)[tf.newaxis, :]
        else:
            hist = tf.concat([q, p, dqdt, dpdt], axis=-1)
        for _ in range(n_steps):
            p_half = p - 0.5 * dt * self.dHdq(q, p)
            q_forward = q + dt / self.mass * p_half
            p_forward = p_half - 0.5 * dt * self.dHdq(q_forward, p_half)
            q = q_forward
            p = p_forward
            dqdt = self.dHdp(q, p)
            dpdt = -self.dHdq(q, p)
            if len(q.shape) == 1:
                qp = tf.concat([q, p, dqdt, dpdt], axis=-1)[tf.newaxis, :]
            else:
                qp = tf.concat([q, p, dqdt, dpdt], axis=-1)
            hist = tf.concat([hist, qp], axis=0)
        return tf.constant(hist)