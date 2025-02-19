import tensorflow as tf


class LatentHamiltonianNeuralNetwork(tf.keras.Model):

    def __init__(
        self,
        num_layers: int,
        num_units: int,
        num_latent_var: int,
        train_set: tf.Tensor,
        test_set: tf.Tensor,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_latent_var = num_latent_var
        self.train_set = train_set
        self.test_set = test_set
        self.dense_layers = [
            tf.keras.layers.Dense(num_units, activation=tf.nn.tanh)
            for _ in range(num_layers)
        ]
        self.dense_layers.append(tf.keras.layers.Dense(num_latent_var, activation=None, use_bias=False))

    def call(self, q, p):
        x = tf.concat([q, p], axis=-1)
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def loss(self, q, p, dqdt, dpdt):
        with tf.GradientTape() as tape:
            tape.watch(q)
            tape.watch(p)
            z = self.call(q, p)
            H = tf.reduce_sum(z, axis=-1)
        dHdq, dHdp = tape.gradient(H, [q, p])
        loss = tf.reduce_sum(
            tf.keras.losses.MSE(dHdp, -dqdt) + tf.keras.losses.MSE(dHdq, dpdt)
        )
        return loss

    def train(self, epochs=10000, batch_size=32, learning_rate=0.001):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        for epoch in range(epochs):
            for i in range(0, self.train_set.shape[0], batch_size):
                batch = self.train_set[i : i + batch_size, :]
                q, p, dqdt, dpdt = tf.split(batch, 4, axis=-1)
                with tf.GradientTape() as tape:
                    loss = self.loss(q, p, dqdt, dpdt)
                trainable_vars = [
                    var
                    for layer in self.dense_layers
                    for var in layer.trainable_variables
                ]
                gradients = tape.gradient(loss, trainable_vars)
                optimizer.apply_gradients(zip(gradients, trainable_vars))
            if epoch % 100 == 0:
                q, p, dqdt, dpdt = tf.split(self.test_set, 4, axis=-1)
                test_loss = self.loss(q, p, dqdt, dpdt)
                print(
                    f"Epoch {epoch}: Train loss {loss.numpy()}, Test loss {test_loss.numpy()}."
                )
        print("Training complete!")

    def forward(self, q, p):
        return tf.reduce_sum(self.call(q, p), axis=-1)
    
    def dHdp(self, q, p):
        with tf.GradientTape() as tape:
            tape.watch(p)
            z = self.call(q, p)
            H = tf.reduce_sum(z, axis=-1)
        return tape.gradient(H, p)
    
    def dHdq(self, q, p):
        with tf.GradientTape() as tape:
            tape.watch(q)
            z = self.call(q, p)
            H = tf.reduce_sum(z, axis=-1)
        return tape.gradient(H, q)
    
    def symplectic_integrate(self, q0, p0, dt, n_steps) -> tf.Tensor:
        q = q0
        p = p0
        dqdt = self.dHdp(q, p)
        dpdt = self.dHdq(q, p)
        hist = tf.concat([q, p, dqdt, dpdt], axis=-1)[tf.newaxis, :]
        for _ in range(n_steps):
            q_forward = q + dt * p
            p_forward = p - dt * self.dHdq(q, p)
            q = q_forward
            p = p_forward
            dqdt = self.dHdp(q, p)
            dpdt = -self.dHdq(q, p)
            qp = tf.concat([q, p, dqdt, dpdt], axis=-1)[tf.newaxis, :]
            hist = tf.concat([hist, qp], axis=0)
        return tf.constant(hist)
