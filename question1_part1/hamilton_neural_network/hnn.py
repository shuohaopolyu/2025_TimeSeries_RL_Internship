import tensorflow as tf


class HamiltonianNeuralNetwork(tf.keras.Model):

    def __init__(self, num_layers: int, num_units: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.dense_layers = [
            tf.keras.layers.Dense(num_units, activation=tf.nn.tanh, use_bias=True)
            for _ in range(num_layers)
        ] + [tf.keras.layers.Dense(1, activation=None, use_bias=False)]

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def build(self, input_shape):
        pq = tf.zeros(input_shape)
        p, q = tf.split(pq, 2, axis=-1)
        inputs = tf.concat([q, p], axis=-1)
        _ = self.call(inputs)
        super().build(input_shape)

    def loss_fcn(self, q, p, dqdt, dpdt):
        with tf.GradientTape() as tape:
            tape.watch([q, p])
            inputs = tf.concat([q, p], axis=-1)
            H = self.call(inputs)
        dHdq, dHdp = tape.gradient(H, [q, p])
        loss = tf.reduce_mean(
            tf.keras.losses.MSE(dHdq, -dpdt) + tf.keras.losses.MSE(dHdp, dqdt)
        )
        return loss

    def train(
        self,
        epochs=10000,
        batch_size=32,
        learning_rate=0.001,
        train_set=None,
        test_set=None,
        save_dir=None,
        print_every=100,
    ):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        train_hist = []
        test_hist = []
        print("Training started...")
        for epoch in range(epochs):
            for i in range(0, train_set.shape[0], batch_size):
                batch = train_set[i : i + batch_size, :]
                q, p, dqdt, dpdt = tf.split(batch, 4, axis=-1)
                with tf.GradientTape() as tape:
                    train_loss = self.loss_fcn(q, p, dqdt, dpdt)
                trainable_vars = self.trainable_variables
                gradients = tape.gradient(train_loss, trainable_vars)
                optimizer.apply_gradients(zip(gradients, trainable_vars))
            if (epoch + 1) % print_every == 0:
                q, p, dqdt, dpdt = tf.split(test_set, 4, axis=-1)
                test_loss = self.loss_fcn(q, p, dqdt, dpdt)
                print(
                    f"Epoch {epoch+1}: Train loss {train_loss.numpy()}, Test loss {test_loss.numpy()}."
                )
                train_hist.append(train_loss.numpy())
                test_hist.append(test_loss.numpy())
                if save_dir:
                    self.save_weights(save_dir)
        train_hist = tf.constant(train_hist)
        test_hist = tf.constant(test_hist)
        print("Training complete!")
        return train_hist, test_hist

    def forward(self, q, p):
        if len(q.shape) == 1:
            q = q[tf.newaxis, :]
            p = p[tf.newaxis, :]
        inputs = tf.concat([q, p], axis=-1)
        return self.call(inputs)

    def dHdp(self, q, p):
        with tf.GradientTape() as tape:
            tape.watch(p)
            inputs = tf.concat([q, p], axis=-1)
            H = self.call(inputs)
        return tape.gradient(H, p)

    def dHdq(self, q, p):
        with tf.GradientTape() as tape:
            tape.watch(q)
            inputs = tf.concat([q, p], axis=-1)
            H = self.call(inputs)
        return tape.gradient(H, q)

    def symplectic_integrate(self, q0, p0, dt, n_steps) -> tf.Tensor:
        assert q0.shape == p0.shape, "q0 and p0 must have the same shape."
        if len(q0.shape) == 1:
            q0 = q0[tf.newaxis, :]
            p0 = p0[tf.newaxis, :]
        q = q0
        p = p0
        dqdt = self.dHdp(q, p)
        dpdt = -self.dHdq(q, p)
        hist = tf.concat([q, p, dqdt, dpdt], axis=-1)
        for _ in range(n_steps):
            p_half = p - dt / 2 * self.dHdq(q, p)
            q_forward = q + dt * p_half
            p_forward = p_half - dt / 2 * self.dHdq(q_forward, p_half)
            q = q_forward
            p = p_forward
            dqdt = self.dHdp(q, p)
            dpdt = -self.dHdq(q, p)
            qp = tf.concat([q, p, dqdt, dpdt], axis=-1)
            hist = tf.concat([hist, qp], axis=0)
        return tf.constant(hist)
