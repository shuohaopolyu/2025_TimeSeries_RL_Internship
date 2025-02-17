import tensorflow as tf


class LatentHamiltonianNeuralNetwork:

    def __init__(
        self,
        num_layers: int,
        num_units: int,
        num_latent_var: int,
        train_set: tf.Tensor,
        test_set: tf.Tensor,
    ):
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_latent_var = num_latent_var
        self.train_set = train_set
        self.test_set = test_set
        self.dense_layers = [
            tf.layers.Dense(num_units, activation=tf.nn.tanh) for _ in range(num_layers)
        ]
        self.dense_layers.append(tf.layers.Dense(num_latent_var, activation=None))

    def call(self, q, p):
        x = tf.concat([q, p], axis=-1)
        for layer in self.dense_layers:
            x = layer(x)
        return x
    
    # def loss(self, q, p, dqdt, dpdt):
    #     with tf.GradientTape() as tape:
    #         tape.watch(q)
    #         tape.watch(p)
    #         z = self.call(q, p)
    #     dzdq, dzdp = tape.gradient(z, [q, p])
    #     loss = tf.keras.losses.MSE(dzdq, dqdt) + tf.keras.losses.MSE(dzdp, dpdt)
    #     return loss