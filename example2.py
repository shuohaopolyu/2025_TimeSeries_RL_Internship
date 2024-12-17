import tensorflow as tf
from utils.gaussian_process import build_gprm
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

num_data = 30
num_data_plt = 1000
jitter = 1e-6
obs_index = tf.linspace(-1.0, 1.0, num_data)
obs = tf.sin(obs_index * 3.14) + tf.random.normal([num_data], 0, 0.4)
obs_index = tf.reshape(obs_index, [-1, 1])
index_points = tf.linspace(-1.0, 1.0, num_data_plt) + jitter
index_points = tf.reshape(index_points, [-1, 1])
mean_fn = lambda x: tf.sin(x * 3.14)[:, 0]

# gpm, _, _ = build_gprm(
#     index_points, obs_index, obs, mean_fn=mean_fn, observation_noise_variance=0.16, debug_mode=True
# )
# upper, lower = gpm.mean() + 2 * gpm.stddev(), gpm.mean() - 2 * gpm.stddev()
# plt.plot(gpm.index_points, gpm.mean())
# plt.fill_between(
#     gpm.index_points[:, 0],
#     upper,
#     lower,
#     alpha=0.1,
#     color="k",
# )
# plt.plot(gpm.index_points, mean_fn(index_points), color="k")
# plt.scatter(obs_index, obs, color="r")
# plt.show()

def causal_std_fn(x):
    return tf.ones(x.shape[:-1], dtype=tf.float32)*0.4
causalgpm, _, _ = build_gprm(
    index_points, obs_index, obs, mean_fn=mean_fn, causal_std_fn=causal_std_fn, debug_mode=True, max_training_step=10000
)
gpm, _, _ = build_gprm(
    index_points, obs_index, obs, debug_mode=True, max_training_step=10000
)
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
upper, lower = causalgpm.mean() + 2 * causalgpm.stddev(), causalgpm.mean() - 2 * causalgpm.stddev()
ax[0].plot(causalgpm.index_points, causalgpm.mean())
ax[0].fill_between(
    causalgpm.index_points[:, 0],
    upper,
    lower,
    alpha=0.1,
    color="k",
)
ax[0].plot(causalgpm.index_points, mean_fn(index_points), color="k")
ax[0].scatter(obs_index, obs, color="r")

upper, lower = gpm.mean() + 2 * gpm.stddev(), gpm.mean() - 2 * gpm.stddev()
ax[1].plot(gpm.index_points, gpm.mean())
ax[1].fill_between(
    gpm.index_points[:, 0],
    upper,
    lower,
    alpha=0.1,
    color="k",
)
ax[1].plot(gpm.index_points, mean_fn(index_points), color="k")
ax[1].scatter(obs_index, obs, color="r")

plt.show()
