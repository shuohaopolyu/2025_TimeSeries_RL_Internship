import tensorflow as tf
from utils.gaussian_process import build_gprm, build_multi_gprm
import matplotlib.pyplot as plt
import tensorflow_probability as tfp


# obs_index = tf.linspace(-1., 1., 100)
# obs = tf.sin(obs_index * 3.14) + tf.random.normal([100], 0, 0.1)
# obs_index = tf.reshape(obs_index, [-1, 1])
# index_points = tf.linspace(-1., 1., 100)
# index_points = tf.reshape(index_points, [-1, 1])
# index_points_2 = tf.constant([[0.5]])
# gpm,_,_ = build_gprm(index_points, obs_index, obs)

# upper, lower = gpm.mean() + 2 * gpm.stddev(), gpm.mean() - 2 * gpm.stddev()
# plt.plot(gpm.index_points, gpm.mean())
# plt.fill_between(
#     gpm.index_points[:, 0],
#     upper,
#     lower,
#     alpha=0.1,
#     color='k',
# )
# plt.scatter(obs_index, obs, color='r')
# plt.show()

obs_index_1 = tfp.distributions.Normal(loc=0., scale=1.).sample(100)
obs_index_2 = tfp.distributions.Normal(loc=0., scale=1.).sample(100)
obs = obs_index_1 + tf.sin(obs_index_2 * 3.14) + tf.random.normal([100], 0, 0.1)
obs_index_1 = tf.reshape(obs_index_1, [-1, 1])
obs_index_2 = tf.reshape(obs_index_2, [-1, 1])
x = [obs_index_1, obs_index_2]
amplitude_factor = [1., 1.]
length_scale_factor = [1., 1.]
observation_noise_variance = [0.01, 0.01]
gpm, _, _ = build_multi_gprm(x, obs, amplitude_factor, length_scale_factor, observation_noise_variance)

new_index_1 = tf.linspace(-1., 1., 100)
new_index_2 = tf.linspace(-1., 1., 100)
gp1 = gpm[0]
gp2 = gpm[1]

pred_1 = gp1.get_marginal_distribution(new_index_1).sample(10)
pred_2 = gp2.get_marginal_distribution(new_index_2).sample(10)

plt.plot(new_index_1, pred_1, color='b', alpha=0.5)
plt.scatter(obs_index_1, obs, color='r')
plt.show()

plt.plot(new_index_2, pred_2, color='b', alpha=0.5)
plt.scatter(obs_index_2, obs, color='r')
plt.show()




