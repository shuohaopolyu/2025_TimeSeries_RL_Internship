import tensorflow as tf
from utils.gaussian_process import build_gprm, clone_gprm
import matplotlib.pyplot as plt

obs_index = tf.linspace(-1., 1., 100)
obs = tf.sin(obs_index * 3.14) + tf.random.normal([100], 0, 0.1)
obs_index = tf.reshape(obs_index, [-1, 1])
index_points = tf.linspace(-1., 1., 100)
index_points = tf.reshape(index_points, [-1, 1])
index_points_2 = tf.constant([[0.5]])
gpm,_,_ = build_gprm(index_points, obs_index, obs)
new_gpm = clone_gprm(gpm, index_points_2)
print(new_gpm.sample())

# mean_fcn = gpm.mean_fn(index_points)
# std_fcn = gpm.stddev_fn(index_points)
# plt.plot(index_points, mean_fcn)
# plt.fill_between(
#     index_points[:, 0],
#     mean_fcn + 2 * std_fcn,
#     mean_fcn - 2 * std_fcn,
#     alpha=0.1,
#     color='k',
# )
# plt.scatter(obs_index, obs, color='r')

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


