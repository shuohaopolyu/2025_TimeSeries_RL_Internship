import tensorflow_probability as tfp
import tensorflow as tf
from sems import Y2X, U2XY, X2Y
import seaborn as sns
import matplotlib.pyplot as plt
from biocd import BIOCausalDiscovery
from collections import OrderedDict

x2y = X2Y()
D_obs = x2y.propagate(5000)


cd = BIOCausalDiscovery(
    true_sem=x2y,
    D_obs=D_obs,
    max_iter=20000,
    num_mixture=50,
    num_monte_carlo=4096,
    beta=0.2,
    num_int=10,
    debug_mode=True
)
cd.run()
cd.recorder_plot()

# D_int = OrderedDict((("X", tf.constant([0.8, -1.2])), ("Y", tf.constant([0.4, 0.9]))))

# cd._update_m_0()
# cd._update_m_1()
# cd._update_bayes_factor_01_int()
# cd._update_prior_p_dc()
# cd._update_D_int(1.0)
# print(cd.D_int)
# cd._find_x_opt()
# print(cd._p_dc(1.2))

# m0 = cd._update_m_0()
# new_samples = m0.sample((5000,))
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# sns.kdeplot(new_samples, ax=axs[0])
# sns.kdeplot(D_obs['Y'], ax=axs[1])
# plt.show()

# m1 = cd._update_m_1()
# sample_1 = m1.sample(x=1.0, num_samples=5000)
# sample_2 = m1.sample(x=-5.0, num_samples=5000)
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].scatter(D_obs['X'], D_obs['Y'])
# sns.kdeplot(sample_1, ax=axs[1])
# sns.kdeplot(sample_2, ax=axs[1])
# plt.show()

# u2xy = U2XY()
# D_obs = u2xy.propagate(5000)
# cd = BIOCausalDiscovery(
#     true_sem=u2xy,
#     D_obs=D_obs,
#     max_iter=20000,
#     num_mixture=100,
#     num_monte_carlo=4096,
#     beta=0.2,
#     num_int=10,
#     debug_mode=False
# )
# cd.run()
# cd.recorder_plot()
