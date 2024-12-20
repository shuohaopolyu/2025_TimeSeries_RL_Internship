import tensorflow as tf
from collections import OrderedDict
import tensorflow_probability as tfp

tfd = tfp.distributions


def sample_from_sem(
    sem: object,
    max_time_step: int,
    intervention: dict = None,
    epsilon: OrderedDict = None,
    seed: int = None,
) -> OrderedDict:
    static_sem = sem.static
    dynamic_sem = sem.dynamic
    if intervention is not None:
        assert set(intervention.keys()) == set(static_sem().keys()).union(
            set(dynamic_sem().keys())
        )
        for key in intervention:
            assert len(intervention[key]) == max_time_step
    # default epsilon is a dictionary of normal distributions, mean 0 and std 1
    if epsilon is None:
        if seed is not None:
            tf.random.set_seed(seed)
        epsilon = OrderedDict(
            [
                (key, tfd.Normal(0.0, 1.0).sample(max_time_step))
                for key in static_sem().keys()
            ]
        )
    the_sample = OrderedDict([(key, []) for key in static_sem().keys()])

    for t in range(max_time_step):
        if t == 0:
            for key in static_sem().keys():
                if intervention is not None and intervention[key][0] is not None:
                    the_sample[key].append(intervention[key][0])
                else:
                    the_sample[key].append(
                        static_sem()[key](epsilon[key][0], t, the_sample)
                    )
        else:
            for key in dynamic_sem().keys():
                if intervention is not None and intervention[key][t] is not None:
                    the_sample[key].append(intervention[key][t])
                else:
                    the_sample[key].append(
                        dynamic_sem()[key](epsilon[key][t], t, the_sample)
                    )
    for key in the_sample.keys():
        the_sample[key] = tf.convert_to_tensor(the_sample[key])
    return the_sample


def draw_samples_from_sem(
    sem: object,
    num_samples: int,
    max_time_step: int,
    intervention: dict = None,
    epsilon: OrderedDict | float = None,
    seed: int = None,
) -> OrderedDict:
    samples = OrderedDict([(key, []) for key in sem.static().keys()])
    if epsilon is None:
        if seed is not None:
            tf.random.set_seed(seed)
        epsilon = OrderedDict(
            [
                (key, tfd.Normal(0.0, 1.0).sample([num_samples, max_time_step]))
                for key in samples.keys()
            ]
        )
    elif isinstance(epsilon, float):
        unit_epsilon = epsilon
        epsilon = OrderedDict(
            [
                (key, unit_epsilon*tf.ones([num_samples, max_time_step]))
                for key in samples.keys()
            ]
        )
    else:
        for key in samples.keys():
            assert epsilon[key].shape == (num_samples, max_time_step), (
                epsilon[key].shape,
                (num_samples, max_time_step),
            )
    for i in range(num_samples):
        i_epsilon = OrderedDict([(key, epsilon[key][i]) for key in epsilon.keys()])
        the_sample = sample_from_sem(sem, max_time_step, intervention, i_epsilon)
        for key in the_sample.keys():
            samples[key].append(the_sample[key])
    for key in samples.keys():
        samples[key] = tf.convert_to_tensor(samples[key])
    return samples


def sample_from_sem_hat(
    sem_hat: object,
    max_time_step: int,
    intervention: dict = None,
    seed: int = None,
) -> OrderedDict:

    static_sem = sem_hat.static
    dynamic_sem = sem_hat.dynamic
    if intervention is not None:
        assert set(intervention.keys()) == set(static_sem().keys()).union(
            set(dynamic_sem().keys())
        ), "Intervention keys should match the keys of the SEM."
        for key in intervention:
            assert len(intervention[key]) == max_time_step, "Intervention length should match the max time step."

    if seed is not None:
        tf.random.set_seed(seed)

    the_sample = OrderedDict([(key, []) for key in static_sem().keys()])

    for t in range(max_time_step):
        if t == 0:
            for key in static_sem().keys():
                if intervention is not None and intervention[key][0] is not None:
                    the_sample[key].append(intervention[key][0])
                else:
                    the_sample[key].append(static_sem()[key](the_sample))
        else:
            for key in dynamic_sem().keys():
                if intervention is not None and intervention[key][t] is not None:
                    the_sample[key].append(intervention[key][t])
                else:
                    the_sample[key].append(dynamic_sem()[key](t, the_sample))
    for key in the_sample.keys():
        the_sample[key] = tf.convert_to_tensor(the_sample[key])
    return the_sample


def draw_samples_from_sem_hat(
    sem_hat: object,
    num_samples: int,
    max_time_step: int,
    intervention: dict = None,
    seed: int = None,
) -> OrderedDict:
    if seed is not None:
        tf.random.set_seed(seed)
    samples = OrderedDict([(key, []) for key in sem_hat.static().keys()])
    for i in range(num_samples):
        the_sample = sample_from_sem_hat(sem_hat, max_time_step, intervention)
        for key in the_sample.keys():
            samples[key].append(the_sample[key])

    for key in samples.keys():
        samples[key] = tf.convert_to_tensor(samples[key])

    return samples
