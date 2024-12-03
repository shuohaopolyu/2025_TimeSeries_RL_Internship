from models.sem import StructuralEquationModel
import tensorflow as tf
from collections import OrderedDict


class StationaryModel(StructuralEquationModel):

    @staticmethod
    def static():
        X = lambda noise, t, sample: noise
        Z = lambda noise, t, sample: tf.exp(-sample["X"][t]) + noise
        Y = (
            lambda noise, t, sample: tf.cos(sample["Z"][t])
            - tf.exp(-sample["Z"][t] / 20.0)
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    @staticmethod
    def dynamic():
        X = lambda noise, t, sample: sample["X"][t - 1] + noise
        Z = (
            lambda noise, t, sample: tf.exp(-sample["X"][t])
            + sample["Z"][t - 1]
            + noise
        )
        Y = (
            lambda noise, t, sample: tf.cos(sample["Z"][t])
            - tf.exp(-sample["Z"][t] / 20.0)
            + sample["Y"][t - 1]
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])
