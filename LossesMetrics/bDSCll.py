"""
=====================================
=== binary Dice-Sørensen log loss ===
=====================================

Sources:
- https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
"""

import tensorflow as tf
import tensorflow.keras.backend as K

from .bDSCl import bDSCl


@tf.keras.utils.register_keras_serializable()
class bDSCll(tf.keras.losses.Loss):

    def __init__(self, smooth=1e-6, name="bDSCll", **kwargs):

        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.loss_fnc = bDSCl(smooth=self.smooth)

    def call(self, y_true, y_pred):

        dsc_loss = self.loss_fnc(y_true, y_pred)
        dsc_loss = K.clip(dsc_loss, K.epsilon(), 1.0 - K.epsilon())  # Safety first
        dsc_logloss = -1.0 * K.log(1.0 - dsc_loss)
        return dsc_logloss

    def get_config(self):

        config = super().get_config()
        config["smooth"] = self.smooth
        return config
