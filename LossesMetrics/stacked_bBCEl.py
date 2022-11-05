"""
=========================================
=== stacked Binary Cross Entropy loss ===
=========================================

For multi output model with stacked outputs
Example: stacked outputs OUTPUT +OUT0 +OUT... +OUTN

Sources:
- https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class stacked_bBCEl(tf.keras.losses.Loss):
    def __init__(self, name="stacked_bBCEl", **kwargs):

        super().__init__(name=name, **kwargs)
        self.loss_fnc = tf.keras.losses.BinaryCrossentropy()

    def call(self, y_true, y_pred):
        y_pred_unstacked = tf.unstack(y_pred)
        loss = 0.0
        for y_pred_i in y_pred_unstacked:
            loss += self.loss_fnc(y_true, y_pred_i)
        return loss

    def get_config(self):

        config = super().get_config()
        return config
