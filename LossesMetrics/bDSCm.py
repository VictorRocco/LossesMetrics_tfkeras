# ===================================
# === binary Dice-Sørensen metric ===
# ===================================

# sources:
# https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

import tensorflow as tf
import tensorflow.keras.backend as K


@tf.keras.utils.register_keras_serializable()
class bDSCm(tf.keras.metrics.Metric):
    def __init__(self, scale=1.0, smooth=1e-6, name="bDSCm", **kwargs):

        super().__init__(name=name, **kwargs)
        self.scale = scale
        self.smooth = smooth

    def update_state(self, y_true, y_pred, sample_weight=None):

        self.y_true = y_true
        self.y_pred = y_pred

    def result(self):

        y_true_f = K.flatten(self.y_true)
        y_pred_f = K.flatten(self.y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dsc_metric = (2.0 * intersection + self.smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth
        )
        return dsc_metric * self.scale

    def get_config(self):

        config = super().get_config()
        config["scale"] = self.scale
        config["smooth"] = self.smooth
        return config
