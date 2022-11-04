# =================================
# === binary Dice-Sørensen loss ===
# =================================

# sources:
# https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

import tensorflow as tf
import tensorflow.keras.backend as K


@tf.keras.utils.register_keras_serializable()
class bDSCl(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, name="bDSCl", **kwargs):

        super().__init__(name=name, **kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dsc_metric = (2.0 * intersection + self.smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth
        )
        dsc_loss = 1.0 - dsc_metric
        return dsc_loss

    def get_config(self):

        config = super().get_config()
        config["smooth"] = self.smooth
        return config
