# =======================================
# === binary False Negative Rate loss ===
# =======================================

# sources:
# https://en.wikipedia.org/wiki/False_positives_and_false_negatives
# https://en.wikipedia.org/wiki/Precision_and_recall
# https://towardsdatascience.com/performance-metrics-confusion-matrix-precision-recall-and-f1-score-a8fe076a2262

import tensorflow as tf
import tensorflow.keras.backend as K


@tf.keras.utils.register_keras_serializable()
class bFNRl(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, name="bFNRl", **kwargs):

        super().__init__(name=name, **kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        true_pos = K.sum(y_true_f * y_pred_f)
        false_neg = K.sum(y_true_f * (1.0 - y_pred_f))
        fnr_metric = (false_neg + self.smooth) / (false_neg + true_pos + self.smooth)
        fnr_loss = 1.0 - fnr_metric
        return fnr_loss

    def get_config(self):

        config = super().get_config()
        config["smooth"] = self.smooth
        return config
