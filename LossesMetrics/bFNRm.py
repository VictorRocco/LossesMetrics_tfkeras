"""
=========================================
=== binary False Negative Rate metric ===
=========================================

Sources:
- https://en.wikipedia.org/wiki/False_positives_and_false_negatives
- https://en.wikipedia.org/wiki/Precision_and_recall
- towardsdatascience.com/performance-metrics-confusion-matrix-precision-recall-and-f1-score-a8fe076a2262
"""

import tensorflow as tf
import tensorflow.keras.backend as K


@tf.keras.utils.register_keras_serializable()
class bFNRm(tf.keras.metrics.Metric):
    def __init__(self, scale=1.0, smooth=1e-6, name="bFNRm", **kwargs):

        super().__init__(name=name, **kwargs)
        self.scale = scale
        self.smooth = smooth

    def update_state(self, y_true, y_pred, sample_weight=None):

        self.y_true = y_true
        self.y_pred = y_pred

    def result(self):

        y_true_f = K.flatten(self.y_true)
        y_pred_f = K.flatten(self.y_pred)
        true_pos = K.sum(y_true_f * y_pred_f)
        false_neg = K.sum(y_true_f * (1.0 - y_pred_f))
        fnr_metric = (false_neg + self.smooth) / (false_neg + true_pos + self.smooth)
        return fnr_metric * self.scale

    def get_config(self):

        config = super().get_config()
        config["scale"] = self.scale
        config["smooth"] = self.smooth
        return config
