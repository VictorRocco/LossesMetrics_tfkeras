"""
===========================================
=== stacked Binary Cross Entropy metric ===
===========================================

For multi output model with stacked outputs
Example: stacked outputs OUTPUT +OUT0 +OUT... +OUTN
only takes OUTPUT for metric

Sources:
- https://keras.io/api/metrics/probabilistic_metrics/#binarycrossentropy-class
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class stacked_bBCEm(tf.keras.metrics.Metric):

    def __init__(self, name="stacked_bBCEm", **kwargs):
        super(stacked_bBCEm, self).__init__(name=name, **kwargs)
        self.metric_fnc = tf.keras.metrics.BinaryCrossentropy()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_unstacked = tf.unstack(y_pred)
        self.metric_fnc.update_state(y_true, y_pred_unstacked[0])

    def result(self):
        return self.metric_fnc.result()

    def reset_state(self):
        self.metric_fnc.reset_state()

    def get_config(self):
        config = super().get_config()
        return config
