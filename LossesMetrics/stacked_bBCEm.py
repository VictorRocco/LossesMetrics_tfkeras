# ===========================================
# === stacked Binary Cross Entropy metric ===
# ===========================================
# For multi output model with stacked outputs
# Example: stacked outputs OUTPUT +OUT0 +OUT... +OUTN
# only takes OUTPUT for metric

# sources:
# https://keras.io/api/metrics/probabilistic_metrics/#binarycrossentropy-class

import tensorflow as tf
import tensorflow.keras.backend as K

@tf.keras.utils.register_keras_serializable()
class stacked_bBCEm(tf.keras.metrics.Metric):

	def __init__(self, name="stacked_bBCEm", **kwargs):
        			
		super().__init__(name=name, **kwargs)
		self.metric_fnc = tf.keras.metrics.BinaryCrossentropy()

	def update_state(self, y_true, y_pred, sample_weight=None):

		self.y_true = y_true
		self.y_pred = y_pred

	def result(self):

		return self.metric_fnc(self.y_true, self.y_pred[0])

	def get_config(self):

		config = super().get_config()
		return config


