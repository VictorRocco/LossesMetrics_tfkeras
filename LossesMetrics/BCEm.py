# Binary Cross Entropy metric

import tensorflow as tf
import tensorflow.keras.backend as K

@tf.keras.utils.register_keras_serializable()
class BCEm(tf.keras.metrics.Metric):

	def __init__(self, name="BCEm", **kwargs):
        			
		super().__init__(name=name, **kwargs)
		self.metric_fnc = tf.keras.metrics.BinaryCrossentropy()

	def update_state(self, y_true, y_pred, sample_weight=None):

		self.y_true = y_true
		self.y_pred = y_pred

	def result(self):

		return self.metric_fnc(self.y_true, self.y_pred)

	def get_config(self):

		config = super().get_config()
		return config


