# Binary DICE (Dice-Sørensen) metric
# https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

import tensorflow as tf
import tensorflow.keras.backend as K

@tf.keras.utils.register_keras_serializable()
class bDICEm(tf.keras.metrics.Metric):

	def __init__(self, name="bDICEm", smooth=1e-6, **kwargs):
        			
		super().__init__(name=name, **kwargs)
		self.smooth = smooth

	def update_state(self, y_true, y_pred, sample_weight=None):

		self.y_true = y_true
		self.y_pred = y_pred

	def result(self):

		y_true_f = K.flatten(self.y_true)
		y_pred_f = K.flatten(self.y_pred)
		intersection = K.sum(y_true_f * y_pred_f)
		return (2. * intersection + self.smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth)

	def get_config(self):

		config = super().get_config()
		config["smooth"] = self.smooth
		return config


