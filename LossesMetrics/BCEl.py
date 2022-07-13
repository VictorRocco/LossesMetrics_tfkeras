# Binary Cross Entropy loss

import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class BCEl(tf.keras.losses.Loss):

	def __init__(self, name="BCEl", **kwargs):
        			
		super().__init__(name=name, **kwargs)
		self.loss_fnc = tf.keras.losses.BinaryCrossentropy()

	def call(self, y_true, y_pred):

		return self.loss_fnc(y_true, y_pred)

	def get_config(self):

		config = super().get_config()
		return config


