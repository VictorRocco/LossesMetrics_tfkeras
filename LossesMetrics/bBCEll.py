# =====================================
# === Binary Cross Entropy log loss ===
# =====================================

# Sources:
# https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
# https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
# https://www.rapidtables.com/calc/math/Log_Calculator.html

import tensorflow as tf
import tensorflow.keras.backend as K

@tf.keras.utils.register_keras_serializable()
class bBCEll(tf.keras.losses.Loss):

	def __init__(self, name="BCEll", **kwargs):
        			
		super().__init__(name=name, **kwargs)
		self.loss_fnc = tf.keras.losses.BinaryCrossentropy()

	def call(self, y_true, y_pred):
		bce_loss = self.loss_fnc(y_true, y_pred)
		bce_loss = K.clip(bce_loss, K.epsilon(), 1.0 - K.epsilon()) #Yeah, it's incredible, i got some out of range
		bce_logloss = -1.0 * K.log(1.0 - bce_loss)
		return bce_logloss

	def get_config(self):

		config = super().get_config()
		return config


