# =================================================================
# === Binary Cross Entropy log loss + binary Dice-Sørensen loss ===
# =================================================================

# Sources:
# https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
# https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
# https://www.rapidtables.com/calc/math/Log_Calculator.html
# https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

import tensorflow as tf
import tensorflow.keras.backend as K

from .bBCEll import BCEll
from .bDSCl import bDSCl


@tf.keras.utils.register_keras_serializable()
class bBCEll_bDSCl(tf.keras.losses.Loss):

	def __init__(self, bDSCl_smooth=1e-6, name="BCEll_bDSCl", **kwargs):
        			
		super().__init__(name=name, **kwargs)
		self.bDSCl_smooth = bDSCl_smooth
		self.BCEll_fnc = BCEll()
		self.bDSCl_fnc = bDSCl(smooth=self.bDSCl_smooth)

	def call(self, y_true, y_pred):
		return self.BCEll_fnc(y_true, y_pred) + self.bDSCl_fnc(y_true, y_pred)

	def get_config(self):

		config = super().get_config()
		config["bDSCl_smooth"] = self.bDSCl_smooth
		return config


