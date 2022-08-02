# ============================
# === BCE loss + SSIM loss ===
# ============================

# sources:
# https://en.wikipedia.org/wiki/Structural_similarity
# https://www.tensorflow.org/api_docs/python/tf/image/ssim

import tensorflow as tf
import tensorflow.keras.backend as K
from .bSSIMl import bSSIMl
from .BCEl import BCEl

@tf.keras.utils.register_keras_serializable()
class BCEl_bSSIMl(tf.keras.losses.Loss):

	def __init__(self, msssim_filter_size=11, msssim_filter_sigma=1.5, name="BCEl_bSSIMl", **kwargs):
		super().__init__(name=name, **kwargs)
		self.msssim_filter_size = msssim_filter_size
		self.msssim_filter_sigma = msssim_filter_sigma

		self.f_bSSIMl = bSSIMl(filter_size=self.msssim_filter_size, filter_sigma=self.msssim_filter_sigma)
		self.f_BCEl = BCEl()

	def call(self, y_true, y_pred):
		bSSIM_loss = self.f_bSSIMl(y_true, y_pred)
		BCE_loss = self.f_BCEl(y_true, y_pred)
		return bSSIM_loss + BCE_loss

	def get_config(self):
		config = super().get_config()
		config['msssim_filter_size'] = self.msssim_filter_size
		config['msssim_filter_sigma'] = self.msssim_filter_sigma
		return config
