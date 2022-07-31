# =================================
# === HUBER loss + MS-SSIM loss ===
# =================================

# sources:
# https://en.wikipedia.org/wiki/Structural_similarity#Multi-Scale_SSIM
# https://www.tensorflow.org/api_docs/python/tf/image/ssim_multiscale
# https://arxiv.org/pdf/1511.08861.pdf
# https://www.tensorflow.org/api_docs/python/tf/keras/losses/Huber

import tensorflow as tf
import tensorflow.keras.backend as K
from .bMSSSIMl import bMSSSIMl

@tf.keras.utils.register_keras_serializable()
class bHUBERl_bMSSSIMl(tf.keras.losses.Loss):

	def __init__(self, msssim_filter_size=3, msssim_filter_sigma=1.5, name="bHUBERl_bMSSSIMl", **kwargs):
		super().__init__(name=name, **kwargs)
		self.msssim_filter_size = msssim_filter_size
		self.msssim_filter_sigma = msssim_filter_sigma

		self.bMSSSIMl = bMSSSIMl(filter_size=self.msssim_filter_size, filter_sigma=self.msssim_filter_sigma)
		self.bHUBERl = tf.keras.losses.Huber()

	def call(self, y_true, y_pred):
		bMSSSIM_loss = self.bMSSSIMl(y_true, y_pred)
		bHUBER_loss = self.bHUBERl(y_true, y_pred)
		return bMSSSIM_loss + bHUBER_loss

	def get_config(self):
		config = super().get_config()
		config['msssim_filter_size'] = self.msssim_filter_size
		config['msssim_filter_sigma'] = self.msssim_filter_sigma
		return config
