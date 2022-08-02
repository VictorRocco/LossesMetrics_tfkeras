# =======================================================
# === MS-SSIM loss ======================================
# === Multi Scale Structural Similarity Index Measure ===
# =======================================================

# sources:
# https://en.wikipedia.org/wiki/Structural_similarity#Multi-Scale_SSIM
# https://www.tensorflow.org/api_docs/python/tf/image/ssim_multiscale
# https://arxiv.org/pdf/1511.08861.pdf

import tensorflow as tf
import tensorflow.keras.backend as K
from .bMSSSIMm import bMSSSIMm

@tf.keras.utils.register_keras_serializable()
class bMSSSIMl(tf.keras.losses.Loss):

	def __init__(self, filter_size=11, filter_sigma=1.5, max_val=1.0, name="bMSSSIMl", **kwargs):
		super().__init__(name=name, **kwargs)
		self.filter_size = filter_size
		self.filter_sigma = filter_sigma
		self.max_val = max_val

		self.bMSSSIM_metric = bMSSSIMm(filter_size=self.filter_size, filter_sigma=self.filter_sigma,
									   max_val=self.max_val)

	def call(self, y_true, y_pred):
		bMSSSIMl_loss = 1.0 - self.bMSSSIM_metric(y_true, y_pred)
		return bMSSSIMl_loss

	def get_config(self):
		config = super().get_config()
		config['filter_size'] = self.filter_size
		config['filter_sigma'] = self.filter_sigma
		config['max_value'] = self.max_val
		return config


