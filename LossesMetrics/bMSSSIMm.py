# =======================================================
# === MS-SSIM metric ====================================
# === Multi Scale Structural Similarity Index Measure ===
# =======================================================

# sources:
# https://en.wikipedia.org/wiki/Structural_similarity#Multi-Scale_SSIM
# https://www.tensorflow.org/api_docs/python/tf/image/ssim_multiscale
# https://arxiv.org/pdf/1511.08861.pdf

import tensorflow as tf
import tensorflow.keras.backend as K

@tf.keras.utils.register_keras_serializable()
class bMSSSIMm(tf.keras.metrics.Metric):

	def __init__(self, filter_size=3, filter_sigma=1.5, max_val=1.0, name="bMSSSIMm", **kwargs):
		super().__init__(name=name, **kwargs)
		self.filter_size = filter_size
		self.filter_sigma = filter_sigma
		self.max_val = max_val

	def update_state(self, y_true, y_pred, sample_weight=None):
		self.y_true = y_true
		self.y_pred = y_pred

	def result(self):
		return K.mean(tf.image.ssim_multiscale(self.y_true, self.y_pred,
											   filter_size=self.filter_size, filter_sigma=self.filter_sigma,
											   max_val=self.max_val))

	def get_config(self):
		config = super().get_config()
		config['filter_size'] = self.filter_size
		config['filter_sigma'] = self.filter_sigma
		config['max_value'] = self.max_val
		return config


