"""
============================
=== BCE loss + SSIM loss ===
============================

Sources:
- https://en.wikipedia.org/wiki/Structural_similarity
- https://www.tensorflow.org/api_docs/python/tf/image/ssim
"""

import tensorflow as tf

from .bBCEl import bBCEl
from .bSSIMl import bSSIMl


@tf.keras.utils.register_keras_serializable()
class bBCEl_bSSIMl(tf.keras.losses.Loss):
    def __init__(
        self, ssim_filter_size=11, ssim_filter_sigma=1.5, name="bBCEl_bSSIMl", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.ssim_filter_size = ssim_filter_size
        self.ssim_filter_sigma = ssim_filter_sigma

        self.f_bSSIMl = bSSIMl(
            filter_size=self.ssim_filter_size, filter_sigma=self.ssim_filter_sigma
        )
        self.f_bBCEl = bBCEl()

    def call(self, y_true, y_pred):
        bSSIM_loss = self.f_bSSIMl(y_true, y_pred)
        bBCE_loss = self.f_bBCEl(y_true, y_pred)
        return bSSIM_loss + bBCE_loss

    def get_config(self):
        config = super().get_config()
        config["ssim_filter_size"] = self.ssim_filter_size
        config["ssim_filter_sigma"] = self.ssim_filter_sigma
        return config
