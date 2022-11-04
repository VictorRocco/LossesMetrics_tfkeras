# ===========================================
# === SSIM loss =============================
# === Structural Similarity Index Measure ===
# ===========================================

# sources:
# https://en.wikipedia.org/wiki/Structural_similarity
# https://www.tensorflow.org/api_docs/python/tf/image/ssim

import tensorflow as tf

from .bSSIMm import bSSIMm


@tf.keras.utils.register_keras_serializable()
class bSSIMl(tf.keras.losses.Loss):
    def __init__(
        self, filter_size=11, filter_sigma=1.5, max_val=1.0, name="bSSIMl", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.max_val = max_val

        self.bSSIM_metric = bSSIMm(
            filter_size=self.filter_size,
            filter_sigma=self.filter_sigma,
            max_val=self.max_val,
        )

    def call(self, y_true, y_pred):
        bSSIMl_loss = 1.0 - self.bSSIM_metric(y_true, y_pred)
        return bSSIMl_loss

    def get_config(self):
        config = super().get_config()
        config["filter_size"] = self.filter_size
        config["filter_sigma"] = self.filter_sigma
        config["max_value"] = self.max_val
        return config
