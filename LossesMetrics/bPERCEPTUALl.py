"""
=================================
=== Perceptual loss ===
=================================

Sources:
- https://arxiv.org/pdf/1603.08155.pdf
- https://github.com/milmor/perceptual-losses-neural-st
- https://github.com/leohu6/Perceptual-Loss-Style-Transfer/blob/master/trainer/utils/losses.py
"""

import tensorflow as tf
import tensorflow.keras.backend as K


@tf.keras.utils.register_keras_serializable()
class bPERCEPTUALl(tf.keras.losses.Loss):
    __VGG_model_instance = None
    __VGG_model_instance_shape = None

    def __init__(
        self,
        input_shape=(256, 256, 3),
        loss_weight=1.0,
        style_weights=[1.0, 1.0, 1.0, 1.0],  # mean near 1.0 for bests results
        content_weights=[0.0, 0.5, 1.0, 0.5],  # mean near 1.0 for bests results
        total_variation_weight=1.0,
        loss="MSE",  # MSE, MAE
        name="bPERCEPTUALl",
        **kwargs
    ):

        super().__init__(name=name, **kwargs)
        self.input_shape = input_shape
        self._loss_weight = loss_weight
        self._style_weights = style_weights
        self._content_weights = content_weights
        self._total_variation_weight = total_variation_weight
        self._loss = loss

        self._vgg_layers = [
            "block1_conv2",
            "block2_conv2",
            "block3_conv3",
            "block4_conv3",
        ]
        self._num_vgg_layers = len(self._vgg_layers)

        self._vgg_model = self._get_vgg_model_instance()

    def _get_vgg_model_instance(self):
        # Creates a VGG model that returns a list of intermediate output values.
        # Load our model. Load pretrained VGG, trained on ImageNet data.

        # print("VGG INSTANCE __:", bPERCEPTUALl.__VGG_model_instance, flush=True)

        # NOTA: si cambia el shape (ej: de Sec2 a Sec3) y no se hace nuevo,
        # da error incompatibilidad
        if (
            bPERCEPTUALl.__VGG_model_instance is None
        ) or bPERCEPTUALl.__VGG_model_instance_shape != self.input_shape:
            vgg = tf.keras.applications.VGG16(
                input_shape=self.input_shape, weights="imagenet", include_top=False
            )
            vgg.trainable = False
            vgg_outputs = [
                vgg.get_layer(layer_name).output for layer_name in self._vgg_layers
            ]

            bPERCEPTUALl.__VGG_model_instance = tf.keras.Model([vgg.input], vgg_outputs)
            bPERCEPTUALl.__VGG_model_instance_shape = self.input_shape

        return bPERCEPTUALl.__VGG_model_instance

    def _get_mean_std(self, input_tensor):
        # print("_get_mean_std input_tensor shape:", input_tensor.shape, flush=True)
        # assert input_tensor.shape.ndims == 4  #ndims 3 o 4

        # Compute the mean and standard deviation of a tensor.
        # NOTA: without gradients (not using Keras)
        # axes = [1, 2] #input_tensor (N, H, W, C)
        # mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=True)
        # standard_deviation = tf.sqrt(variance + 1e-6)

        mean = K.mean(input_tensor)
        standard_deviation = K.std(input_tensor)

        return mean, standard_deviation

    def _normalized(self, input_tensor):
        mean, std = self._get_mean_std(input_tensor)
        normalized = (input_tensor - mean) / std
        return normalized

    def _gram_matrix(self, input_tensor):
        # print("_gram_matrix input_tensor shape:", input_tensor.shape, flush=True)
        assert input_tensor.shape.ndims == 4

        # input_tensor = tf.cast(input_tensor, tf.float32)  # avoid mixed_precision nan
        result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
        # input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(
            input_tensor.shape[1] * input_tensor.shape[2], tf.float32
        )
        return result / (num_locations)

    def _vgg_style_outputs(self, input):
        # print("_vgg_style_outputs input shape:", input.shape, flush=True)
        assert input.shape.ndims == 3

        # Expects float input in [0,1] (output of sigmoid)
        input = input * 255.0  # [0, 1] -> [0, 255]
        input = tf.image.grayscale_to_rgb(input)

        input = tf.expand_dims(input, axis=0)
        # print("_vgg_style_outputs (added dimension) input shape:",
        # input.shape, flush=True)

        # preprocessed_input = tf.keras.applications.vgg19.preprocess_input(input)
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(input)
        model_outputs = self._vgg_model(preprocessed_input)
        # print("_vgg_style_outputs outputs (item[0]) shape:",
        # outputs[0].shape, flush=True)

        return [style_output for style_output in model_outputs]

    def _vgg_content_outputs(self, input):
        assert input.shape.ndims == 4

        # Expects float input in [0,1] (output of sigmoid)
        input = input * 255.0  # [0, 1] -> [0, 255]
        input = tf.image.grayscale_to_rgb(input)

        # input = tf.expand_dims(input, axis=0)

        # preprocessed_input = tf.keras.applications.vgg19.preprocess_input(input)
        preprocessed_input = tf.keras.applications.vgg16.preprocess_input(input)
        model_outputs = self._vgg_model(preprocessed_input)
        # print("_vgg_content_outputs outputs (item[0]) shape:",
        # outputs[0].shape, flush=True)

        return [content_output for content_output in model_outputs]

    def _selected_loss(self, y_true, y_pred):
        assert y_true.shape.ndims == y_pred.shape.ndims
        assert (y_pred.shape.ndims == 4) or (y_pred.shape.ndims == 3)
        if self._loss == "MSE":
            loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        else:  # MAE
            loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
        return loss * self._loss_weight

    def _style_loss(self, y_true, y_pred):
        # print("_style_loss y_true shape:", y_true.shape, flush=True)
        # print("_style_loss y_pred shape:", y_pred.shape, flush=True)
        assert y_true.shape.ndims == 3
        assert y_pred.shape.ndims == 3

        y_true_style_outputs = self._vgg_style_outputs(y_true)  # out shape.ndims = 4
        # print("_style_loss y_true_style_outputs[0] shape:",
        # y_true_style_outputs[0].shape, flush=True)
        y_true_style_outputs = [
            self._gram_matrix(style) for style in y_true_style_outputs
        ]  # out shape.ndims = 3
        # print("_style_loss y_true_style_outputs[0] (after gram_matrix) shape:",
        # y_true_style_outputs[0].shape, flush=True)

        y_pred_style_outputs = self._vgg_style_outputs(y_pred)  # out shape.ndims = 4
        y_pred_style_outputs = [
            self._gram_matrix(style) for style in y_pred_style_outputs
        ]  # out shape.ndims = 3

        # print("_style_loss y_true_style_outputs shape:",
        # y_true_style_outputs.shape, flush=True)
        # print("_style_loss y_pred_style_outputs shape:",
        # y_pred_style_outputs.shape, flush=True)

        style_loss = 0
        for true, pred, weight in zip(
            y_true_style_outputs, y_pred_style_outputs, self._style_weights
        ):
            # print("_style_loss FOR true shape:", true.shape, flush=True)

            true_normalized = self._normalized(true)
            pred_normalized = self._normalized(pred)

            # print("_style_loss true mean std shapes:",
            # true_mean.shape, true_std.shape)
            # tf.print("_style_loss true_style mean std:", true_mean, true_std)
            # tf.print("_style_loss pred_style mean std:", pred_mean, pred_std)

            # style_loss += weight *
            # tf.keras.losses.MeanSquaredError()(true_normalized, pred_normalized)
            style_loss += weight * self._selected_loss(true_normalized, pred_normalized)

        style_loss /= self._num_vgg_layers
        # print("_style_loss style_loss shape dtype:",
        # style_loss.shape, style_loss.dtype, flush=True)

        return style_loss

    def _content_loss(self, y_true, y_pred):
        assert y_true.shape.ndims == 4
        assert y_pred.shape.ndims == 4

        y_true_content_outputs = self._vgg_content_outputs(y_true)
        y_pred_content_outputs = self._vgg_content_outputs(y_pred)

        content_loss = 0
        for true, pred, weight in zip(
            y_true_content_outputs, y_pred_content_outputs, self._content_weights
        ):
            # print("_content_loss FOR true shape:", true.shape, flush=True)

            true_normalized = self._normalized(true)
            pred_normalized = self._normalized(pred)

            # print("_content_loss true mean std shapes:",
            # true_mean.shape, true_std.shape)
            # tf.print("_content_loss true_style mean std:", true_mean, true_std)
            # tf.print("_content_loss pred_style mean std:", pred_mean, pred_std)

            # content_loss += weight *
            # tf.keras.losses.MeanSquaredError()(true_normalized, pred_normalized)
            content_loss += weight * self._selected_loss(
                true_normalized, pred_normalized
            )

        content_loss /= self._num_vgg_layers
        # print("_content_loss content_loss shape dtype:",
        # content_loss.shape, content_loss.dtype, flush=True)

        return content_loss

    def _total_variation_loss(self, y_pred):
        assert y_pred.shape.ndims == 4
        # high frequency variation
        x_deltas = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        y_deltas = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        total_variation_loss = tf.reduce_mean(
            tf.reduce_mean(tf.abs(x_deltas)) + tf.reduce_mean(tf.abs(y_deltas))
        )
        return total_variation_loss * self._total_variation_weight

    def call(self, y_true, y_pred):

        # --- direct loss
        loss_loss = self._selected_loss(y_true, y_pred)

        # --- style loss
        style_loss = tf.map_fn(
            lambda y: self._style_loss(y[0], y[1]),
            (y_true, y_pred),
            fn_output_signature=tf.float32,
        )
        # print("call style_loss shape dtype:",
        # style_loss.shape, style_loss.dtype, flush=True)
        style_loss = tf.reduce_mean(style_loss)  # cambio de shape
        # print("call style_loss shape dtype:",
        # style_loss.shape, style_loss.dtype, flush=True)

        # --- content loss
        content_loss = self._content_loss(y_true, y_pred)
        # print("call content_loss shape dtype:",
        # content_loss.shape, content_loss.dtype, flush=True)

        # --- total variation loss
        total_variation_loss = self._total_variation_loss(y_pred)

        # tf.print("loss style/content/tv:",
        # style_loss, content_loss, total_variation_loss)
        # print("style_loss / contento_loss / tv_loss:",
        # style_loss, content_loss, total_variation_loss)

        # --- loss
        return loss_loss + style_loss + content_loss + total_variation_loss

    def get_config(self):

        config = super().get_config()
        config["input_shape"] = self.input_shape
        config["loss_weight"] = self._mse_weight
        config["style_weights"] = self._style_weights
        config["content_weights"] = self._content_weights
        config["total_variation_weight"] = self._total_variation_weight
        config["loss"] = self._loss
        return config
