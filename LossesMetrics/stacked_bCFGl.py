"""
========================================
=== stacked binary CONFIGURABLE loss ===
========================================

For multi output model with stacked outputs
Example: stacked outputs in model: OUTPUT +OUT0 +OUT... +OUTN
Code in model: return Model(inputs, tf.stack([OUTPUT, OUT0, ... , OUTN]))
loss_fnc parameter example: [[bBCEl(), bSSIMl(), bDSCl()], bBCEl(), ... , bBCEl()]
in OUPUT we apply bBCEl+bSSIMl+bDSCl, in OUT0 bBCEl, ... , in OUTN bBCEl.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class stacked_bCFGl(tf.keras.losses.Loss):

    def __init__(self, loss_fnc=None, name="stacked_bCFGl", **kwargs):
        assert (loss_fnc is not None), "missing loss_fnc parameter (list of losses functions)"
        assert isinstance(loss_fnc, list), "loss_fnc parameter is not a list (of losses functions)"
        super().__init__(name=name, **kwargs)
        self.loss_fnc = loss_fnc

    def call(self, y_true, y_pred):
        y_pred_unstacked = tf.unstack(y_pred)
        loss = 0.0

        if len(y_pred_unstacked) != len(self.loss_fnc):
            raise ValueError("len of y_pred unstacked != len of list of losses")

        for y_pred_i, loss_fnc_i in zip(y_pred_unstacked, self.loss_fnc):
            if isinstance(loss_fnc_i, list):
                for y_pred_i, loss_fnc_ij in zip(y_pred_unstacked, loss_fnc_i):
                    loss += loss_fnc_ij(y_true, y_pred_i)
            else:
                loss += loss_fnc_i(y_true, y_pred_i)

        return loss

    def get_config(self):

        config = super().get_config()
        config["loss_fnc"] = self.loss_fnc
        return config
