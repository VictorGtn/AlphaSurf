from alphasurf.tasks.s3f_pretrain.checkpointing import strip_frozen_esm_weights


def test_strip_frozen_esm_weights_preserves_trainable_state():
    checkpoint = {
        "state_dict": {
            "model.encoder.weight": object(),
            "model.esm_model.layer.weight": object(),
            "model.esm_model.layer.bias": object(),
            "model.residue_head.weight": object(),
        }
    }

    removed = strip_frozen_esm_weights(checkpoint)

    assert removed == 2
    assert set(checkpoint["state_dict"]) == {
        "model.encoder.weight",
        "model.residue_head.weight",
    }


def test_strip_frozen_esm_weights_handles_missing_state_dict():
    checkpoint = {}

    assert strip_frozen_esm_weights(checkpoint) == 0
