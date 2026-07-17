ESM_STATE_PREFIX = "model.esm_model."


def strip_frozen_esm_weights(checkpoint):
    state_dict = checkpoint.get("state_dict", {})
    esm_keys = [key for key in state_dict if key.startswith(ESM_STATE_PREFIX)]
    for key in esm_keys:
        del state_dict[key]
    return len(esm_keys)
