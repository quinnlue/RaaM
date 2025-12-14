"""Utility functions for model loading and conversion."""

import torch
from contextlib import contextmanager


def load_hf_weights(custom_model, hf_model):
    """
    Load weights from a HuggingFace model into the custom model.
    
    Args:
        custom_model: Custom CausalLM instance
        hf_model: HuggingFace AutoModelForCausalLM instance
    
    Returns:
        custom_model with loaded weights
    """
    hf_state = hf_model.state_dict()
    custom_model.load_state_dict(hf_state, strict=True)
    
    print(f"Successfully loaded {len(hf_state)} parameter tensors")
    return custom_model


@contextmanager
def preserve_inv_freq(model):
    """
    Context manager to preserve rotary embedding inv_freq during dtype conversion.
    
    The inv_freq tensor should remain in float32 for numerical accuracy,
    even when the rest of the model is converted to bfloat16/float16.
    
    Usage:
        with preserve_inv_freq(model):
            model = model.to(dtype=torch.bfloat16, device="cuda")
    
    Args:
        model: Model with rotary embeddings at model.model.rotary_emb.inv_freq
    """
    # Save inv_freq before conversion
    inv_freq = model.model.rotary_emb.inv_freq.clone()
    original_device = inv_freq.device
    
    yield
    
    # Restore inv_freq to float32 on the model's current device
    target_device = next(model.parameters()).device
    model.model.rotary_emb.inv_freq = inv_freq.to(device=target_device, dtype=torch.float32)


def load_and_convert_model(custom_model, hf_model, dtype=torch.bfloat16, device="cuda"):
    """
    Load HuggingFace weights and convert model to specified dtype/device.
    
    Handles the inv_freq preservation automatically.
    
    Args:
        custom_model: Custom CausalLM instance
        hf_model: HuggingFace AutoModelForCausalLM instance
        dtype: Target dtype (default: torch.bfloat16)
        device: Target device (default: "cuda")
    
    Returns:
        custom_model with loaded weights, converted to target dtype/device
    """
    custom_model = load_hf_weights(custom_model, hf_model)
    
    with preserve_inv_freq(custom_model):
        custom_model = custom_model.to(dtype=dtype, device=device)
    
    return custom_model

