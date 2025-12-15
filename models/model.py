"""Model creation utilities."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from . import (
    CausalLM,
    preserve_inv_freq,
)


def create_model(config_name: str):
    """Create a custom CausalLM from HuggingFace config.
    
    Args:
        config_name: HuggingFace model name to fetch config from
        
    Returns:
        Tuple of (model, config)
    """
    config = AutoConfig.from_pretrained(config_name)
    model = CausalLM(config)
    with preserve_inv_freq(model):
        model = model.to(dtype=torch.bfloat16, device="cuda")
    return model, config


def load_model(model_name: str | None = None, checkpoint_path: str | None = None, config_name: str | None = None):
    """Load a model either from HuggingFace or from a checkpoint.
    
    Two modes:
    - HF mode: Pass model_name to load pretrained model + tokenizer from HuggingFace
    - Checkpoint mode: Pass checkpoint_path + config_name to load weights from checkpoint
    
    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-0.5B")
        checkpoint_path: Path to saved checkpoint file
        config_name: HuggingFace model name to fetch config from (required with checkpoint_path)
        
    Returns:
        Tuple of (model, tokenizer, config)
    """
    if model_name:
        # Load directly from HuggingFace
        config = AutoConfig.from_pretrained(model_name)
            
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif checkpoint_path and config_name:
        # Load checkpoint with separate HF config
        model, config = create_model(config_name)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda"))
        tokenizer = AutoTokenizer.from_pretrained("quinnlue/cot-tokenizer")
    else:
        raise ValueError("Provide model_name OR (checkpoint_path + config_name)")
    return model, tokenizer, config

def get_model():
    try: 
        print("Attempting to load model from checkpoint...")
        model, tokenizer, config = load_model(
            checkpoint_path="checkpoints/qwen-base.pt",
            config_name="Qwen/Qwen2.5-0.5B"
        )
        print(f"Loaded checkpoint with config: {config._name_or_path}")
        return model, tokenizer, config
    except Exception as e:
        print("Attempting to load model from HuggingFace...")
        model, tokenizer, config = load_model(model_name="Qwen/Qwen2.5-0.5B")
        print(f"Loaded HF model: {config._name_or_path}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model, tokenizer, config

if __name__ == "__main__":
    model, tokenizer, config = load_model(model_name="Qwen/Qwen2.5-0.5B")
    torch.save(model.state_dict(), "checkpoints/qwen-base.pt")
