"""Test forward pass equivalence between custom and HuggingFace models."""

import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from models import CausalLM, load_and_convert_model


# Test prompts for equivalence testing
TEST_PROMPTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "In mathematics, a prime number is",
    "def fibonacci(n):",
]

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer once for all tests."""
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


@pytest.fixture(scope="module")
def hf_model():
    """Load HuggingFace model once for all tests."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def custom_model(hf_model):
    """Load custom model with HuggingFace weights."""
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config._attn_implementation = "sdpa"
    
    model = CausalLM(config)
    model = load_and_convert_model(model, hf_model, dtype=torch.bfloat16, device="cuda")
    model.eval()
    return model


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
def test_forward_pass_equivalence(prompt, tokenizer, hf_model, custom_model):
    """
    Test that custom model produces identical outputs to HuggingFace model.
    
    Args:
        prompt: Test input text
        tokenizer: Shared tokenizer fixture
        hf_model: HuggingFace model fixture
        custom_model: Custom model fixture with loaded weights
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(hf_model.device)
    
    with torch.no_grad():
        hf_output = hf_model(input_ids)
        hf_logits = hf_output.logits
        
        custom_output = custom_model(input_ids)
        custom_logits = custom_output.logits
    
    # Compare logits with tolerance for bfloat16
    max_diff = (hf_logits.float() - custom_logits.float()).abs().max().item()
    mean_diff = (hf_logits.float() - custom_logits.float()).abs().mean().item()
    
    # Assert equivalence with reasonable tolerance
    assert torch.allclose(
        hf_logits.float(), 
        custom_logits.float(), 
        atol=1e-4, 
        rtol=1e-3
    ), f"Logits mismatch for prompt '{prompt[:40]}...': max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"


def test_exact_match_zero_diff(tokenizer, hf_model, custom_model):
    """
    Test that logits are exactly identical (zero difference) for basic input.
    
    This stricter test verifies that weight loading is perfect.
    """
    prompt = "Hello world"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(hf_model.device)
    
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        custom_logits = custom_model(input_ids).logits
    
    max_diff = (hf_logits - custom_logits).abs().max().item()
    
    assert max_diff == 0.0, f"Expected exact match but got max_diff={max_diff:.2e}"


def test_generation_equivalence(tokenizer, hf_model, custom_model):
    """
    Test that both models generate identical token sequences.
    """
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(hf_model.device)
    
    # Use greedy decoding for deterministic comparison
    gen_kwargs = {
        "max_new_tokens": 20,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    with torch.no_grad():
        hf_output = hf_model.generate(input_ids, **gen_kwargs)
        custom_output = custom_model.generate(input_ids, **gen_kwargs)
    
    assert torch.equal(hf_output, custom_output), (
        f"Generated sequences differ:\n"
        f"HF: {tokenizer.decode(hf_output[0])}\n"
        f"Custom: {tokenizer.decode(custom_output[0])}"
    )


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])

