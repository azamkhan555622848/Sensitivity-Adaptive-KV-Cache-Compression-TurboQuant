"""Multi-model loader for evaluation."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name: str, dtype: str = "auto", device_map: str = "auto"):
    if dtype == "auto":
        if any(n in model_name.lower() for n in ["llama", "mistral", "qwen"]):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map)
    model.eval()
    return model, tokenizer

def get_model_info(model) -> dict:
    config = model.config
    # Handle nested configs (e.g., Gemma-4 has text_config)
    cfg = getattr(config, "text_config", config)
    n_layers = cfg.num_hidden_layers
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = cfg.hidden_size // cfg.num_attention_heads
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    return {"n_layers": n_layers, "head_dim": head_dim, "n_heads": n_heads,
            "n_kv_heads": n_kv_heads, "hidden_size": cfg.hidden_size,
            "max_position_embeddings": getattr(cfg, "max_position_embeddings", 4096)}
