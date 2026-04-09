"""Multi-model loader for evaluation."""
import torch
import transformers.generation.utils as _gu
# Compatibility shim: Nemotron-NAS custom code (pinned to transformers 4.44.x)
# imports `NEED_SETUP_CACHE_CLASSES_MAPPING` from transformers.generation.utils,
# which was removed in newer transformers versions. Inject an empty dict so the
# import succeeds. The mapping is only consulted by legacy cache-setup code that
# we don't exercise.
if not hasattr(_gu, "NEED_SETUP_CACHE_CLASSES_MAPPING"):
    _gu.NEED_SETUP_CACHE_CLASSES_MAPPING = {}
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name: str, dtype: str = "auto", device_map: str = "auto"):
    if dtype == "auto":
        if any(n in model_name.lower() for n in ["llama", "mistral", "qwen", "nemotron"]):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = getattr(torch, dtype)
    # Nemotron-NAS and some custom architectures require trust_remote_code
    needs_trust = any(n in model_name.lower() for n in ["nemotron"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=needs_trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device_map,
        trust_remote_code=needs_trust,
    )
    model.eval()
    return model, tokenizer

def _nemotron_nas_info(config) -> dict:
    """Extract info from a Nemotron-NAS (heterogeneous) config.
    Uses the n_heads_in_group from the first non-no_op attention layer."""
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // n_heads)
    # Find first real attention layer
    n_heads_in_group = None
    real_attn_layers = []
    for i, bc in enumerate(config.block_configs):
        attn = getattr(bc, "attention", None)
        if attn is None:
            continue
        no_op = getattr(attn, "no_op", False)
        replace_linear = getattr(attn, "replace_with_linear", False)
        if no_op or replace_linear:
            continue
        real_attn_layers.append(i)
        if n_heads_in_group is None:
            n_heads_in_group = getattr(attn, "n_heads_in_group", None)
    if n_heads_in_group is None or n_heads_in_group == 0:
        n_kv_heads = n_heads  # fallback: MHA
    else:
        n_kv_heads = n_heads // n_heads_in_group
    return {
        "n_layers": n_layers,
        "head_dim": head_dim,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "hidden_size": config.hidden_size,
        "max_position_embeddings": getattr(config, "max_position_embeddings", 4096),
        "nemotron_real_attn_layers": real_attn_layers,
        "nemotron_n_heads_in_group": n_heads_in_group,
    }

def get_model_info(model) -> dict:
    config = model.config
    # Handle nested configs (e.g., Gemma-4 has text_config)
    cfg = getattr(config, "text_config", config)
    # Nemotron-NAS has a heterogeneous layer structure
    if getattr(cfg, "model_type", None) == "nemotron-nas" and hasattr(cfg, "block_configs"):
        return _nemotron_nas_info(cfg)
    n_layers = cfg.num_hidden_layers
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = cfg.hidden_size // cfg.num_attention_heads
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    if n_kv_heads is None:
        n_kv_heads = n_heads
    return {"n_layers": n_layers, "head_dim": head_dim, "n_heads": n_heads,
            "n_kv_heads": n_kv_heads, "hidden_size": cfg.hidden_size,
            "max_position_embeddings": getattr(cfg, "max_position_embeddings", 4096)}
