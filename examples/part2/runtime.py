"""Optional pretrained weight/tokenizer loader; not part of the forward algorithm."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .forward import Config, Qwen3


def load_pretrained(path, revision=None):
    hf_config = AutoConfig.from_pretrained(path, revision=revision, trust_remote_code=False)
    config = Config.from_hf(hf_config)  # Reject unsupported architectures BEFORE loading weights.
    reference = AutoModelForCausalLM.from_pretrained(
        path, revision=revision, torch_dtype=torch.float32,
        attn_implementation="eager", trust_remote_code=False)
    # Meta construction avoids allocating another full model before assigning weights.
    with torch.device("meta"):
        model = Qwen3(config)
    model.load_state_dict(reference.state_dict(), strict=True, assign=True)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision, trust_remote_code=False)
    return model.eval(), tokenizer
