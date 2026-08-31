import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# A small but noticeably more capable instruct model than 135M, kept inside
# free-tier RAM (~1GB) by loading in bfloat16 (half precision) instead of
# float32. Running locally on your own machine with no RAM limit? Swap this
# for "Qwen/Qwen2.5-0.5B-Instruct" or "Qwen/Qwen2.5-1.5B-Instruct" (the model
# from the original brief) by typing it into the app's sidebar — no code change needed.
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


@st.cache_resource(show_spinner="Loading model (first run only, then cached)...")
def load_model(model_name: str = DEFAULT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,   # half the memory footprint of float32
        attn_implementation="eager",  # needed so attention weights are actually returned
    )
    model.eval()
    return tokenizer, model


def run_forward_pass(tokenizer, model, prompt: str):
    """One forward pass through the model -> logits, hidden states, and attentions
    for every layer, in a single shot."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    return inputs, outputs


def generate_step_by_step(tokenizer, model, prompt, max_new_tokens=20,
                           temperature=0.7, top_k=50, top_p=0.95):
    """Generates a response one token at a time, recording the top candidate
    tokens considered at each step (used for the 'watch it generate' view)."""
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    steps = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :].float()  # bfloat16 -> float32

        scaled = logits / max(temperature, 1e-5)

        if top_k > 0:
            topk_vals, topk_idx = torch.topk(scaled, top_k)
            filtered = torch.full_like(scaled, float("-inf"))
            filtered[topk_idx] = topk_vals
            scaled = filtered

        probs = torch.softmax(scaled, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = (cumulative > top_p).nonzero()
        cutoff_idx = int(cutoff[0].item()) + 1 if len(cutoff) else len(sorted_probs)
        keep_idx = sorted_idx[:cutoff_idx]
        mask = torch.zeros_like(probs)
        mask[keep_idx] = probs[keep_idx]
        probs = mask / mask.sum()

        next_token = torch.multinomial(probs, num_samples=1)
        top5_vals, top5_idx = torch.topk(probs, min(5, probs.shape[-1]))

        steps.append({
            "text_so_far": tokenizer.decode(input_ids[0], skip_special_tokens=True),
            "new_token": tokenizer.decode([next_token.item()]),
            "top5": [(tokenizer.decode([i]), float(v)) for i, v in zip(top5_idx, top5_vals)],
        })

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    final_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return final_text, steps
