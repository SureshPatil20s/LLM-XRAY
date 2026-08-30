def tokenize_text(tokenizer, text: str):
    """Returns (readable_tokens, token_ids, raw_encoding) for a piece of text."""
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded["input_ids"][0]
    raw_tokens = tokenizer.convert_ids_to_tokens(ids)
    # Sub-word tokenizers mark spaces with special characters (Ġ or ▁) - clean those up
    readable = [t.replace("\u0120", " ").replace("\u2581", " ").strip() or "\u00b7" for t in raw_tokens]
    return readable, ids.tolist(), encoded
