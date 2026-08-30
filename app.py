import streamlit as st
import torch

from model import load_model, run_forward_pass, generate_step_by_step, DEFAULT_MODEL
from tokenizer_utils import tokenize_text
from visualization import plot_embedding_pca, plot_attention_heatmap, plot_top_probabilities

st.set_page_config(page_title="LLM X-Ray", page_icon="\U0001FA7B", layout="wide")
st.title("\U0001FA7B LLM X-Ray")
st.caption("Enter a prompt and watch every internal step an LLM takes to answer it.")

with st.sidebar:
    st.header("\u2699\ufe0f Settings")
    model_name = st.text_input(
        "Model (Hugging Face repo id)", value=DEFAULT_MODEL,
        help="Any causal LM works. Smaller models load faster on CPU.",
    )
    st.divider()
    st.subheader("Generation controls")
    max_new_tokens = st.slider("Max new tokens", 1, 60, 20)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7)
    top_k = st.slider("Top-K", 0, 100, 50)
    top_p = st.slider("Top-P", 0.1, 1.0, 0.95)

tokenizer, model = load_model(model_name)

prompt = st.text_area("Enter your prompt", value="What is AI?", height=80)
run = st.button("\U0001F50D Run X-Ray", type="primary")

if run and prompt.strip():
    with st.spinner("Running forward pass through the model..."):
        tokens, token_ids, encoded = tokenize_text(tokenizer, prompt)
        inputs, outputs = run_forward_pass(tokenizer, model, prompt)

    tab_names = [
        "1. Tokenization", "2. Embeddings", "3. Architecture",
        "4. Attention", "5. Hidden States", "6. Logits & Probabilities",
        "7. Generate Token-by-Token",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.subheader("Input \u2192 Tokenizer \u2192 Tokens \u2192 Token IDs")
        cols = st.columns(len(tokens))
        for c, tok, tid in zip(cols, tokens, token_ids):
            c.metric(label=tok, value=tid)

    with tabs[1]:
        st.subheader("Token Embeddings")
        embeddings = model.get_input_embeddings()(inputs["input_ids"])[0].detach().numpy()
        st.write(f"Embedding dimension: **{embeddings.shape[1]}**")
        st.write("First 8 values of each token's embedding vector:")
        st.dataframe({tok: vec[:8] for tok, vec in zip(tokens, embeddings)})
        fig = plot_embedding_pca(embeddings, tokens)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 tokens for a PCA plot \u2014 try a longer prompt.")

    with tabs[2]:
        st.subheader("Model Architecture")
        n_layers = model.config.num_hidden_layers
        st.write(f"This model has **{n_layers} transformer layers**.")
        flow = ["Input Embeddings"] + [f"Layer {i}" for i in range(1, n_layers + 1)] + ["Output"]
        if len(flow) > 8:
            shown = " \u2192 ".join(flow[:4]) + "  \u2192 \u22ef \u2192  " + " \u2192 ".join(flow[-3:])
        else:
            shown = " \u2192 ".join(flow)
        st.markdown(f"`{shown}`")
        st.caption("Explore any specific layer's internals in the Attention and Hidden States tabs.")

    with tabs[3]:
        st.subheader("Attention Heatmap")
        n_layers = len(outputs.attentions)
        n_heads = outputs.attentions[0].shape[1]
        c1, c2 = st.columns(2)
        layer_idx = c1.slider("Layer", 1, n_layers, min(6, n_layers)) - 1
        head_idx = c2.slider("Head", 1, n_heads, 1) - 1
        attn = outputs.attentions[layer_idx][0, head_idx].detach().numpy()
        st.plotly_chart(plot_attention_heatmap(attn, tokens, layer_idx, head_idx), use_container_width=True)
        st.caption("Brighter cells = the row token 'attends' more strongly to the column token.")

    with tabs[4]:
        st.subheader("Hidden States Across Layers")
        n_hidden = len(outputs.hidden_states)
        layer_idx = st.slider("Layer (0 = raw embeddings)", 0, n_hidden - 1, min(10, n_hidden - 1))
        hidden = outputs.hidden_states[layer_idx][0].detach().numpy()
        st.write(f"Shape at this layer: `{hidden.shape}` (tokens \u00d7 hidden dimension)")
        fig = plot_embedding_pca(hidden, tokens)
        if fig:
            fig.update_layout(title=f"Hidden States PCA \u2014 Layer {layer_idx}")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        st.subheader("Next-Token Prediction")
        st.caption("Hidden State \u2192 LM Head \u2192 Logits \u2192 Softmax \u2192 Probabilities")
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, 8)
        top_tokens = [tokenizer.decode([i]) for i in top_idx]
        st.plotly_chart(plot_top_probabilities(top_tokens, top_probs.tolist()), use_container_width=True)

    with tabs[6]:
        st.subheader("Watch the Response Get Generated")
        if st.button("\u25b6\ufe0f Generate token-by-token"):
            with st.spinner("Generating..."):
                final_text, steps = generate_step_by_step(
                    tokenizer, model, prompt,
                    max_new_tokens=max_new_tokens, temperature=temperature,
                    top_k=top_k, top_p=top_p,
                )
            for i, step in enumerate(steps, 1):
                label = f"Step {i}: + \"{step['new_token']}\""
                with st.expander(label):
                    st.write(f"Text so far: `{step['text_so_far']}`")
                    st.write("Top candidates considered at this step:")
                    for tok, p in step["top5"]:
                        st.write(f"- `{tok}` \u2014 {p*100:.1f}%")
            st.success("Final response:")
            st.write(final_text)
else:
    st.info("Enter a prompt above and click **Run X-Ray** to begin.")
