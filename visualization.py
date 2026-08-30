import plotly.graph_objects as go
from sklearn.decomposition import PCA


def plot_embedding_pca(vectors, labels):
    """2D PCA scatter plot of token vectors (works for input embeddings or hidden states)."""
    if len(vectors) < 2:
        return None
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)
    fig = go.Figure(data=go.Scatter(
        x=reduced[:, 0], y=reduced[:, 1],
        mode="markers+text", text=labels, textposition="top center",
        marker=dict(size=12, color="#6c5ce7"),
    ))
    fig.update_layout(title="Token Vectors (PCA, 2D)", xaxis_title="PC 1", yaxis_title="PC 2")
    return fig


def plot_attention_heatmap(attention_matrix, tokens, layer, head):
    """Heatmap of attention weights: rows = query tokens, cols = key tokens."""
    fig = go.Figure(data=go.Heatmap(
        z=attention_matrix, x=tokens, y=tokens,
        colorscale="Viridis", colorbar=dict(title="Attention"),
    ))
    fig.update_layout(
        title=f"Attention Heatmap — Layer {layer + 1}, Head {head + 1}",
        xaxis_title="Key tokens", yaxis_title="Query tokens",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_top_probabilities(top_tokens, top_probs):
    """Horizontal bar chart of the model's top next-token candidates."""
    fig = go.Figure(data=go.Bar(
        x=top_probs, y=top_tokens, orientation="h", marker_color="#00b894",
        text=[f"{p*100:.1f}%" for p in top_probs], textposition="auto",
    ))
    fig.update_layout(
        title="Top Next-Token Probabilities",
        xaxis_title="Probability", yaxis=dict(autorange="reversed"),
        xaxis=dict(tickformat=".0%"),
    )
    return fig
