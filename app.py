import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    f1_score,
    classification_report,
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Network Trainer",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Mono', monospace;
    }
    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0;
    }
    .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-top: 4px;
        margin-bottom: 32px;
    }
    .metric-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 12px;
        padding: 20px 24px;
        color: white;
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-label {
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 1rem;
        color: #0f172a;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin-bottom: 16px;
        margin-top: 24px;
    }
    .info-box {
        background: #f0f9ff;
        border-left: 4px solid #38bdf8;
        border-radius: 4px;
        padding: 12px 16px;
        font-size: 0.9rem;
        color: #0369a1;
        margin-bottom: 16px;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 0;
        font-family: 'Space Mono', monospace;
        font-size: 0.9rem;
        font-weight: 700;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.88;
    }
    div[data-testid="stSidebar"] {
        background: #0f172a;
    }
    div[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label,
    div[data-testid="stSidebar"] .stNumberInput label {
        color: #94a3b8 !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .dataset-chip {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

DATASETS = {
    "Iris (3 classes)": "iris",
    "Wine (3 classes)": "wine",
    "Breast Cancer (binary)": "breast_cancer",
    "Circles (nonlinear)": "circles",
    "Moons (nonlinear)": "moons",
    "Blobs (random clusters)": "blobs",
}

OPTIMIZERS = {
    "Adam": "adam",
    "SGD": "sgd",
    "L-BFGS": "lbfgs",
}


@st.cache_data
def load_dataset(name: str):
    """Return (X, y, feature_names, class_names)."""
    if name == "iris":
        d = datasets.load_iris()
        return d.data, d.target, d.feature_names, d.target_names
    elif name == "wine":
        d = datasets.load_wine()
        return d.data, d.target, d.feature_names, d.target_names
    elif name == "breast_cancer":
        d = datasets.load_breast_cancer()
        return d.data, d.target, d.feature_names, d.target_names
    elif name == "circles":
        X, y = datasets.make_circles(n_samples=500, noise=0.1, random_state=42)
        return X, y, ["x1", "x2"], ["Class 0", "Class 1"]
    elif name == "moons":
        X, y = datasets.make_moons(n_samples=500, noise=0.15, random_state=42)
        return X, y, ["x1", "x2"], ["Class 0", "Class 1"]
    else:  # blobs
        X, y = datasets.make_blobs(n_samples=500, centers=4, random_state=42)
        return X, y, ["x1", "x2"], ["A", "B", "C", "D"]


def train_model(X_train, y_train, hidden_neurons, optimizer, lr, epochs):
    """Train MLP and capture per-epoch loss."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    loss_curve = []
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_neurons,),
        solver=optimizer,
        learning_rate_init=lr,
        max_iter=1,          # we iterate manually to capture loss
        warm_start=True,
        random_state=42,
    )

    progress = st.progress(0, text="Training…")
    for epoch in range(1, epochs + 1):
        model.fit(X_train_sc, y_train)
        loss_curve.append(model.loss_)
        progress.progress(epoch / epochs, text=f"Epoch {epoch}/{epochs}  —  loss: {model.loss_:.4f}")

    progress.empty()
    return model, scaler, loss_curve


def evaluate_model(model, scaler, X_test, y_test, class_names):
    X_test_sc = scaler.transform(X_test)
    y_pred = model.predict(X_test_sc)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    return y_pred, acc, prec, f1, cm, report


def plot_loss_curve(loss_curve):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(range(1, len(loss_curve) + 1), loss_curve,
            color="#0ea5e9", linewidth=2.5, label="Training Loss")
    ax.fill_between(range(1, len(loss_curve) + 1), loss_curve,
                    alpha=0.12, color="#0ea5e9")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training Loss Curve", fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(max(4, len(class_names) * 1.4),
                                    max(3.5, len(class_names) * 1.2)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="#e2e8f0",
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_data_preview(X, y, class_names, feature_names):
    """2-D scatter using first two features."""
    palette = ["#0ea5e9", "#6366f1", "#f59e0b", "#10b981"]
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, name in enumerate(class_names):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1],
                   color=palette[i % len(palette)],
                   label=name, alpha=0.65, edgecolors="white",
                   linewidth=0.4, s=40)
    ax.set_xlabel(feature_names[0], fontsize=10)
    ax.set_ylabel(feature_names[1], fontsize=10)
    ax.set_title("Dataset Preview (first 2 features)", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 NN Trainer")
    st.markdown("---")

    st.markdown("### Dataset")
    dataset_label = st.selectbox("Choose a dataset", list(DATASETS.keys()))

    st.markdown("### Model Parameters")
    optimizer_label = st.selectbox("Optimizer", list(OPTIMIZERS.keys()))
    lr = st.select_slider(
        "Learning Rate",
        options=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5],
        value=0.001,
    )
    hidden_neurons = st.slider("Hidden Neurons", min_value=4, max_value=256,
                               value=64, step=4)
    epochs = st.slider("Epochs", min_value=10, max_value=500,
                       value=100, step=10)

    st.markdown("---")
    run_btn = st.button("▶  Train Model")


# ── Main Area ──────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🧠 Neural Network Trainer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Train, evaluate, and visualize a neural network — no code required.</p>',
            unsafe_allow_html=True)

# Dataset info section
X, y, feature_names, class_names = load_dataset(DATASETS[dataset_label])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

col_info, col_preview = st.columns([1, 1], gap="large")

with col_info:
    st.markdown('<p class="section-header">📊 Dataset Info</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        <b>{dataset_label}</b><br>
        {len(X)} samples &nbsp;·&nbsp; {X.shape[1]} features &nbsp;·&nbsp;
        {len(class_names)} classes<br>
        Train: <b>{len(X_train)}</b> samples (70%) &nbsp;·&nbsp;
        Test: <b>{len(X_test)}</b> samples (30%)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Classes:**")
    chips = "".join(f'<span class="dataset-chip">{c}</span>' for c in class_names)
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown("<br>**Selected Parameters:**", unsafe_allow_html=True)
    params_df = pd.DataFrame({
        "Parameter": ["Optimizer", "Learning Rate", "Hidden Neurons", "Epochs"],
        "Value": [optimizer_label, lr, hidden_neurons, epochs],
    })
    st.dataframe(params_df, hide_index=True, use_container_width=True)

with col_preview:
    st.markdown('<p class="section-header">🔍 Data Preview</p>', unsafe_allow_html=True)
    st.pyplot(plot_data_preview(X, y, class_names, feature_names))

st.markdown("---")

# ── Training Section ───────────────────────────────────────────────────────────

if run_btn:
    st.markdown('<p class="section-header">⚙️ Training…</p>', unsafe_allow_html=True)

    model, scaler, loss_curve = train_model(
        X_train, y_train,
        hidden_neurons=hidden_neurons,
        optimizer=OPTIMIZERS[optimizer_label],
        lr=lr,
        epochs=epochs,
    )

    y_pred, acc, prec, f1, cm, report = evaluate_model(
        model, scaler, X_test, y_test, class_names
    )

    # ── Metrics ──
    st.markdown('<p class="section-header">📈 Results</p>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{acc:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Precision (weighted)</div>
            <div class="metric-value">{prec:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">F1 Score (weighted)</div>
            <div class="metric-value">{f1:.1%}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Plots ──
    col_loss, col_cm = st.columns(2, gap="large")
    with col_loss:
        st.markdown('<p class="section-header">📉 Training Curve</p>', unsafe_allow_html=True)
        st.pyplot(plot_loss_curve(loss_curve))

    with col_cm:
        st.markdown('<p class="section-header">🗂 Confusion Matrix</p>', unsafe_allow_html=True)
        st.pyplot(plot_confusion_matrix(cm, class_names))

    # ── Classification Report ──
    st.markdown('<p class="section-header">📋 Full Classification Report</p>', unsafe_allow_html=True)
    st.code(report, language="text")

    # ── Test sample preview ──
    st.markdown('<p class="section-header">🔬 Test Sample Predictions (first 20)</p>',
                unsafe_allow_html=True)
    results_df = pd.DataFrame({
        "True Label": [class_names[i] for i in y_test[:20]],
        "Predicted":  [class_names[i] for i in y_pred[:20]],
        "Correct?":   ["✅" if t == p else "❌" for t, p in zip(y_test[:20], y_pred[:20])],
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)

else:
    st.markdown("""
    <div class="info-box">
        👈 &nbsp; Configure your dataset and parameters in the sidebar, then click <b>▶ Train Model</b>.
    </div>
    """, unsafe_allow_html=True)
