"""
Bayessche Optimierung — Interaktive Streamlit App
==================================================
Optimierung Verfahren | Simon Gwangwaa

Starte mit:  streamlit run bayesian_optimization_app.py
Install:     pip install streamlit scikit-learn plotly numpy scipy
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Bayessche Optimierung",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0F172A; }
    [data-testid="stSidebar"] { background-color: #1E293B; }
    h1, h2, h3 { color: #F8FAFC !important; }
    .stMarkdown p { color: #CBD5E1; }
    .metric-card {
        background: #1E293B;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border-left: 4px solid;
    }
    .metric-value { font-size: 2.2em; font-weight: 800; }
    .metric-label { font-size: 0.85em; color: #94A3B8; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── Farben ───────────────────────────────────────────────────────
COL = {
    'primary': '#3B82F6', 'secondary': '#10B981', 'accent': '#F59E0B',
    'danger': '#EF4444', 'bg': '#0F172A', 'panel': '#1E293B',
    'text': '#F8FAFC', 'muted': '#94A3B8',
    'conf_light': 'rgba(59,130,246,0.1)', 'conf_dark': 'rgba(59,130,246,0.25)',
}

# ─── Zielfunktionen ──────────────────────────────────────────────
def branin_1d(x):
    return -(np.sin(3*x) * (x - 0.5)**2 + 0.5*np.sin(7*x) + 0.3*np.sin(13*x))

def multimodal(x):
    return -(np.sin(5*x) * np.cos(3*x) + 0.5*np.sin(11*x) - 0.3*x)

def simple_quadratic(x):
    return (x - 0.3)**2 - 0.5*np.sin(8*x)

def rastrigin_1d(x):
    A = 1.5
    return A + x**2 - A * np.cos(2 * np.pi * x * 2)

def steep_valleys(x):
    return np.sin(10*x) * np.exp(-2*(x-0.5)**2) + 0.5*(x-0.7)**2

FUNCTIONS = {
    'Branin (modifiziert)': branin_1d,
    'Multimodal': multimodal,
    'Quadratisch + Sinus': simple_quadratic,
    'Rastrigin': rastrigin_1d,
    'Steile Täler': steep_valleys,
}

# ─── Acquisition Functions ───────────────────────────────────────
def expected_improvement(X, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X.reshape(-1, 1), return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    Z = (y_best - mu - xi) / sigma
    return (y_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)

def upper_confidence_bound(X, gp, kappa=2.0):
    mu, sigma = gp.predict(X.reshape(-1, 1), return_std=True)
    return -(mu - kappa * sigma)

def probability_of_improvement(X, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X.reshape(-1, 1), return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    return norm.cdf((y_best - mu - xi) / sigma)

ACQ_FUNCTIONS = {
    'Expected Improvement (EI)': 'ei',
    'Upper Confidence Bound (UCB)': 'ucb',
    'Probability of Improvement (PI)': 'pi',
}

# ─── Plotly Layout ────────────────────────────────────────────────
def base_layout(title="", height=400):
    return dict(
        template="plotly_dark",
        paper_bgcolor=COL['bg'],
        plot_bgcolor=COL['bg'],
        title=dict(text=title, font=dict(size=16, color=COL['text'])),
        font=dict(color=COL['muted'], size=12),
        height=height,
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis=dict(gridcolor='#334155', zerolinecolor='#334155'),
        yaxis=dict(gridcolor='#334155', zerolinecolor='#334155'),
    )


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎯 Bayessche Optimierung")
    st.markdown("---")

    page = st.radio("**Modus wählen**", [
        "🔍 Schritt-für-Schritt",
        "⚔️ Vergleich",
        "🧠 Hyperparameter-Tuning",
        "📖 Theorie",
    ], index=0)

    st.markdown("---")
    st.markdown("##### Optimierung Verfahren")
    st.markdown("Simon Gwangwaa")


# ═══════════════════════════════════════════════════════════════════
#  PAGE 1: Schritt-für-Schritt
# ═══════════════════════════════════════════════════════════════════
if page == "🔍 Schritt-für-Schritt":
    st.markdown("# Bayessche Optimierung — Schritt für Schritt")
    st.markdown("*Beobachte wie der Algorithmus die unbekannte Funktion Schritt für Schritt lernt*")

    # Controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        func_name = st.selectbox("**Zielfunktion**", list(FUNCTIONS.keys()))
    with col2:
        acq_name = st.selectbox("**Acquisition Function**", list(ACQ_FUNCTIONS.keys()))
    with col3:
        kernel_name = st.selectbox("**Kernel**", ['Matérn (ν=2.5)', 'Matérn (ν=1.5)', 'RBF'])
    with col4:
        n_steps = st.slider("**Anzahl Schritte**", 0, 30, 5, 1)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        length_scale = st.slider("**Kernel Length Scale**", 0.05, 1.0, 0.2, 0.05,
                                  help="Wie glatt schätzt der GP die Funktion? Klein = flexibel, groß = glatt")
    with col_b:
        if ACQ_FUNCTIONS[acq_name] == 'ei':
            xi = st.slider("**ξ (Exploration)**", 0.0, 0.5, 0.01, 0.01,
                            help="Höher = mehr Exploration, niedriger = mehr Exploitation")
        elif ACQ_FUNCTIONS[acq_name] == 'ucb':
            kappa = st.slider("**κ (Exploration)**", 0.1, 5.0, 2.0, 0.1,
                               help="Höher = mehr Exploration")
        else:
            xi = st.slider("**ξ (Exploration)**", 0.0, 0.5, 0.01, 0.01)
    with col_c:
        noise = st.slider("**Rauschen (α)**", 1e-6, 0.1, 1e-4, format="%.5f",
                           help="Messunsicherheit in den Beobachtungen")

    show_true = st.checkbox("Wahre Funktion anzeigen", value=True,
                             help="In der Praxis ist diese unbekannt!")

    # ── Berechnung ──
    func = FUNCTIONS[func_name]
    X_plot = np.linspace(0, 1, 500)
    y_true = func(X_plot)

    # Kernel
    if kernel_name == 'Matérn (ν=2.5)':
        kernel = Matern(nu=2.5, length_scale=length_scale) + ConstantKernel(1.0)
    elif kernel_name == 'Matérn (ν=1.5)':
        kernel = Matern(nu=1.5, length_scale=length_scale) + ConstantKernel(1.0)
    else:
        kernel = RBF(length_scale=length_scale) + ConstantKernel(1.0)

    # Initial samples
    X_s = np.array([0.1, 0.9])
    y_s = func(X_s)

    gp = GaussianProcessRegressor(kernel=kernel, alpha=noise,
                                   n_restarts_optimizer=5, normalize_y=True)
    gp.fit(X_s.reshape(-1, 1), y_s)

    # BO Steps
    acq_type = ACQ_FUNCTIONS[acq_name]
    for step in range(n_steps):
        if acq_type == 'ei':
            acq_vals = expected_improvement(X_plot, gp, y_s.min(), xi=xi)
        elif acq_type == 'ucb':
            acq_vals = upper_confidence_bound(X_plot, gp, kappa=kappa)
        else:
            acq_vals = probability_of_improvement(X_plot, gp, y_s.min(), xi=xi)

        x_next = X_plot[np.argmax(acq_vals)]
        y_next = func(x_next)
        X_s = np.append(X_s, x_next)
        y_s = np.append(y_s, y_next)
        gp.fit(X_s.reshape(-1, 1), y_s)

    # Final prediction + acquisition
    mu, sigma = gp.predict(X_plot.reshape(-1, 1), return_std=True)
    if acq_type == 'ei':
        acq_vals = expected_improvement(X_plot, gp, y_s.min(), xi=xi)
    elif acq_type == 'ucb':
        acq_vals = upper_confidence_bound(X_plot, gp, kappa=kappa)
    else:
        acq_vals = probability_of_improvement(X_plot, gp, y_s.min(), xi=xi)

    # ── Metrics ──
    best_idx = np.argmin(y_s)
    true_min_idx = np.argmin(y_true)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card" style="border-color:{COL['primary']}">
            <div class="metric-value" style="color:{COL['primary']}">{len(X_s)}</div>
            <div class="metric-label">Samples</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card" style="border-color:{COL['secondary']}">
            <div class="metric-value" style="color:{COL['secondary']}">{y_s[best_idx]:.4f}</div>
            <div class="metric-label">Bestes f(x)</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card" style="border-color:{COL['accent']}">
            <div class="metric-value" style="color:{COL['accent']}">{X_s[best_idx]:.4f}</div>
            <div class="metric-label">Bestes x</div></div>""", unsafe_allow_html=True)
    with m4:
        error = abs(X_s[best_idx] - X_plot[true_min_idx])
        st.markdown(f"""<div class="metric-card" style="border-color:{COL['danger']}">
            <div class="metric-value" style="color:{COL['danger']}">{error:.4f}</div>
            <div class="metric-label">Fehler zum wahren Min.</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── GP Plot ──
    fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35],
                        vertical_spacing=0.08,
                        subplot_titles=["Gaussian Process Schätzung",
                                        f"Acquisition Function: {acq_name}"])

    # Confidence bands
    fig.add_trace(go.Scatter(x=X_plot, y=mu+2*sigma, mode='lines',
                             line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=X_plot, y=mu-2*sigma, mode='lines',
                             line=dict(width=0), fill='tonexty',
                             fillcolor=COL['conf_light'],
                             name='95% Konfidenz'), row=1, col=1)
    fig.add_trace(go.Scatter(x=X_plot, y=mu+sigma, mode='lines',
                             line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=X_plot, y=mu-sigma, mode='lines',
                             line=dict(width=0), fill='tonexty',
                             fillcolor=COL['conf_dark'],
                             name='68% Konfidenz'), row=1, col=1)

    # True function
    if show_true:
        fig.add_trace(go.Scatter(x=X_plot, y=y_true, mode='lines',
                                 line=dict(color=COL['danger'], width=1.5, dash='dash'),
                                 name='Wahre Funktion', opacity=0.5), row=1, col=1)

    # GP Mean
    fig.add_trace(go.Scatter(x=X_plot, y=mu, mode='lines',
                             line=dict(color=COL['primary'], width=3),
                             name='GP Schätzung (μ)'), row=1, col=1)

    # Samples
    fig.add_trace(go.Scatter(x=X_s[:2], y=y_s[:2], mode='markers',
                             marker=dict(color=COL['secondary'], size=10,
                                         line=dict(color='white', width=2)),
                             name='Initiale Samples'), row=1, col=1)
    if len(X_s) > 2:
        fig.add_trace(go.Scatter(x=X_s[2:], y=y_s[2:], mode='markers',
                                 marker=dict(color=COL['accent'], size=8, symbol='diamond',
                                             line=dict(color='white', width=1.5)),
                                 name='BO Samples'), row=1, col=1)

    # Best point
    fig.add_trace(go.Scatter(x=[X_s[best_idx]], y=[y_s[best_idx]], mode='markers',
                             marker=dict(color=COL['accent'], size=16, symbol='star',
                                         line=dict(color='white', width=2)),
                             name='Bester Punkt'), row=1, col=1)

    # Acquisition Function
    fig.add_trace(go.Scatter(x=X_plot, y=acq_vals, mode='lines',
                             line=dict(color=COL['accent'], width=2),
                             fill='tozeroy', fillcolor='rgba(245,158,11,0.15)',
                             name='Acquisition', showlegend=False), row=2, col=1)

    x_next_prop = X_plot[np.argmax(acq_vals)]
    fig.add_trace(go.Scatter(x=[x_next_prop], y=[acq_vals.max()], mode='markers',
                             marker=dict(color=COL['accent'], size=14, symbol='triangle-down',
                                         line=dict(color='white', width=2)),
                             name='Nächster Vorschlag', showlegend=False), row=2, col=1)
    fig.add_vline(x=x_next_prop, line_dash="dash", line_color=COL['accent'],
                  opacity=0.5, row=2, col=1)

    layout = base_layout(height=650)
    layout['legend'] = dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='center', x=0.5, font=dict(size=11))
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="x", row=2, col=1, gridcolor='#334155')
    fig.update_yaxes(title_text="f(x)", row=1, col=1, gridcolor='#334155')
    fig.update_yaxes(title_text="α(x)", row=2, col=1, gridcolor='#334155')
    fig.update_annotations(font=dict(color=COL['text'], size=14))

    st.plotly_chart(fig, use_container_width=True)

    # Explanation
    with st.expander("💡 Was passiert hier?"):
        st.markdown(f"""
        **Schritt {n_steps}** der Bayesschen Optimierung:

        1. Der **Gaussian Process** (blaue Linie) schätzt die unbekannte Funktion basierend auf {len(X_s)} Beobachtungen
        2. Das **Konfidenzband** (blauer Bereich) zeigt die Unsicherheit — breit wo wenig Daten, eng wo viele
        3. Die **Acquisition Function** (goldene Kurve unten) bewertet, wo die nächste Messung am meisten bringt
        4. Das **▼ Dreieck** zeigt den vorgeschlagenen nächsten Messpunkt

        **Probiere aus:**
        - Schiebe den **Length Scale** Slider: kleiner = der GP passt sich stärker an, größer = glattere Schätzung
        - Ändere **ξ/κ**: mehr Exploration (unbekannte Regionen erkunden) vs Exploitation (bei gutem Wert bleiben)
        - Vergleiche verschiedene **Acquisition Functions** und **Kernel**
        """)


# ═══════════════════════════════════════════════════════════════════
#  PAGE 2: Vergleich
# ═══════════════════════════════════════════════════════════════════
elif page == "⚔️ Vergleich":
    st.markdown("# BO vs Random vs Grid Search")
    st.markdown("*Wer findet das Minimum mit dem geringsten Budget?*")

    col1, col2, col3 = st.columns(3)
    with col1:
        func_name = st.selectbox("**Zielfunktion**", list(FUNCTIONS.keys()), index=1)
    with col2:
        budget = st.slider("**Evaluierungsbudget**", 5, 40, 20)
    with col3:
        seed = st.slider("**Random Seed**", 0, 100, 42,
                          help="Ändere den Seed um verschiedene Random-Ergebnisse zu sehen")

    func = FUNCTIONS[func_name]
    X_plot = np.linspace(0, 1, 500)
    y_true = func(X_plot)
    true_min = np.min(y_true)
    np.random.seed(seed)

    # ── Bayessche Optimierung ──
    bo_X = np.array([0.1, 0.9])
    bo_y = func(bo_X)
    bo_best = [np.min(bo_y)]
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale=0.2) + ConstantKernel(1.0),
        alpha=1e-6, n_restarts_optimizer=5, normalize_y=True)
    gp.fit(bo_X.reshape(-1, 1), bo_y)

    for _ in range(budget):
        ei = expected_improvement(X_plot, gp, bo_y.min())
        x_n = X_plot[np.argmax(ei)]
        bo_X = np.append(bo_X, x_n)
        bo_y = np.append(bo_y, func(x_n))
        gp.fit(bo_X.reshape(-1, 1), bo_y)
        bo_best.append(np.min(bo_y))

    # ── Random Search ──
    rand_X = np.random.uniform(0, 1, budget + 2)
    rand_y = func(rand_X)
    rand_best = [np.min(rand_y[:i+1]) for i in range(len(rand_y))]

    # ── Grid Search ──
    grid_X = np.linspace(0, 1, budget + 2)
    grid_y = func(grid_X)
    grid_best = [np.min(grid_y[:i+1]) for i in range(len(grid_y))]

    n = min(len(bo_best), len(rand_best), len(grid_best))
    bo_best, rand_best, grid_best = bo_best[:n], rand_best[:n], grid_best[:n]

    # ── Metrics ──
    m1, m2, m3 = st.columns(3)
    for col_m, (label, best, color) in zip(
        [m1, m2, m3],
        [("Bayessche Opt.", bo_best[-1], COL['primary']),
         ("Random Search", rand_best[-1], COL['secondary']),
         ("Grid Search", grid_best[-1], COL['accent'])]):
        with col_m:
            gap = abs(best - true_min)
            st.markdown(f"""<div class="metric-card" style="border-color:{color}">
                <div class="metric-value" style="color:{color}">{best:.4f}</div>
                <div class="metric-label">{label}<br>Abstand zum Optimum: {gap:.4f}</div></div>""",
                unsafe_allow_html=True)

    st.markdown("")

    # ── Sample Distribution Plots ──
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Bayessche Optimierung", "Random Search",
                                        "Grid Search", "Konvergenzverlauf"],
                        vertical_spacing=0.12, horizontal_spacing=0.08)

    data_sets = [(bo_X, bo_y, COL['primary']),
                 (rand_X, rand_y, COL['secondary']),
                 (grid_X, grid_y, COL['accent'])]
    positions = [(1,1), (1,2), (2,1)]

    for (Xs, ys, col), (r, c) in zip(data_sets, positions):
        fig.add_trace(go.Scatter(x=X_plot, y=y_true, mode='lines',
                                 line=dict(color=COL['danger'], width=1, dash='dash'),
                                 opacity=0.4, showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=Xs, y=ys, mode='markers',
                                 marker=dict(color=col, size=7,
                                             line=dict(color='white', width=1)),
                                 showlegend=False), row=r, col=c)
        bi = np.argmin(ys)
        fig.add_trace(go.Scatter(x=[Xs[bi]], y=[ys[bi]], mode='markers',
                                 marker=dict(color=col, size=16, symbol='star',
                                             line=dict(color='white', width=2)),
                                 showlegend=False), row=r, col=c)

    # Convergence
    steps = list(range(n))
    for vals, name, col, dash in [
        (bo_best, 'Bayessche Opt.', COL['primary'], 'solid'),
        (rand_best, 'Random', COL['secondary'], 'dash'),
        (grid_best, 'Grid', COL['accent'], 'dot')]:
        fig.add_trace(go.Scatter(x=steps, y=vals, mode='lines',
                                 line=dict(color=col, width=2.5, dash=dash),
                                 name=name), row=2, col=2)
    fig.add_hline(y=true_min, line_dash="dash", line_color=COL['danger'],
                  opacity=0.4, row=2, col=2)

    layout = base_layout(height=650)
    layout['legend'] = dict(orientation='h', yanchor='bottom', y=-0.05,
                            xanchor='center', x=0.75, font=dict(size=11))
    fig.update_layout(**layout)
    fig.update_annotations(font=dict(color=COL['text'], size=13))
    for i in range(1, 5):
        fig.update_xaxes(gridcolor='#334155', row=(i-1)//2+1, col=(i-1)%2+1)
        fig.update_yaxes(gridcolor='#334155', row=(i-1)//2+1, col=(i-1)%2+1)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("💡 Was sehen wir?"):
        st.markdown(f"""
        - **Bayessche Optimierung** konzentriert Samples in der Nähe des Optimums — sie *lernt* wo es sich lohnt zu suchen
        - **Random Search** verteilt Punkte gleichmäßig zufällig — findet das Minimum manchmal durch Glück
        - **Grid Search** verteilt Punkte auf einem Gitter — kann enge Optima verpassen

        Ändere den **Seed** um zu sehen: Random Search ist unzuverlässig, BO konvergiert fast immer.
        """)


# ═══════════════════════════════════════════════════════════════════
#  PAGE 3: Hyperparameter-Tuning
# ═══════════════════════════════════════════════════════════════════
elif page == "🧠 Hyperparameter-Tuning":
    st.markdown("# Hyperparameter-Tuning mit BO")
    st.markdown("*Optimierung von Learning Rate & Regularisierung eines Neural Networks*")

    col1, col2 = st.columns(2)
    with col1:
        n_bo_steps = st.slider("**BO Schritte**", 3, 25, 12)
    with col2:
        n_init = st.slider("**Initiale Samples**", 2, 8, 3)

    with st.spinner("Trainiere Neural Networks... (kann einige Sekunden dauern)"):
        from sklearn.datasets import make_moons
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import cross_val_score

        X_data, y_data = make_moons(n_samples=200, noise=0.25, random_state=42)

        @st.cache_data
        def compute_landscape():
            lr_range = np.linspace(-4, -1, 15)
            alpha_range = np.linspace(-5, -1, 15)
            Z = np.zeros((len(alpha_range), len(lr_range)))
            for i, a in enumerate(alpha_range):
                for j, l in enumerate(lr_range):
                    lr = 10**l; alpha = 10**a
                    clf = MLPClassifier(hidden_layer_sizes=(32, 16),
                                        learning_rate_init=lr, alpha=alpha,
                                        max_iter=300, random_state=42)
                    scores = cross_val_score(clf, X_data, y_data, cv=3, scoring='accuracy')
                    Z[i, j] = scores.mean()
            return lr_range, alpha_range, Z

        lr_range, alpha_range, Z_acc = compute_landscape()

        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator((alpha_range, lr_range), -Z_acc, method='linear')

        def objective_2d(lr_log, alpha_log):
            lr_log = np.clip(lr_log, -4, -1)
            alpha_log = np.clip(alpha_log, -5, -1)
            return float(interp([[alpha_log, lr_log]])[0])

        # Run BO
        np.random.seed(42)
        bo_lr = np.random.uniform(-4, -1, n_init)
        bo_alpha = np.random.uniform(-5, -1, n_init)
        bo_scores = np.array([objective_2d(l, a) for l, a in zip(bo_lr, bo_alpha)])

        gp_2d = GaussianProcessRegressor(
            kernel=Matern(nu=2.5) + ConstantKernel(1.0),
            alpha=1e-4, n_restarts_optimizer=5, normalize_y=True)

        lr_grid = np.linspace(-4, -1, 40)
        alpha_grid = np.linspace(-5, -1, 40)
        LR_G, ALPHA_G = np.meshgrid(lr_grid, alpha_grid)
        X_grid_flat = np.column_stack([LR_G.ravel(), ALPHA_G.ravel()])

        for step in range(n_bo_steps):
            X_bo = np.column_stack([bo_lr, bo_alpha])
            gp_2d.fit(X_bo, bo_scores)
            mu_g, sigma_g = gp_2d.predict(X_grid_flat, return_std=True)
            sigma_g = np.maximum(sigma_g, 1e-9)
            y_best = bo_scores.min()
            Zv = (y_best - mu_g) / sigma_g
            ei_g = (y_best - mu_g) * norm.cdf(Zv) + sigma_g * norm.pdf(Zv)
            best_i = np.argmax(ei_g)
            nl, na = X_grid_flat[best_i]
            bo_lr = np.append(bo_lr, nl)
            bo_alpha = np.append(bo_alpha, na)
            bo_scores = np.append(bo_scores, objective_2d(nl, na))

    # Metrics
    best_overall = np.argmin(bo_scores)
    best_acc = -bo_scores[best_overall]
    best_lr = 10**bo_lr[best_overall]
    best_alpha_val = 10**bo_alpha[best_overall]

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="metric-card" style="border-color:{COL['primary']}">
            <div class="metric-value" style="color:{COL['primary']}">{best_acc:.1%}</div>
            <div class="metric-label">Beste Accuracy</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card" style="border-color:{COL['secondary']}">
            <div class="metric-value" style="color:{COL['secondary']}">{best_lr:.5f}</div>
            <div class="metric-label">Optimale Learning Rate</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card" style="border-color:{COL['accent']}">
            <div class="metric-value" style="color:{COL['accent']}">{best_alpha_val:.5f}</div>
            <div class="metric-label">Optimale Regularisierung</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    # Plots
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Accuracy-Landschaft", "BO Suchpfad", "Konvergenz"],
                        horizontal_spacing=0.08)

    # Landscape
    fig.add_trace(go.Contour(x=lr_range, y=alpha_range, z=Z_acc,
                              colorscale='Viridis', showscale=False,
                              contours=dict(showlabels=False),
                              name='Accuracy'), row=1, col=1)

    # Search path
    fig.add_trace(go.Contour(x=lr_range, y=alpha_range, z=Z_acc,
                              colorscale='Viridis', showscale=False, opacity=0.4,
                              contours=dict(showlabels=False),
                              name=''), row=1, col=2)
    fig.add_trace(go.Scatter(x=bo_lr, y=bo_alpha, mode='lines+markers',
                             line=dict(color=COL['muted'], width=1),
                             marker=dict(color=list(range(len(bo_lr))),
                                         colorscale='Plasma', size=8,
                                         line=dict(color='white', width=1),
                                         showscale=True,
                                         colorbar=dict(title='Schritt', x=0.68)),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[bo_lr[best_overall]], y=[bo_alpha[best_overall]],
                             mode='markers',
                             marker=dict(color=COL['accent'], size=18, symbol='star',
                                         line=dict(color='white', width=2)),
                             showlegend=False), row=1, col=2)

    # Convergence
    best_so_far = [-np.min(bo_scores[:i+1]) for i in range(len(bo_scores))]
    fig.add_trace(go.Scatter(x=list(range(len(best_so_far))), y=best_so_far,
                             mode='lines+markers',
                             line=dict(color=COL['primary'], width=2.5),
                             marker=dict(size=5, color=COL['primary']),
                             showlegend=False), row=1, col=3)

    layout = base_layout(height=450)
    fig.update_layout(**layout)
    fig.update_annotations(font=dict(color=COL['text'], size=13))
    fig.update_xaxes(title_text="log₁₀(LR)", row=1, col=1, gridcolor='#334155')
    fig.update_xaxes(title_text="log₁₀(LR)", row=1, col=2, gridcolor='#334155')
    fig.update_xaxes(title_text="Evaluierungen", row=1, col=3, gridcolor='#334155')
    fig.update_yaxes(title_text="log₁₀(α)", row=1, col=1, gridcolor='#334155')
    fig.update_yaxes(title_text="log₁₀(α)", row=1, col=2, gridcolor='#334155')
    fig.update_yaxes(title_text="Accuracy", row=1, col=3, gridcolor='#334155')

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("💡 Was passiert hier?"):
        st.markdown("""
        Wir optimieren zwei Hyperparameter eines MLPClassifier (Neural Network) auf dem Make-Moons Datensatz:

        - **Learning Rate**: Wie schnell lernt das Netzwerk? (log-Skala: 10⁻⁴ bis 10⁻¹)
        - **Regularisierung α**: Wie stark wird Overfitting bestraft? (log-Skala: 10⁻⁵ bis 10⁻¹)

        Die BO findet die optimale Kombination mit nur wenigen Trainingsläufen — statt das gesamte Grid durchzuprobieren.
        Der **Suchpfad** zeigt, wie die BO sich sukzessive zur optimalen Region bewegt.
        """)


# ═══════════════════════════════════════════════════════════════════
#  PAGE 4: Theorie
# ═══════════════════════════════════════════════════════════════════
elif page == "📖 Theorie":
    st.markdown("# Theorie der Bayesschen Optimierung")

    st.markdown("---")
    st.markdown("## Das Problem")
    st.markdown("""
    Wir suchen das Minimum einer **teuren Black-Box-Funktion** f(x):

    > **min** f(x), wobei jede Auswertung von f(x) kostspielig ist (Zeit, Geld, Ressourcen)

    Wir haben keinen Zugang zum Gradienten, nur zur Eingabe-Ausgabe-Beziehung.
    """)

    st.markdown("---")
    st.markdown("## Gaussian Process (GP)")
    st.markdown("""
    Der GP ist ein **probabilistisches Surrogatmodell**, das die unbekannte Funktion approximiert.

    Für jeden Punkt x liefert er eine **Normalverteilung**:
    - **μ(x)** — die Schätzung (Mittelwert)
    - **σ(x)** — die Unsicherheit (Standardabweichung)

    Der **Kernel** bestimmt, wie ähnlich sich nahe Punkte verhalten. Wichtige Kernel:
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card" style="border-color:{COL['primary']}">
            <div style="color:{COL['primary']}; font-weight:700; font-size:1.1em">Matérn (ν=2.5)</div>
            <div class="metric-label">Zweimal differenzierbar.<br>Standard in der Praxis.</div></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card" style="border-color:{COL['secondary']}">
            <div style="color:{COL['secondary']}; font-weight:700; font-size:1.1em">Matérn (ν=1.5)</div>
            <div class="metric-label">Einmal differenzierbar.<br>Für rauere Funktionen.</div></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card" style="border-color:{COL['accent']}">
            <div style="color:{COL['accent']}; font-weight:700; font-size:1.1em">RBF (Squared Exp.)</div>
            <div class="metric-label">Unendlich glatt.<br>Oft zu optimistisch.</div></div>""",
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Acquisition Functions")
    st.markdown("""
    Die Acquisition Function entscheidet, **wo als nächstes gemessen wird**. Sie balanciert:
    - **Exploitation**: Nahe beim bisherigen Minimum suchen
    - **Exploration**: Unsichere Regionen erkunden
    """)

    st.markdown("""
    | Acquisition Function | Formel | Eigenschaft |
    |---|---|---|
    | **Expected Improvement (EI)** | E[max(f* - f(x), 0)] | Beliebteste Wahl, gute Balance |
    | **Upper Confidence Bound (UCB)** | μ(x) - κ·σ(x) | Direkter Trade-off über κ |
    | **Probability of Improvement (PI)** | P(f(x) < f*) | Einfach, aber zu gierig |
    """)

    st.markdown("---")
    st.markdown("## Der Algorithmus")
    st.code("""
    Gegeben: Black-Box-Funktion f, Budget T, Kernel k

    1. Initialisiere mit n₀ zufälligen Samples → D₀ = {(xᵢ, yᵢ)}
    2. Für t = 1, 2, ..., T:
       a) Fitte GP auf Dₜ₋₁
       b) Berechne Acquisition Function α(x)
       c) xₜ = argmax α(x)        ← Nächster Messpunkt
       d) yₜ = f(xₜ) + ε          ← Teure Evaluation!
       e) Dₜ = Dₜ₋₁ ∪ {(xₜ, yₜ)}  ← Update Datensatz
    3. Gib x* = argmin yᵢ zurück
    """, language="text")

    st.markdown("---")
    st.markdown("## Anwendungen")
    st.markdown("""
    | Bereich | Beispiel | Warum BO? |
    |---|---|---|
    | **Machine Learning** | Hyperparameter-Tuning | Jedes Training dauert Stunden |
    | **Materialforschung** | Optimale Legierung | Jedes Experiment ist teuer |
    | **Pharma** | Wirkstoff-Dosierung | Klinische Tests sind limitiert |
    | **Robotik** | Steuerungsparameter | Physische Tests kosten Zeit |
    | **Industrie** | Fertigungsprozesse | Produktionsstopps vermeiden |
    """)
