"""
Standard LSTM Streamflow Prediction + DACP Uncertainty Quantification
5 Snowmelt-Driven Watersheds | Walk-Forward Training
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os, json
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

DEVICE = 'cpu'
SEQ_LEN = 14
HIDDEN = 128
N_LAYERS = 2
DROPOUT = 0.2
LR = 1e-3
MAX_EPOCHS = 200
PATIENCE = 25
BATCH = 32

# ─── Site configs ───────────────────────────────────────────────────────────
SITES = {
    'Bunchgrass_Meadow': {
        'file': '/mnt/user-data/uploads/Bunchgrass_Meadow_Inflow_Data_FILLED__1_.csv',
        'log_transform': True,
        'max_cfs': 2922
    },
    'Easy_Pass': {
        'file': '/mnt/user-data/uploads/Easy_Pass_Inflow_Data_FILLED__1_.csv',
        'log_transform': False,
        'max_cfs': 277
    },
    'Paradise': {
        'file': '/mnt/user-data/uploads/Paradise_Inflow_Data_FILLED__1_.csv',
        'log_transform': True,
        'max_cfs': 79
    },
    'Swift_Creek': {
        'file': '/mnt/user-data/uploads/Swift_Creek_Inflow_Data_FILLED__1_.csv',
        'log_transform': False,
        'max_cfs': 171
    },
    'Touchet': {
        'file': '/mnt/user-data/uploads/Touchet_Inflow_Data_FILLED__1_.csv',
        'log_transform': False,
        'max_cfs': 672
    }
}

FEATURE_COLS = ['WESD', 'GHI', 'PRCP', 'SNWD', 'TAVG']


# ─── Feature engineering ────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df['prcp_7d'] = df['PRCP'].rolling(7, min_periods=1).sum()
    df['API'] = df['PRCP'].ewm(alpha=0.08, min_periods=1).mean()  # k=0.92 decay
    doy = np.arange(len(df))
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365)
    return df

FEAT_COLS = ['WESD', 'GHI', 'PRCP', 'SNWD', 'TAVG', 'prcp_7d', 'doy_sin', 'doy_cos']


# ─── Dataset ────────────────────────────────────────────────────────────────
def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ─── Model ──────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# ─── Training ───────────────────────────────────────────────────────────────
def train_model(X_tr, y_tr, n_features):
    torch.manual_seed(42)
    model = LSTMModel(n_features).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.HuberLoss(delta=1.0)

    # Val split
    n_val = max(int(len(X_tr) * 0.1), SEQ_LEN + 1)
    X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]
    X_tr2, y_tr2 = X_tr[:-n_val], y_tr[:-n_val]

    best_val, best_state, wait = float('inf'), None, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        idx = np.random.permutation(len(X_tr2))
        for i in range(0, len(idx), BATCH):
            b = idx[i:i+BATCH]
            xb = torch.tensor(X_tr2[b]).to(DEVICE)
            yb = torch.tensor(y_tr2[b]).to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val).to(DEVICE)
            yv = torch.tensor(y_val).to(DEVICE)
            val_loss = criterion(model(xv), yv).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


# ─── Walk-forward prediction ─────────────────────────────────────────────────
def walk_forward(df, log_transform):
    df = engineer_features(df)
    target_col = 'DISCHRG'

    if log_transform:
        df['target'] = np.log1p(df[target_col].values)
    else:
        df['target'] = df[target_col].values

    X_all = df[FEAT_COLS].values
    y_all = df['target'].values
    y_true_cfs = df[target_col].values

    n = len(df)
    train_end = int(n * 0.85)

    preds_log = []
    preds_cfs = []
    trues_cfs = []

    # Single train/test split (walk-forward on test)
    scaler_x = StandardScaler()
    scaler_x.fit(X_all[:train_end])
    X_scaled = scaler_x.transform(X_all)

    scaler_y = StandardScaler()
    scaler_y.fit(y_all[:train_end].reshape(-1, 1))
    y_scaled = scaler_y.transform(y_all.reshape(-1, 1)).ravel()

    X_seq, y_seq = make_sequences(X_scaled, y_scaled, SEQ_LEN)
    y_true_seq = y_true_cfs[SEQ_LEN:]

    X_tr = X_seq[:train_end - SEQ_LEN]
    y_tr = y_seq[:train_end - SEQ_LEN]
    X_te = X_seq[train_end - SEQ_LEN:]
    y_te_cfs = y_true_seq[train_end - SEQ_LEN:]

    print(f"  Training on {len(X_tr)} samples, testing on {len(X_te)} samples")
    model = train_model(X_tr, y_tr, len(FEAT_COLS))

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(X_te).to(DEVICE)).numpy()

    pred_log = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    if log_transform:
        pred_cfs = np.expm1(pred_log)
    else:
        pred_cfs = pred_log

    pred_cfs = np.maximum(pred_cfs, 0)

    return pred_cfs, y_te_cfs, model, scaler_x, scaler_y, X_scaled, y_scaled, X_seq, y_true_seq, train_end


# ─── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(true, pred):
    r2 = r2_score(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    # MRE: mean |pred - true| / (|true| + eps)
    mre = np.mean(np.abs(pred - true) / (np.abs(true) + 1e-6))
    nse = 1 - np.sum((true - pred)**2) / np.sum((true - np.mean(true))**2)
    return {'R2': round(r2, 3), 'RMSE': round(rmse, 2), 'MRE': round(mre, 3), 'NSE': round(nse, 3)}


# ─── DACP ───────────────────────────────────────────────────────────────────
def dacp(true, pred, target_coverage=0.95):
    alpha_target = 1 - target_coverage
    n = 180         # rolling window
    t_m = 0.35      # initial partition
    t_min, t_max = 0.10, 0.60
    mu1, mu2 = 0.03, 0.50
    lam = 0.01
    alpha_m = alpha_target
    gamma = 0.1

    N = len(true)
    lowers, uppers = np.zeros(N), np.zeros(N)
    covered = []

    for i in range(N):
        # Build error history
        if i < n:
            errors = np.abs(true[:i] - pred[:i]) if i > 0 else np.array([0.0])
        else:
            errors = np.abs(true[i-n:i] - pred[i-n:i])

        n_cal = max(int((1 - t_m) * len(errors)), 1)
        cal_errors = np.sort(errors)[-n_cal:][::-1]
        # Actually take the right portion:
        cal_errors = errors[-n_cal:]

        q_level = min(1 - alpha_m, 1.0)
        q = np.quantile(cal_errors, q_level) if len(cal_errors) > 0 else np.mean(errors) if len(errors) > 0 else 0

        lowers[i] = max(pred[i] - q, 0)
        uppers[i] = pred[i] + q

        cov = (true[i] >= lowers[i]) and (true[i] <= uppers[i])
        covered.append(float(cov))

        # Update regulators
        e_m = 0.0 if cov else 1.0
        if cov:
            t_m = np.clip(t_m + gamma * mu1, t_min, t_max)
        else:
            t_m = np.clip(t_m + gamma * mu2, t_min, t_max)
        alpha_m = np.clip(alpha_m + lam * (alpha_target - e_m), 0.01, 0.5)

    coverage = np.mean(covered)
    avg_width = np.mean(uppers - lowers)

    # Interval score (Winkler)
    alpha = alpha_target
    widths = uppers - lowers
    miss_low = np.maximum(lowers - true, 0)
    miss_high = np.maximum(true - uppers, 0)
    IS = np.mean(widths + (2/alpha) * (miss_low + miss_high))

    return lowers, uppers, coverage, avg_width, IS


# ─── Main ───────────────────────────────────────────────────────────────────
os.makedirs('/home/claude/outputs', exist_ok=True)
os.makedirs('/home/claude/outputs/plots', exist_ok=True)

all_results = {}
targets = [0.70, 0.80, 0.90, 0.95]

COLORS = {
    'observed': '#1a3a5c',
    'predicted': '#2e8b57',
    'interval': '#f0a500',
}

for site_name, cfg in SITES.items():
    print(f"\n{'='*60}")
    print(f"Processing: {site_name}")
    df = pd.read_csv(cfg['file'])
    log_t = cfg['log_transform']

    pred_cfs, true_cfs, model, sx, sy, X_scaled, y_scaled, X_seq, y_true_seq, train_end = \
        walk_forward(df, log_t)

    metrics = compute_metrics(true_cfs, pred_cfs)
    print(f"  R²={metrics['R2']}, RMSE={metrics['RMSE']}, MRE={metrics['MRE']}, NSE={metrics['NSE']}")

    site_results = {'metrics': metrics, 'coverage': {}}

    # Run DACP for each target
    for tc in targets:
        lo, hi, cov, aw, IS = dacp(true_cfs, pred_cfs, target_coverage=tc)
        site_results['coverage'][str(tc)] = {
            'coverage': round(cov * 100, 1),
            'avg_width': round(aw, 2),
            'IS': round(IS, 2)
        }
        print(f"  Target={tc:.0%}: Coverage={cov:.1%}, Avg Width={aw:.2f} cfs, IS={IS:.2f}")

    # Plot for 95% target
    lo95, hi95, cov95, aw95, _ = dacp(true_cfs, pred_cfs, target_coverage=0.95)
    n_plot = len(true_cfs)
    x_axis = np.arange(n_plot)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(x_axis, lo95, hi95, alpha=0.35, color='#f0a500', label='DACP 95% Interval')
    ax.plot(x_axis, true_cfs, color='#1a3a5c', lw=1.2, label='Observed', alpha=0.85)
    ax.plot(x_axis, pred_cfs, color='#2e8b57', lw=1.0, label='LSTM Prediction', alpha=0.9)
    ax.set_title(f'LSTM + DACP: {site_name.replace("_"," ")} | Target: 95%\n'
                 f'Coverage = {cov95:.1%} | Avg Width = {aw95:.2f} cfs | R² = {metrics["R2"]}',
                 fontsize=11)
    ax.set_xlabel('Test Data Point Index')
    ax.set_ylabel('Discharge (cfs)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, n_plot)
    plt.tight_layout()
    plt.savefig(f'/home/claude/outputs/plots/{site_name}_95pct.png', dpi=150, bbox_inches='tight')
    plt.close()

    all_results[site_name] = site_results
    print(f"  Plot saved.")

# Save results JSON
with open('/home/claude/outputs/results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n\nAll sites complete! Results saved.")
print(json.dumps(all_results, indent=2))
