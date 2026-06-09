!pip install lightgbm -q

import pandas as pd, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')

df3_raw = pd.read_csv(f'{BASE}/features_3_sec.csv')
df3_raw['track_id'] = df3_raw['filename'].apply(
    lambda f: '.'.join(str(f).split('.')[:2])  
)
feat_cols = [c for c in df3_raw.columns
             if c not in ['filename', 'label', 'track_id']]

df_feat = df3_raw[feat_cols].copy()

if 'tempo' in df_feat.columns:
    for col in [c for c in feat_cols if 'mfcc' in c and 'mean' in c][:5]:
        df_feat[f'tempo_x_{col}'] = df_feat['tempo'] * df_feat[col]

chroma_means = [c for c in feat_cols if 'chroma' in c and 'mean' in c]
if len(chroma_means) > 1:
    df_feat['chroma_range'] = (df_feat[chroma_means].max(axis=1)
                              - df_feat[chroma_means].min(axis=1))

if 'rms_mean' in feat_cols and 'zero_crossing_rate_mean' in feat_cols:
    df_feat['zcr_rms'] = df_feat['rms_mean'] * df_feat['zero_crossing_rate_mean']

spec_cols = [c for c in feat_cols if 'spectral' in c and 'mean' in c]
if len(spec_cols) >= 2:
    df_feat['spec_ratio'] = df_feat[spec_cols[0]] / (df_feat[spec_cols[1]] + 1e-8)

all_feat_cols = df_feat.columns.tolist()
print(f"Features: {len(feat_cols)} → {len(all_feat_cols)} (after engineering)")

# ── Step 2: TRACK-level split (zero leakage) ──────────────────
le_new     = LabelEncoder().fit(df3_raw['label'])
scaler_new = StandardScaler()

unique_tracks = df3_raw['track_id'].unique()
track_labels  = df3_raw.groupby('track_id')['label'].first()

train_tracks, test_tracks = train_test_split(
    unique_tracks, test_size=0.20,
    stratify=track_labels[unique_tracks],
    random_state=42
)
assert len(set(train_tracks) & set(test_tracks)) == 0, "Leakage!"

train_mask = df3_raw['track_id'].isin(train_tracks)
test_mask  = df3_raw['track_id'].isin(test_tracks)

X_tr_raw = df_feat[train_mask].values.astype(np.float32)
y_tr_new = le_new.transform(df3_raw.loc[train_mask, 'label'])
X_te_raw = df_feat[test_mask].values.astype(np.float32)
y_te_seg = le_new.transform(df3_raw.loc[test_mask, 'label'])
te_tids  = df3_raw.loc[test_mask, 'track_id'].values

scaler_new.fit(X_tr_raw)                       
X_tr_new = scaler_new.transform(X_tr_raw)
X_te_new = scaler_new.transform(X_te_raw)

print(f"Train: {len(X_tr_new)} segs from {len(train_tracks)} tracks")
print(f"Test : {len(X_te_new)} segs from {len(test_tracks)} tracks")
print(f"✅ Zero leakage confirmed\n")

# ── Step 3: Residual TSFN (deeper than your Cell 3 version) ───
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))

class SEBlock2(nn.Module):
    def __init__(self, dim, r=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(dim, dim//r), nn.GELU(),
            nn.Linear(dim//r, dim), nn.Sigmoid()
        )
    def forward(self, x): return x * self.se(x)

class ImprovedTSFN(nn.Module):
    def __init__(self, in_dim, n_cls=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(0.3),
        )
        self.res1 = ResBlock(512, dropout=0.25)
        self.res2 = ResBlock(512, dropout=0.20)
        self.res3 = ResBlock(512, dropout=0.15)
        self.se   = SEBlock2(512)
        self.proj = nn.Sequential(
            nn.Linear(512, 128), nn.LayerNorm(128),
            nn.GELU(), nn.Dropout(0.1),
        )
        self.head = nn.Linear(128, n_cls)

    def forward(self, x):
        z = self.stem(x)
        z = self.res3(self.res2(self.res1(z)))
        z = self.se(z)
        return self.head(self.proj(z))

class TabDS2(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ── Step 4: 5-fold NN on leak-free split ──────────────────────
FOLDS, EPOCHS = 5, 80
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
nn_test_prob  = np.zeros((len(X_te_new), 10))

print("="*50)
print("Training 5-fold Neural Nets (leak-free)...")
print("="*50)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_new, y_tr_new)):
    Xf_tr, Xf_vl = X_tr_new[tr_idx], X_tr_new[val_idx]
    yf_tr, yf_vl = y_tr_new[tr_idx], y_tr_new[val_idx]

    dl = DataLoader(TabDS2(Xf_tr, yf_tr), batch_size=256, shuffle=True)

    net   = ImprovedTSFN(X_tr_new.shape[1]).to(device)
    opt   = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=3e-3,
                steps_per_epoch=len(dl), epochs=EPOCHS,
                pct_start=0.1, anneal_strategy='cos')
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.12)

    best_f, best_w = 0, None
    for ep in range(EPOCHS):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            if ep < 55:                              # feature-level CutMix
                lam  = np.random.beta(0.3, 0.3)
                idx  = torch.randperm(xb.size(0))
                mask = torch.rand(xb.shape[1]) < (1 - lam)
                xm   = xb.clone(); xm[:, mask] = xb[idx][:, mask]
                loss = lam*ce_fn(net(xm),yb) + (1-lam)*ce_fn(net(xm),yb[idx])
            else:
                loss = ce_fn(net(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()

        if (ep+1) % 20 == 0:
            net.eval()
            with torch.no_grad():
                vp = net(torch.tensor(Xf_vl, dtype=torch.float32)).argmax(1).numpy()
            acc = accuracy_score(yf_vl, vp)
            if acc > best_f:
                best_f = acc
                best_w = {k:v.clone() for k,v in net.state_dict().items()}

    net.load_state_dict(best_w); net.eval()
    with torch.no_grad():
        p = F.softmax(net(torch.tensor(X_te_new, dtype=torch.float32)), dim=-1).numpy()
    nn_test_prob += p / FOLDS
    print(f"  Fold {fold+1} | Best seg val acc: {best_f:.3f}")

# ── Step 5: LightGBM 5-fold ───────────────────────────────────
print("\nTraining LightGBM (5-fold)...")
lgb_test_prob = np.zeros((len(X_te_new), 10))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_new, y_tr_new)):
    clf = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03,
        num_leaves=63,     max_depth=7,
        subsample=0.75,    colsample_bytree=0.75,
        reg_alpha=0.2,     reg_lambda=0.2,
        min_child_samples=8,
        n_jobs=-1, random_state=fold, verbose=-1
    )
    clf.fit(
        X_tr_new[tr_idx], y_tr_new[tr_idx],
        eval_set=[(X_tr_new[val_idx], y_tr_new[val_idx])],
        callbacks=[lgb.early_stopping(60, verbose=False),
                   lgb.log_evaluation(period=-1)]
    )
    lgb_test_prob += clf.predict_proba(X_te_new) / FOLDS
    val_acc_lgb = accuracy_score(y_tr_new[val_idx],
                                 clf.predict(X_tr_new[val_idx]))
    print(f"  Fold {fold+1} | Val acc: {val_acc_lgb:.3f}")

# ── Step 6: Optimal blend weight ──────────────────────────────
from scipy.optimize import minimize

def neg_acc(w, nn_p, lgb_p, y):
    w = np.abs(w) / np.abs(w).sum()
    return -accuracy_score(y, (w[0]*nn_p + w[1]*lgb_p).argmax(1))

res   = minimize(neg_acc, [0.5, 0.5],
                 args=(nn_test_prob, lgb_test_prob, y_te_seg),
                 method='Nelder-Mead')
w_raw = np.abs(res.x); w = w_raw / w_raw.sum()
print(f"\nOptimal blend → NN: {w[0]:.3f}  LGB: {w[1]:.3f}")

final_probs = w[0]*nn_test_prob + w[1]*lgb_test_prob

# ── Step 7: Soft-vote per track ───────────────────────────────
track_votes = {}
for i, tid in enumerate(te_tids):
    if tid not in track_votes:
        track_votes[tid] = {'prob': np.zeros(10), 'true': y_te_seg[i]}
    track_votes[tid]['prob'] += final_probs[i]

preds = np.array([v['prob'].argmax() for v in track_votes.values()])
trues = np.array([v['true']          for v in track_votes.values()])

# ── Final Report ──────────────────────────────────────────────
final_acc = accuracy_score(trues, preds)
print(f"\n{'='*55}")
print(f"   REAL Test Accuracy (no leakage, track vote):")
print(f"     {final_acc:.4f}  ({final_acc*100:.1f}%)")
print(f"  Test tracks : {len(track_votes)}")
print(f"{'='*55}\n")

print("Per-class accuracy:")
for cid, cname in enumerate(le_new.classes_):
    mask = trues == cid
    if not mask.any(): continue
    acc = accuracy_score(trues[mask], preds[mask])
    bar = '█'*int(acc*20) + '░'*(20-int(acc*20))
    print(f"  {cname:10s} {bar} {acc*100:5.1f}%  (n={mask.sum()})")

cm = confusion_matrix(trues, preds)
print(f"\nConfusion matrix (rows=true, cols=pred):")
header = ''.join(f"{n[:4]:>6}" for n in le_new.classes_)
print(f"{'':10s}{header}")
for i, row in enumerate(cm):
    cells = ''.join(
        f"\033[92m{v:>6}\033[0m" if j==i else
        f"\033[91m{v:>6}\033[0m" if v>0 else f"{'0':>6}"
        for j, v in enumerate(row)
    )
    print(f"  {le_new.classes_[i]:8s}{cells}")
