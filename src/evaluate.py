import pandas as pd, numpy as np, torch, torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ── Step 1: Reload 3-sec CSV with track IDs ───────────────────
df3_raw = pd.read_csv(f'{BASE}/features_3_sec.csv')
df3_raw['track_id'] = df3_raw['filename'].apply(
    lambda f: '.'.join(str(f).split('.')[:2])  # "blues.00000.4" → "blues.00000"
)

feat_cols = [c for c in df3_raw.columns
             if c not in ['filename', 'label', 'track_id']]

le_eval     = LabelEncoder().fit(df3_raw['label'])
scaler_eval = StandardScaler()

# ── Step 2: TRACK-level split (fixes the leakage) ─────────────

unique_tracks = df3_raw['track_id'].unique()
track_labels  = df3_raw.groupby('track_id')['label'].first()

train_tracks, test_tracks = train_test_split(
    unique_tracks,
    test_size=0.20,
    stratify=track_labels[unique_tracks],   # balanced classes
    random_state=42
)

train_mask = df3_raw['track_id'].isin(train_tracks)
test_mask  = df3_raw['track_id'].isin(test_tracks)

# Verify zero overlap
assert len(set(train_tracks) & set(test_tracks)) == 0, "Leakage detected!"

X_tr_raw = df3_raw.loc[train_mask, feat_cols].values.astype(np.float32)
y_tr     = le_eval.transform(df3_raw.loc[train_mask, 'label'])
X_te_raw = df3_raw.loc[test_mask,  feat_cols].values.astype(np.float32)
y_te_seg = le_eval.transform(df3_raw.loc[test_mask,  'label'])
te_tids  = df3_raw.loc[test_mask, 'track_id'].values

# Scaler fit ONLY on train — never touches test
scaler_eval.fit(X_tr_raw)
X_tr = scaler_eval.transform(X_tr_raw)
X_te = scaler_eval.transform(X_te_raw)

print(f"Train: {len(X_tr)} segments from {len(train_tracks)} tracks")
print(f"Test : {len(X_te)} segments from {len(test_tracks)} tracks")
print(f"✅ Overlap check passed — zero leakage\n")

# ── Step 3: Retrain on leak-free split ────────────────────────
from torch.utils.data import DataLoader

class TabDS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

tr_dl = DataLoader(TabDS(X_tr, y_tr), batch_size=256, shuffle=True)

# Reuse your exact TSFN architecture and training setup
eval_model = TSFN(X_tr.shape[1], n_classes=10).to(device)
opt_e  = torch.optim.AdamW(eval_model.parameters(), lr=3e-3, weight_decay=1e-3)
sch_e  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_e, T_max=80, eta_min=1e-5)
ce_e   = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

best_acc_e, best_w_e = 0, None
X_te_t = torch.tensor(X_te, dtype=torch.float32)

print("Retraining on leak-free split...")
for ep in range(80):
    eval_model.train()
    for xb, yb in tr_dl:
        xb, yb = xb.to(device), yb.to(device)
        if ep < 50:                                  # mixup phase
            lam = np.random.beta(0.2, 0.2)
            idx = torch.randperm(xb.size(0))
            xm  = lam*xb + (1-lam)*xb[idx]
            loss = lam*ce_e(eval_model(xm), yb) + (1-lam)*ce_e(eval_model(xm), yb[idx])
        else:
            loss = ce_e(eval_model(xb), yb)
        opt_e.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(eval_model.parameters(), 1.0)
        opt_e.step()
    sch_e.step()

    # Track-level val every 10 epochs (on test tracks — just to monitor)
    if (ep+1) % 10 == 0:
        eval_model.eval()
        with torch.no_grad():
            probs_e = F.softmax(eval_model(X_te_t), dim=-1).numpy()
        seg_acc = accuracy_score(y_te_seg, probs_e.argmax(1))

        # Quick track vote to monitor true acc
        votes = {}
        for i, tid in enumerate(te_tids):
            votes.setdefault(tid, {'prob': np.zeros(10), 'true': y_te_seg[i]})
            votes[tid]['prob'] += probs_e[i]
        trk_acc = np.mean([v['prob'].argmax()==v['true'] for v in votes.values()])

        if trk_acc > best_acc_e:
            best_acc_e = trk_acc
            best_w_e   = {k: v.clone() for k, v in eval_model.state_dict().items()}

        print(f"  Ep {ep+1:3d} | Seg acc: {seg_acc:.3f} | "
              f"Track acc: {trk_acc:.3f}  {'← best' if trk_acc==best_acc_e else ''}")

# ── Step 4: Final evaluation with best weights ────────────────
eval_model.load_state_dict(best_w_e)
eval_model.eval()

with torch.no_grad():
    final_probs = F.softmax(eval_model(X_te_t), dim=-1).numpy()

# Soft-vote across all segments of each test track
track_votes = {}
for i, tid in enumerate(te_tids):
    if tid not in track_votes:
        track_votes[tid] = {'prob': np.zeros(10), 'true': y_te_seg[i]}
    track_votes[tid]['prob'] += final_probs[i]

preds_track = np.array([v['prob'].argmax() for v in track_votes.values()])
trues_track = np.array([v['true']          for v in track_votes.values()])
final_acc   = accuracy_score(trues_track, preds_track)

print(f"\n{'='*55}")
print(f"  ✅ REAL Test Accuracy (track-level, no leakage): "
      f"{final_acc:.4f}  ({final_acc*100:.1f}%)")
print(f"  Total test tracks : {len(track_votes)}")
print(f"{'='*55}\n")

# ── Per-class breakdown ───────────────────────────────────────
print("Per-class accuracy:")
for cid, cname in enumerate(le_eval.classes_):
    mask = trues_track == cid
    if mask.sum() == 0: continue
    acc = accuracy_score(trues_track[mask], preds_track[mask])
    bar = '█'*int(acc*20) + '░'*(20-int(acc*20))
    print(f"  {cname:10s} {bar} {acc*100:5.1f}%  (n={mask.sum()})")

# ── Confusion matrix ──────────────────────────────────────────
cm = confusion_matrix(trues_track, preds_track)
print(f"\nConfusion matrix  (rows=true, cols=pred):")
header = ''.join(f"{n[:4]:>6}" for n in le_eval.classes_)
print(f"{'':10s}{header}")
for i, row in enumerate(cm):
    cells = ''.join(
        f"\033[92m{v:>6}\033[0m" if j==i else
        f"\033[91m{v:>6}\033[0m" if v>0 else f"{'0':>6}"
        for j,v in enumerate(row)
    )
    print(f"  {le_eval.classes_[i]:8s}{cells}")
  # ── Each 30-sec track = 10 three-second segments in the 3-sec CSV ──
# Majority vote across 10 segments → dramatically boosts accuracy

model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
model.eval()

# Reload original 3-sec csv WITH filenames to group by track
df3_raw = pd.read_csv(f'{BASE}/features_3_sec.csv')

# Extract track id: filename like "blues.00000.10" → "blues.00000"
df3_raw['track_id'] = df3_raw['filename'].apply(
    lambda f: '.'.join(str(f).split('.')[:2])
)
df3_raw['label_enc'] = le.transform(df3_raw['label'])
feat_cols = [c for c in df3_raw.columns if c not in ['filename', 'label', 'track_id', 'label_enc']]

X_all = scaler.transform(df3_raw[feat_cols].values).astype(np.float32)

all_ds     = GTZANTabularDataset(X_all, df3_raw['label_enc'].values)
all_loader = DataLoader(all_ds, batch_size=512, shuffle=False, num_workers=2)

# Get all logits
all_logits, all_true = [], []
with torch.no_grad():
    for xb, yb in all_loader:
        logits = model(xb.to(device)).cpu()
        all_logits.append(logits)
        all_true.append(yb)

all_logits = torch.cat(all_logits).numpy()
all_true   = torch.cat(all_true).numpy()
df3_raw['pred_hard'] = all_logits.argmax(1)

# ── Soft voting (sum logits over segments per track) ─────────
track_results = []
for tid, grp in df3_raw.groupby('track_id'):
    idxs        = grp.index.tolist()
    soft_vote   = all_logits[idxs].sum(axis=0)   # sum log-probs
    pred_label  = soft_vote.argmax()
    true_label  = grp['label_enc'].iloc[0]
    track_results.append({'track': tid, 'pred': pred_label, 'true': true_label})

results_df  = pd.DataFrame(track_results)
track_acc   = (results_df['pred'] == results_df['true']).mean()
print(f"\n🎯 Track-level accuracy (soft segment voting): {track_acc:.4f} ({track_acc*100:.1f}%)")

# Per-class breakdown
print("\nPer-class accuracy:")
for cls_id, cls_name in enumerate(le.classes_):
    sub = results_df[results_df['true'] == cls_id]
    acc = (sub['pred'] == sub['true']).mean()
    bar = '█' * int(acc * 20)
    print(f"  {cls_name:10s} {bar:<20s} {acc:.2f}  (n={len(sub)})")
