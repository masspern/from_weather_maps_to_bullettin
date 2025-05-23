# Massimo Perna
# Consorzio LaMMA
# 2025


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from collections import defaultdict


# === PARAMETRI ===
NUM_POS = 34
BATCH_SIZE = 8
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
X_DIR = "/path/to/meteo data/"
OUT_NPZ = "/path/to/metadata.npz"

# === MASCHERA VALIDA ===
with open("/path/to//pos2cat.json") as f:
    pos2cat = json.load(f)
pos2cat = {str(k): v[0] if isinstance(v, list) else v for k, v in pos2cat.items()}

cat2label2id = {
    "nuvola": json.load(open("/path/to/label/nuvola.json")),
    "mare": json.load(open("/path/to/label/mare.json")),
    "vento": json.load(open("/path/to/label/vento.json")),
    "temperatura": json.load(open("/path/to/label/temperatura.json")),
}

NUM_ID = max([max(d.values()) for d in cat2label2id.values()]) + 1
valid_mask = torch.zeros((NUM_POS, NUM_ID), dtype=torch.bool)

for pos, categoria in pos2cat.items():
    if categoria in cat2label2id:
        for id_ in cat2label2id[categoria].values():
            valid_mask[int(pos), id_] = True

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from collections import defaultdict
import numpy as np
import torch


# === DATASET ===
class MeteoMapDataset(torch.utils.data.Dataset):
    def __init__(self, metadati_path, x_dir):
        data = np.load(metadati_path, allow_pickle=True)
        self.file_X = data["file_X"]
        self.id_icon = data["id_icon"]
        self.x_dir = x_dir

    def __len__(self):
        return len(self.file_X)

    def __getitem__(self, idx):
        x_path = os.path.join(self.x_dir, self.file_X[idx])
        x = torch.tensor(np.load(x_path).astype(np.float32))
        y = np.array(self.id_icon[idx])
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        elif y.ndim > 1:
            y = y.squeeze()
        y = torch.tensor(y, dtype=torch.long)
        return x, y, self.file_X[idx]

# === MODELLO RESNET50 MIGLIORATO ===
class IconPositionPredictor(nn.Module):
    def __init__(self, num_positions=34, num_ids=72, dropout_rate=0.5):
        super().__init__()
        self.num_positions = num_positions
        self.num_ids = num_ids
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_positions * num_ids)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out.reshape(-1, self.num_positions, self.num_ids)

#   FUNZIONE DI FOCAL LOSS  per migliorare recall su vento
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # pesi per classe (torch tensor o None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:
            at = self.alpha.gather(0, target)
            logpt *= at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss




# === LOSS CON MASCHERA DINAMICA ===
class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, base_mask, cat2label2id, pos2cat, class_weights):
        super().__init__()
        self.base_mask = base_mask
        self.cat2label2id = cat2label2id
        self.pos2cat = pos2cat
        self.class_weights = class_weights
        self.ce = FocalLoss(alpha=class_weights.to(DEVICE), gamma=2.0)

        self.luna_ids = set()
        self.diurna_ids = set()
              
        print(" ID luna:", sorted(self.luna_ids))
        print(" ID giorno:", sorted(self.diurna_ids))
        print(" Verifica ID luna / giorno")
        for cat, label2id in cat2label2id.items():
            for label, idx in label2id.items():
                if "luna" in label.lower():
                    self.luna_ids.add(idx)
                    label_giorno = label.lower().replace("_luna", "").replace("luna", "").strip("_")
                    
                    for other_label, other_idx in label2id.items():
                        if other_label.lower() == label_giorno:
                            self.diurna_ids.add(other_idx)


    def forward(self, logits, targets, filenames):
        B, POS, C = logits.shape
        base_mask = self.base_mask.to(logits.device).unsqueeze(0).repeat(B, 1, 1)

        for i, fname in enumerate(filenames):
            step = int(os.path.splitext(fname)[0].split("_")[-1])
            for pos in range(POS):
                cat = self.pos2cat.get(str(pos), None)
                if cat != "nuvola":
                    continue
                if step not in [3, 6, 9]:
                    base_mask[i, pos, list(self.luna_ids)] = False

        logits = logits.reshape(-1, C)
        targets = targets.reshape(-1)
        mask = base_mask.reshape(-1, C)
        masked_logits = logits.masked_fill(~mask, -1e9)
        invalid = mask[torch.arange(mask.size(0)), targets] == 0
            # Debug: controlla target bloccati
        invalid = mask[torch.arange(mask.size(0)), targets] == 0
        if invalid.any():
            print("Target BLOCCATI dalla maschera:")
            for idx in torch.nonzero(invalid).flatten()[:10]:
                batch_idx = idx.item() // POS
                pos_idx = idx.item() % POS
                target_id = targets[idx].item()
                allowed_ids = mask[idx].nonzero().flatten().tolist()
                filename = filenames[batch_idx]
                cat = self.pos2cat.get(str(pos_idx), '?')
                print(f"  ➤ File: {filename}, Pos: {pos_idx}, Cat: {cat}, Target ID: {target_id}, Ammessi: {allowed_ids}")

        return self.ce(masked_logits, targets)




# === BILANCIAMENTO CLASSI ===
id2freq = {34: 33682, 35: 20532, 18: 17504, 30: 13547, 29: 7825, 32: 7788, 19: 7095, 31: 6127, 36: 6127, 7: 5712, 37: 3693,
          23: 3554, 60: 3510, 33: 3297, 40: 2967, 8: 2965, 9: 2801, 70: 2740, 24: 2700, 20: 2640, 2: 2605, 52: 2567, 56: 2547,
          26: 2377, 46: 1902, 21: 1878, 53: 1695, 63: 1596, 38: 1548, 41: 1316, 39: 1302, 49: 1222, 66: 1192, 25: 851, 27: 824,
          67: 652, 12: 584, 43: 581, 22: 573, 71: 485, 6: 469, 64: 463, 45: 458, 3: 458, 47: 456, 28: 388, 42: 323, 57: 296,
          11: 249, 50: 215, 4: 185, 54: 160, 15: 156, 61: 96, 16: 92, 68: 81, 1: 70, 14: 62, 10: 40, 44: 37, 17: 21, 51: 6,
          5: 5, 65: 1}
'''
freq_array = np.ones(NUM_ID)
for id_, freq in id2freq.items():
    freq_array[id_] = freq
weights = 1.0 / (freq_array + 1e-6)
weights = weights / weights.sum() * NUM_ID
class_weights = torch.tensor(weights, dtype=torch.float32)
'''
def compute_class_weights(id2freq, num_classes, gamma=1.0, min_weight=0.1, max_weight=10.0):
    freq_array = np.ones(num_classes)
    for id_, freq in id2freq.items():
        freq_array[id_] = freq

    # Inverse frequency + stabilizzazione
    weights = 1.0 / (freq_array + 1e-3)

    # Aggiusta contrasto: più gamma > 1, più sbilanciati i pesi
    weights = weights ** gamma

    # Normalizza rispetto al massimo
    weights = weights / weights.max()

    # Rescale tra [min_weight, max_weight]
    weights = weights * (max_weight - min_weight) + min_weight

    return torch.tensor(weights, dtype=torch.float32)

class_weights = compute_class_weights(id2freq, NUM_ID, gamma=1.2, min_weight=0.2, max_weight=5.0)


# === TRAINING ===
dataset = MeteoMapDataset(metadati_path=OUT_NPZ, x_dir=X_DIR)
train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, shuffle=True)
train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = IconPositionPredictor(num_positions=NUM_POS, num_ids=NUM_ID).to(DEVICE)
criterion = MaskedCrossEntropyLoss(
    base_mask=valid_mask,
    cat2label2id=cat2label2id,
    pos2cat=pos2cat,
    class_weights=class_weights
)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

train_losses, val_losses = [], []
train_accs, val_accs = [], []


def evaluate_per_category_detailed(masked_logits, targets, base_mask, filenames, pos2cat, luna_ids, id2label=None, step_range=[3, 6, 9]):
    from collections import defaultdict
    import pandas as pd

    B, POS, C = base_mask.shape
    logits = masked_logits.clone().detach().cpu()
    targets = targets.clone().detach().cpu()
    base_mask = base_mask.cpu()

    records = []

    for b in range(B):
        step = int(filenames[b].split("_")[-1].split(".")[0])
        for pos in range(POS):
            cat = pos2cat.get(str(pos), None)
            if cat is None:
                continue

            valid_mask = base_mask[b, pos].clone()
            if cat == "nuvola" and step not in step_range:
                valid_mask[list(luna_ids)] = False

            logit = logits[b, pos]
            masked_logit = logit.clone()
            masked_logit[~valid_mask] = -1e9

            pred_id = torch.argmax(masked_logit).item()
            true_id = targets[b, pos].item()

            if valid_mask[true_id]:
                records.append({
                    "filename": filenames[b],
                    "pos": pos,
                    "cat": cat,
                    "true_id": true_id,
                    "pred_id": pred_id,
                    "true_label": id2label.get(true_id, str(true_id)) if id2label else str(true_id),
                    "pred_label": id2label.get(pred_id, str(pred_id)) if id2label else str(pred_id),
                    "correct": int(true_id == pred_id)
                })

    df = pd.DataFrame(records)

    # Metriche per categoria
    results = {}
    for cat in df['cat'].unique():
        df_cat = df[df['cat'] == cat]
        acc = (df_cat['correct']).mean()
        prec, recall, f1, _ = precision_recall_fscore_support(
            df_cat["true_id"], df_cat["pred_id"], average='weighted', zero_division=0
        )
        results[cat] = {
            "accuracy": acc,
            "precision": prec,
            "recall": recall,
            "f1_score": f1,
            "n": len(df_cat)
        }

    return results, df


def plot_confusion_per_category(all_preds, all_targets, pos2cat, id2label=None, save_dir="/mnt/d/DATA/DRIVE-F/LAVORO/AI_BOLLETTINI/dataset_modello/resnet50/"):
    from collections import defaultdict
    os.makedirs(save_dir, exist_ok=True)

    # Aggrega predizioni e target per categoria
    cat2preds = defaultdict(list)
    cat2tgts = defaultdict(list)
    for b in range(len(all_preds)):
        for pos in range(len(all_preds[b])):
            cat = pos2cat.get(str(pos), None)
            if cat:
                cat2preds[cat].append(all_preds[b][pos])
                cat2tgts[cat].append(all_targets[b][pos])

    for cat, preds in cat2preds.items():
        tgts = cat2tgts[cat]
        if not preds:
            continue

        y_true = np.array(tgts)
        y_pred = np.array(preds)

        labels = sorted(set(y_true.tolist() + y_pred.tolist()))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        label_names = [id2label.get(i, str(i)) for i in labels] if id2label else [str(i) for i in labels]

        plt.figure(figsize=(max(6, len(labels) * 0.3), 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                    xticklabels=label_names, yticklabels=label_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - Categoria: {cat}")
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(fontsize=6)
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"confmat_{cat}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Salvata: {out_path}")

id2label = {}
for cat, label2id in cat2label2id.items():
    for label, idx in label2id.items():
        id2label[idx] = label

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_idx, (x, y, filenames) in enumerate(tqdm(train_loader, desc=f"Train Epoca {epoch+1}/{EPOCHS}")):
        x, y = x.to(DEVICE), y.to(DEVICE)
        print(f"Train batch: x.shape={x.shape}, y.shape={y.shape}")
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y, filenames)
        loss.backward()
        optimizer.step()
 
   
        # Calcolo accuracy del batch
        preds = logits.argmax(dim=-1)
        corrects = (preds == y).float().sum().item()
        totals = y.numel()
        batch_acc = corrects / totals
        if loss.item() > 1e4:
            print(f"\n Loss anomala: {loss.item():.4f}")
            print("Filenames:", filenames)
            print("Target sample:", y[0])
            print("Valid IDs in mask:", (criterion.base_mask[0] > 0).sum(dim=-1))



         # Logging
        print(f"[Train][Batch {batch_idx+1}] Loss: {loss.item():.4f} | Accuracy: {batch_acc:.4f}")
     
        
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=-1) == y).float().sum().item()
        total += y.numel()
    train_losses.append(total_loss / total)
    train_accs.append(correct / total)

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    from collections import defaultdict

    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    all_logits = []
    all_targets = []
    all_filenames = []
    
    with torch.no_grad():
        for x, y, fnames in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
    
            # Calcolo loss
            loss = criterion(logits, y, fnames)
            val_loss += loss.item() * x.size(0)
            if batch_idx == 0:  # primo batch
                print("Logits (shape):", logits.shape)
                print("Targets (shape):", y.shape)
                print("Esempio logits[0]:", logits[0])
                print("Esempio target[0]:", y[0])
                print("Loss batch:", loss.item())
    
            # Per accuracy globale
            preds = logits.argmax(dim=-1)
            val_correct += (preds == y).float().sum().item()
            val_total += y.numel()
    
            # Salva per metriche per categoria
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())
            all_filenames.extend(fnames)
    
    # Accumulate tensori e mask
    logits_all = torch.cat(all_logits, dim=0)  # (N, 34, C)
    targets_all = torch.cat(all_targets, dim=0)  # (N, 34)
    
    # Costruzione della mask come nella loss
    base_mask = criterion.base_mask.to(logits_all.device).unsqueeze(0).repeat(logits_all.size(0), 1, 1)
    
    for i, fname in enumerate(all_filenames):
        step = int(os.path.splitext(fname)[0].split("_")[-1])
        for pos in range(logits_all.size(1)):
            cat = criterion.pos2cat.get(str(pos), None)
            if cat == "nuvola" and step not in [3, 6, 9]:
                base_mask[i, pos, list(criterion.luna_ids)] = False
    
    masked_logits_all = logits_all.masked_fill(~base_mask, -1e9)
    
    # Calcolo metriche per categoria
    metrics_per_cat, df_per_category = evaluate_per_category_detailed(
    masked_logits_all, targets_all, base_mask,
    all_filenames, criterion.pos2cat, criterion.luna_ids,
    id2label=id2label  # passalo qui
    )

    df_per_category.to_csv("/path/to/output_details_per_category.csv", index=False)


    
    # Stampa metriche per categoria
    print("\n METRICHE DI VALIDAZIONE PER CATEGORIA:")
    for cat, vals in metrics_per_cat.items():
        print(f"[{cat.upper():7}] Acc={vals['accuracy']:.3f}  Prec={vals['precision']:.3f}  "
              f"Recall={vals['recall']:.3f}  F1={vals['f1_score']:.3f}  (N={vals['n']})")
    
    # Salvataggio globali
    val_losses.append(val_loss / val_total)
    val_accs.append(val_correct / val_total)

    from collections import defaultdict

    # Costruisci lista di pred e target (shape: B x POS)
    preds_all = masked_logits_all.argmax(dim=-1).tolist()
    tgts_all = targets_all.tolist()
    
    # Genera confusion matrix per categoria
    plot_confusion_per_category(preds_all, tgts_all, criterion.pos2cat, id2label)


    
    # Appiattisci tutte le predizioni e target validi
    y_true = []
    y_pred = []
    for b in range(logits_all.size(0)):
        for pos in range(logits_all.size(1)):
            cat = criterion.pos2cat.get(str(pos), None)
            if cat is None:
                continue
            step = int(os.path.splitext(all_filenames[b])[0].split("_")[-1])
            valid_mask = base_mask[b, pos].clone()
            if cat == "nuvola" and step not in [3, 6, 9]:
                valid_mask[list(criterion.luna_ids)] = False
    
            logit = logits_all[b, pos]
            masked_logit = logit.clone()
            masked_logit[~valid_mask] = -1e9
    
            pred = torch.argmax(masked_logit).item()
            tgt = targets_all[b, pos].item()
    
            if valid_mask[tgt]:
                y_pred.append(pred)
                y_true.append(tgt)
    '''
    
    # Calcola matrice
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_ID)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    '''
    
    scheduler.step(val_losses[-1])

    print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, Train Acc={train_accs[-1]:.4f}, Val Acc={val_accs[-1]:.4f}")

    torch.save(model.state_dict(), f"/path/to/model/model_epoch_{epoch+1}.pt")

    # === GRAFICI ===
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss per Epoca')
    plt.grid()
    plt.savefig("/path/to/loss_curve.png")
    plt.show()
    
    plt.figure()
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy per Epoca')
    plt.grid()
    plt.savefig("/path/to/accuracy_curve.png")
    plt.show()
