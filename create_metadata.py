# Massimo Perna
# Consorzio LaMMA
# 2025



import os
import json
import numpy as np

# === Percorsi ===
X_DIR = '/path/to/meteo_data/'
JSON_LABELS_DIR = '/path/to/labels_json/'
OUT_NPZ = '/path/to/metadatipath/metadata.npz'

# === Carica mapping unificati
cat2label2id = {
    "nuvola": json.load(open("/path/to/label/nuvola.json")),
    "mare": json.load(open("/path/to/label/mare.json")),
    "vento": json.load(open("//path/to/label/vento.json")),
    "temperatura": json.load(open("/path/to/label/temperatura.json")),
}

# Calcola NUM_ID massimo
NUM_ID = max([max(d.values()) for d in cat2label2id.values()]) + 1

# === Mappatura categorie → interi
cat2idx = {
    "nuvola": 0,
    "vento": 1,
    "mare": 2,
    "temperatura": 3
}

# === Container
file_X = []
id_icon = []
categoria = []

# === Prima: imposta le date sospette ===
date_sospette = {
    "2011-04-23", "2011-10-17", "2012-04-21", "2012-04-24", "2012-07-27", "2011-12-02", "2011-12-08", "2011-12-16"  # <-- aggiungi qui i giorni da escludere
}


# === Loop principale ===
json_files = sorted([f for f in os.listdir(JSON_LABELS_DIR) if f.endswith('_labels.json')])

for json_name in json_files:
    base = json_name.replace("boll_toscana_", "").replace("_labels.json", "")
    giorno = base.split("_")[0]

    if giorno in date_sospette:
        print(f" Giorno sospetto, salto: {giorno}")
        continue

    fname = f"{base}.npy"
    json_path = os.path.join(JSON_LABELS_DIR, json_name)
    npy_path = os.path.join(X_DIR, fname)

    if not os.path.exists(npy_path):
        print(f" File .npy mancante per {fname}, salto")
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    # ... continua elaborazione ...

    id_array = np.full(34, -1, dtype=np.int64)
    cat_array = np.full(34, -1, dtype=np.int64)

    for categoria_nome, icone in data.items():
        if categoria_nome not in cat2idx:
            print(f"Categoria sconosciuta '{categoria_nome}' in {json_name}")
            continue
        cat_idx = cat2idx[categoria_nome]

        for icona in icone:
            pos = icona.get("posizione_idx")
            icon_id = icona.get("id")

            if pos is None or icon_id is None:
                print(f" Icona malformata in {json_name}: {icona}")
                continue
            if pos < 0 or pos >= 34:
                print(f" posizione_idx fuori range in {json_name}: {pos}")
                continue
            if icon_id < 0 or icon_id >= NUM_ID:
                print(f" ID fuori range in {json_name}: {icon_id}")
                continue

            id_array[pos] = icon_id
            cat_array[pos] = cat_idx

    if np.any(id_array == -1) or np.any(cat_array == -1):
        print(f" Dati incompleti nel file {json_name}, salto.")
        continue

    file_X.append(fname)
    id_icon.append(id_array)
    categoria.append(cat_array)

# === Salvataggio finale ===
file_X = np.array(file_X)
id_icon = np.stack(id_icon)
categoria = np.stack(categoria)
id_posizione = np.arange(34, dtype=np.int64)

np.savez(
    OUT_NPZ,
    file_X=file_X,
    id_icon=id_icon,
    categoria=categoria,
    id_posizione=id_posizione
)

print(f" Metadati salvati in: {OUT_NPZ}")
print(f" Totale esempi validi: {len(file_X)}")
