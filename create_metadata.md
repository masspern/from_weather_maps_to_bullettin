# Crea Metadati Etichette (con ID icona unificati)

Questo script Python genera un file `.npz` contenente metadati strutturati a partire da etichette annotate in formato JSON e dati associati in formato `.npy`. Viene utilizzato per standardizzare le etichette in diverse categorie meteorologiche (nuvola, mare, vento, temperatura) e associarle a posizioni specifiche all'interno di ogni esempio.

---

## Input

- **`X_DIR`**: Cartella contenente i file `.npy` con le feature degli esempi.
- **`JSON_LABELS_DIR`**: Cartella con file JSON contenenti le etichette (`*_labels.json`).
- **`cat2label2id`**: Mapping JSON delle etichette originali a ID unificati per ciascuna categoria (`nuvola`, `mare`, `vento`, `temperatura`).
- **`date_sospette`**: Giorni da escludere (es. dati incompleti o corrotti).

---

## Elaborazione

Per ogni file JSON:

1. **Controlla date sospette**: Se il giorno è incluso in `date_sospette`, il file viene ignorato.
2. **Verifica esistenza del file `.npy`** corrispondente.
3. **Carica etichette** dal file JSON.
4. **Inizializza array** `id_array` e `cat_array` (lunghezza 34) con valore di default `-1`.
5. **Per ogni categoria e icona**:
   - Recupera `posizione_idx` e `id`.
   - Verifica validità di posizione e ID.
   - Assegna `icon_id` e `cat_idx` alle rispettive posizioni.

6. Se l'array contiene valori incompleti (`-1`), l'esempio viene scartato.

---

## Output

Un file `.npz` contenente:

- `file_X`: Nome dei file `.npy` elaborati.
- `id_icon`: Matrice `[N, 34]` con gli ID unificati delle icone.
- `categoria`: Matrice `[N, 34]` con gli ID delle categorie (0 = nuvola, 1 = vento, ecc.).
- `id_posizione`: Array `[34]` con le posizioni (da 0 a 33).

---

## Esempio di uso

```python
npz = np.load(OUT_NPZ)
print(npz['file_X'].shape)        # → (N,)
print(npz['id_icon'].shape)       # → (N, 34)
print(npz['categoria'].shape)     # → (N, 34)
print(npz['id_posizione'].shape)  # → (34,)
