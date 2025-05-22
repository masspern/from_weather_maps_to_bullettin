
# Funzione `evaluate_per_category_detailed`

## Scopo

Questa funzione valuta le performance del modello suddividendo i risultati **per categoria** (es. *nuvola*, *mare*, ecc.) e per posizione, rispettando eventuali maschere di validità (inclusa la logica specifica per la **categoria "nuvola"** legata al "passo" temporale). Produce metriche dettagliate e opzionalmente restituisce un DataFrame con le predizioni.

---

## Firma

```python
evaluate_per_category_detailed(
    masked_logits,
    targets,
    base_mask,
    filenames,
    pos2cat,
    luna_ids,
    id2label=None,
    step_range=[3, 6, 9]
)
```

---

## Parametri

| Nome           | Tipo                     | Descrizione                                                                 |
|----------------|--------------------------|-----------------------------------------------------------------------------|
| `masked_logits`| `Tensor[B, POS, C]`       | Output del modello (logits mascherati)                                     |
| `targets`      | `Tensor[B, POS]`          | Target ground truth                                                        |
| `base_mask`    | `Tensor[B, POS, C]`       | Maschera base che indica quali classi sono valide per ogni posizione       |
| `filenames`    | `List[str]`               | Lista dei nomi dei file per ciascun elemento del batch                     |
| `pos2cat`      | `Dict[str, str]`          | Mappa delle posizioni alle categorie (`pos -> categoria`)                  |
| `luna_ids`     | `Set[int]`                | Set di ID validi solo nei passi notturni                                   |
| `id2label`     | `Dict[int, str]`, opzionale| Mappa degli ID alle etichette leggibili (usata per output più chiaro)      |
| `step_range`   | `List[int]`               | Passi orari considerati validi per le icone "luna"                         |

---

## Output

```python
results, df
```

- `results`: `Dict[str, Dict[str, Any]]`  
  Dizionario con metrica aggregata per categoria, contenente:
  - `accuracy`
  - `precision`
  - `recall`
  - `f1_score`
  - `n`: numero di campioni

- `df`: `pandas.DataFrame`  
  DataFrame con dettaglio riga per riga delle predizioni, contenente:
  - `filename`, `pos`, `cat`, `true_id`, `pred_id`, `true_label`, `pred_label`, `correct`

---

## Logica speciale

- Per le **posizioni con categoria "nuvola"**, se il `step` (estratto dal nome del file) **non è nei valori di `step_range`**, allora vengono disabilitati tutti gli ID presenti in `luna_ids`.
- Il sistema considera la **maschera base** e la **modifica dinamica per ogni file e posizione**.
- Le predizioni vengono effettuate solo sugli ID validi (logits non mascherati a `-1e9`).

---

## Esempio di utilizzo

```python
results, df_preds = evaluate_per_category_detailed(
    masked_logits=logits,
    targets=targets,
    base_mask=valid_mask,
    filenames=filenames,
    pos2cat=pos2cat,
    luna_ids=criterion.luna_ids,
    id2label=id2label
)
```

---

## Esempio di struttura di `results`

```python
{
  'nuvola': {
    'accuracy': 0.85,
    'precision': 0.87,
    'recall': 0.84,
    'f1_score': 0.85,
    'n': 1275
  },
  'vento': {
    'accuracy': 0.92,
    ...
  }
}
```
