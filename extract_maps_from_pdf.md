- Massimo Perna
- Consorzio LaMMA
- 2025


# 🗺️ Estrazione di Mappe da PDF (Griglia 3x3)

Questo script Python estrae automaticamente 9 mappe da PDF (presumibilmente disposte in una griglia 3x3), le ritaglia e le salva come immagini `.png`.

---

## 📂 Struttura delle Cartelle

- **Input PDF**:  
  `pdf_folder = "path/to/pdfs"`

- **Output PNG**:  
  `output_folder = "path/to/maps"`

---

## ⚙️ Configurazione Principale

- `dpi = 150` – risoluzione dell'immagine derivata dal PDF
- `min_area = 20000` – area minima per accettare un rettangolo come mappa
- `aspect_ratio_range = (0.8, 2.2)` – range accettabile del rapporto larghezza/altezza

---

## 🧠 Funzioni Chiave

### `aspect_ratio_filter(bbox, min_ratio, max_ratio, min_area)`

Filtra i rettangoli troppo piccoli o con proporzioni anomale.

---

### `is_valid_crop(crop, threshold=20)`

Verifica se il contenuto dell'immagine ritagliata non è vuoto (soglia sullo std-dev in scala di grigi).

---

### `ordina_3x3_per_riga_colonna(boxes)`

Ordina i 9 rettangoli trovati in righe e colonne, basandosi sulle coordinate dei loro centri.

⚠️ Solleva un errore se i box non sono esattamente 9.

---

### `rileva_blocchi(img_pil)`

Utilizza OpenCV per:

1. Convertire l'immagine in scala di grigi.
2. Binarizzarla.
3. Trovare i contorni.
4. Estrarre e filtrare bounding boxes validi.

---

## 🔄 Processo di Estrazione

1. **Scansione PDF**:  
   Legge tutti i file `.pdf` dalla cartella.

2. **Conversione in Immagine**:  
   Usa `PyMuPDF` per convertire la prima pagina del PDF in immagine PIL.

3. **Rilevamento Blocchi**:  
   Trova rettangoli candidati alle mappe.

4. **Ordinamento**:  
   Ordina i box in una struttura coerente 3x3.

5. **Ritaglio + Centratura**:  
   Ogni mappa è centrata in un'immagine bianca di dimensioni uniformi.

6. **Salvataggio**:  
   Le immagini vengono salvate come:
