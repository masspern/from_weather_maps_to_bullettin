import os
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2

# === CONFIGURAZIONE ===
pdf_folder = "path/to/pdfs"
output_folder = "path/to/maps"
dpi = 150
min_area = 20000
aspect_ratio_range = (0.8, 2.2)
os.makedirs(output_folder, exist_ok=True)

# === FUNZIONI ===
def aspect_ratio_filter(bbox, min_ratio, max_ratio, min_area):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    area = w * h
    if area < min_area:
        return False
    ratio = w / h
    return min_ratio <= ratio <= max_ratio

def is_valid_crop(crop, threshold=20):
    gray = np.array(crop.convert("L"))
    return gray.std() > threshold

def ordina_3x3_per_riga_colonna(boxes):
    if len(boxes) != 9:
        raise ValueError("⚠️ Sono richieste esattamente 9 mappe.")
    centers = [(i, (b[1] + b[3]) // 2, (b[0] + b[2]) // 2) for i, b in enumerate(boxes)]
    centers_np = np.array(centers)
    row_indices = np.argsort(centers_np[:, 1])
    righe = [[], [], []]
    for idx in row_indices:
        y = centers_np[idx, 1]
        if len(righe[0]) == 0 or abs(y - np.mean([c[1] for c in righe[0]])) < 50:
            righe[0].append(centers_np[idx])
        elif len(righe[1]) == 0 or abs(y - np.mean([c[1] for c in righe[1]])) < 50:
            righe[1].append(centers_np[idx])
        else:
            righe[2].append(centers_np[idx])
    boxes_ordinati = []
    for riga in righe:
        riga_ordinata = sorted(riga, key=lambda c: c[2])
        for c in riga_ordinata:
            idx = int(c[0])
            boxes_ordinati.append(boxes[idx])
    return boxes_ordinati

def rileva_blocchi(img_pil):
    img_cv = np.array(img_pil)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x0, y0, x1, y1 = x, y, x + w, y + h
        if aspect_ratio_filter((x0, y0, x1, y1), *aspect_ratio_range, min_area):
            boxes.append((x0, y0, x1, y1))
    return boxes

# === PROCESSING ===
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
print(f"📄 Trovati {len(pdf_files)} PDF")

for pdf_file in sorted(pdf_files):
    pdf_path = os.path.join(pdf_folder, pdf_file)
    pdf_base = os.path.splitext(pdf_file)[0]

    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

        boxes = rileva_blocchi(img)

        if len(boxes) < 9:
            print(f"⚠️ {pdf_file}: solo {len(boxes)} box trovati.")
            continue

        try:
            boxes = ordina_3x3_per_riga_colonna(boxes[:9])
        except Exception as e:
            print(f"❌ Errore ordinamento {pdf_file}: {e}")
            continue

        max_w = max(x1 - x0 for (x0, y0, x1, y1) in boxes)
        max_h = max(y1 - y0 for (x0, y0, x1, y1) in boxes)

        for i, (x0, y0, x1, y1) in enumerate(boxes):
            crop = img.crop((x0, y0, x1, y1))
            if not is_valid_crop(crop):
                print(f"⚠️ Crop non valido in {pdf_base}_{i+1}.png, salto.")
                continue

            fixed = Image.new("RGB", (max_w, max_h), (255, 255, 255))
            offset_x = (max_w - (x1 - x0)) // 2
            offset_y = (max_h - (y1 - y0)) // 2
            fixed.paste(crop, (offset_x, offset_y))

            filename = f"{pdf_base}_{i+1}.png"
            fixed.save(os.path.join(output_folder, filename))
            print(f"✅ Salvata: {filename} | size: {fixed.size}")

    except Exception as e:
        print(f"❌ Errore con {pdf_file}: {e}")
