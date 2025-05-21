# 📌 Allineamento di Mappe con OpenCV (Feature Matching con ORB)

Questo script allinea una serie di immagini `.png` (mappe) rispetto a una "mappa master", usando **feature detection** con ORB (Oriented FAST and Rotated BRIEF) e trasformazioni affini.

---

## 🗂️ Struttura delle Cartelle

- **Mappa di riferimento ("master")**:  
  `/path/to//bull_2011-01-01_1.png`

- **Cartella input mappe da allineare**:  
  `/path/to/maps`

- **Cartella output per mappe allineate**:  
  `/path/to/maps`

---

## ⚙️ Configurazione

```python
master_path = '/path/to//bull_2011-01-01_1.png'
input_dir = '/path/to/maps'
output_dir = '/path/to/maps'
