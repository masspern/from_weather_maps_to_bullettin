#   ESTRAI VARIABILI DAI GRIB

import os
import numpy as np
import xarray as xr
import datetime

# === CONFIG ===
GRIB_DIR = "/path/to/bulletin/"
OUT_DIR = "/path/to/meteo_data/"
VARIABILI = ["tcc", "tp", "u10", "v10", "t2m"]

INDICI_E_STEP = {
    1: 12,
    2: 18,
    3: 24,
    4: 36,
    5: 42,
    6: 48,
    7: 60,
    8: 66,
    9: 72
}

os.makedirs(OUT_DIR, exist_ok=True)

def estrai_x(grib_path, step_hour):
    try:
        ds = xr.open_dataset(grib_path, engine="cfgrib", decode_timedelta=True)
        dati = []
        for var in VARIABILI:
            if var not in ds:
                print(f"⚠️ Variabile {var} non trovata in {grib_path}")
                return None
            data = ds[var].sel(step=datetime.timedelta(hours=step_hour))
            dati.append(data.values)
        return np.stack(dati, axis=0)  # shape (5, H, W)
    except Exception as e:
        print(f"⛔ Errore GRIB {grib_path} step {step_hour}: {e}")
        return None

# === LOOP sui GRIB ===
for fname in sorted(os.listdir(GRIB_DIR)):
    if not fname.endswith(".grib1"):
        continue

    date_str = fname.split("_")[-1].replace(".grib1", "")
    try:
        base_date = datetime.datetime.strptime(date_str, "%Y%m%d")
    except:
        print(f"⚠️ Nome file non valido: {fname}")
        continue

    grib_path = os.path.join(GRIB_DIR, fname)

    for idx, step_hour in INDICI_E_STEP.items():
        step_time = base_date.strftime("%Y-%m-%d") + f"_{idx}"
        X = estrai_x(grib_path, step_hour)
        if X is None:
            continue

        out_path = os.path.join(OUT_DIR, f"{step_time}.npy")
        np.save(out_path, X)
        print(f"✅ Salvato: {out_path}  | shape: {X.shape}")

