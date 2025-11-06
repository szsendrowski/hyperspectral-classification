import os
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi


scene_dir = r"C:\Users\sciezka\do\folderu\ang20160917t203013_rfl_SIMPLE"
hdr_name = "ang20160917t203013_rdn_v1n2_img_bsq_rfl_simple.hdr"
bands_to_check = [30, 100, 150]  # przykładowe pasma


# import
hdr_path = os.path.join(scene_dir, hdr_name)
base = hdr_path.replace(".hdr", "")
candidates = [base + ext for ext in ["", ".bsq", ".img", ".dat", ".bin"]]
img_path = next((c for c in candidates if os.path.exists(c)), None)
if img_path is None:
    raise FileNotFoundError(f"Nie znaleziono pliku binarnego dla: {hdr_path}")

print(f"[INFO] Wczytano plik binarny: {os.path.basename(img_path)}")
img = envi.open(hdr_path, img_path)
meta = img.metadata

# długości fal
try:
    wls = np.array([float(w) for w in meta["wavelength"]], dtype=np.float32)
except KeyError:
    raise RuntimeError("Brak informacji o długościach fal w nagłówku ENVI.")

if np.nanmean(wls) > 10:
    wls *= 1e-3  # nm → µm

print(f"[INFO] Rozmiar sceny: {img.nrows} × {img.ncols} × {img.nbands}")
print(f"[INFO] Zakres długości fal: {wls.min():.3f}–{wls.max():.3f} µm")

# ANALIZA PASM
for b in bands_to_check:
    if b >= img.nbands:
        print(f"[WARN] Pasmo {b} poza zakresem ({img.nbands}) — pomijam.")
        continue

    band = img.read_band(b).astype(np.float32)
    valid_mask = band > 0
    if not np.any(valid_mask):
        print(f"[WARN] Pasmo {b} zawiera same zera — pomijam.")
        continue

    band_valid = band[valid_mask]
    min_val, max_val, mean_val = band_valid.min(), band_valid.max(), band_valid.mean()
    eps = 1e-4
    frac_clipped = np.mean((band_valid <= eps) | (band_valid >= 1.2))

    print(f"\n=== Pasmo {b} (λ={wls[b]:.4f} µm) ===")
    print(f"Min: {min_val:.3f}, Max: {max_val:.3f}, Średnia: {mean_val:.3f}")
    print(f"% pikseli przyciętych (~0 lub >1.2): {100 * frac_clipped:.2f}%")

    # Histogram reflektancji
    plt.figure(figsize=(5, 4))
    plt.hist(band_valid, bins=80, color="steelblue", edgecolor="black", alpha=0.8)
    plt.title(f"Histogram reflektancji\npasmo {b}, λ={wls[b]:.3f} µm")
    plt.xlabel("Reflektancja")
    plt.ylabel("Liczba pikseli")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ŚREDNIE SPEKTRUM SCENY
print("\n[INFO] Liczę średnie spektrum całej sceny (bez pasm pustych)...")
mean_spectrum = np.zeros(img.nbands, dtype=np.float32)
for b in range(img.nbands):
    band = img.read_band(b).astype(np.float32)
    mask = band > 0
    mean_spectrum[b] = np.mean(band[mask]) if np.any(mask) else 0

plt.figure(figsize=(8, 5))
plt.plot(wls, mean_spectrum, "-o", markersize=3)
plt.title("Średnie spektrum reflektancji sceny")
plt.xlabel("Długość fali [µm]")
plt.ylabel("Średnia reflektancja")
plt.grid(True)
plt.tight_layout()
plt.show()

# NDVI
print("\n[INFO] Obliczam NDVI (sanity check)...")
red_idx = np.argmin(np.abs(wls - 0.66))
nir_idx = np.argmin(np.abs(wls - 0.87))
Rred = img.read_band(red_idx).astype(np.float32)
Rnir = img.read_band(nir_idx).astype(np.float32)
ndvi = (Rnir - Rred) / (Rnir + Rred + 1e-6)

ndvi_valid = ndvi[np.isfinite(ndvi)]
print(f"NDVI: min={ndvi_valid.min():.3f}, max={ndvi_valid.max():.3f}, mean={ndvi_valid.mean():.3f}")
