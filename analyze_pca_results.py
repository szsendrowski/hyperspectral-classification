import os
import json
import numpy as np
import matplotlib.pyplot as plt
from spectral import envi
from sklearn.decomposition import PCA


# Input
BASE_DIR = r"C:\Users\sciezka\do\folderu\GitHub\hyperspectral-classification"

RFL_HDR = os.path.join(BASE_DIR, "ang20160917t203013_rfl", "ang20160917t203013_rdn_v1n2_img_bsq_rfl.hdr")
PCA_DIR = os.path.join(BASE_DIR, "ang20160917t203013_pca")
PCA_PATH = os.path.join(PCA_DIR, "Xpca.dat")
META_PATH = PCA_PATH + ".json"


# WCZYTANIE PCA (memmap + metadane)

if not (os.path.exists(PCA_PATH) and os.path.exists(META_PATH)):
    raise FileNotFoundError("Nie znaleziono plików PCA (.dat i .json)!")

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

shape = tuple(meta["shape"])
dtype = np.dtype(meta["dtype"])
print(f"[INFO] Wczytano meta: shape={shape}, dtype={dtype}")

Xpca = np.memmap(PCA_PATH, dtype=dtype, mode="r", shape=shape)
n_samples = min(50000, shape[0])
idx = np.random.choice(shape[0], size=n_samples, replace=False)
X_sample = Xpca[idx]

print(f"[INFO] Wczytano próbkę {X_sample.shape[0]} pikseli z {shape[1]} komponentami PCA.")


# Wykres wariancji wyjaśnionej

pca_check = PCA()
pca_check.fit(X_sample)
explained = pca_check.explained_variance_ratio_
cum_explained = np.cumsum(explained)

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, len(explained) + 1), cum_explained, marker='o')
plt.title("Kumulatywna wariancja wyjaśniona przez PCA")
plt.xlabel("Liczba komponentów głównych")
plt.ylabel("Skumulowana wariancja")
plt.grid(True)
plt.tight_layout()
plt.show()


# Histogramy pierwszych komponentów PCA

plt.figure(figsize=(10, 6))
for i in range(min(5, shape[1])):
    plt.hist(X_sample[:, i], bins=60, alpha=0.6, label=f"PC{i+1}")
plt.title("Histogramy wartości pierwszych komponentów PCA")
plt.xlabel("Wartość komponentu")
plt.ylabel("Liczba pikseli")
plt.legend()
plt.tight_layout()
plt.show()


#  Macierz korelacji między komponentami
corr = np.corrcoef(X_sample.T)
plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Macierz korelacji między komponentami PCA")
plt.colorbar(label="Korelacja")
plt.tight_layout()
plt.show()


# Porównanie PCA vs reflektancja (fragment sceny)
if not os.path.exists(RFL_HDR):
    raise FileNotFoundError("Nie znaleziono pliku reflektancji (.hdr)!")

rfl = envi.open(RFL_HDR)
rows, cols = rfl.nrows, rfl.ncols

# reshape PCA do wymiarów sceny
pca_img = np.reshape(Xpca, (rows, cols, shape[1]))

row_offset = -900   # ujemne w dół
col_offset = 400    # ujemne w lewo
# wycięcie środka sceny
crop = 512
r1 = rows // 2 - crop // 2 + row_offset
r2 = r1 + crop
c1 = cols // 2 - crop // 2 + col_offset
c2 = c1 + crop

# reflektancja RGB (trzy przykładowe pasma)
rfl_bands = [54, 36, 20]
rfl_crop = rfl.read_subregion((r1, r2), (c1, c2))[:, :, rfl_bands]
pca_crop = pca_img[r1:r2, c1:c2, :3]  # PC1, PC2, PC3

# normalizacja z maską
def normalize_with_mask(img):
    mask = np.any(img > 0, axis=-1)
    img_norm = np.zeros_like(img, dtype=float)
    for i in range(img.shape[-1]):
        band = img[..., i]
        valid = band[mask]
        if valid.size > 0:
            bmin, bmax = np.percentile(valid, [1, 99])
            band = np.clip((band - bmin) / (bmax - bmin), 0, 1)
        img_norm[..., i] = band
    img_norm[~mask] = np.nan
    return img_norm, mask

rfl_norm, mask_rfl = normalize_with_mask(rfl_crop)
pca_norm, mask_pca = normalize_with_mask(pca_crop)

# wyświetlenie porównania
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(rfl_norm)
plt.title("Reflektancja (środek sceny)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pca_norm)
plt.title("PCA (PC1–PC3, środek sceny)")
plt.axis("off")

plt.tight_layout()
plt.show()


# 6️⃣ Korelacja między RFL a PCA (kanały)
common_mask = mask_rfl & mask_pca
rfl_flat = rfl_norm[..., 0][common_mask].ravel()
pca_flat = pca_norm[..., 0][common_mask].ravel()
corr_value = np.corrcoef(rfl_flat, pca_flat)[0, 1]

print(f"[INFO] Korelacja (RFL Band {rfl_bands[0]} vs PCA1) = {corr_value:.4f}")
print("[DONE] Analiza PCA + Reflektancja zakończona pomyślnie")
