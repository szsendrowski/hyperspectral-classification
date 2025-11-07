import os
import json
import numpy as np
from tqdm import tqdm
import spectral.io.envi as envi
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE



# 1. Wczytywanie AVIRIS — strumieniowo
def load_aviris_cube(scene_dir, hdr_name, drop_absorption=True, subset=None):
    # Otwiera bez ładowania danych do pamieci RAM
    hdr_path = os.path.join(scene_dir, hdr_name)
    img_path = hdr_path.replace(".hdr", "")
    if not os.path.exists(img_path):
        for ext in [".img", ".bil", ".bsq"]:
            if os.path.exists(hdr_path.replace(".hdr", ext)):
                img_path = hdr_path.replace(".hdr", ext)
                break

    img = envi.open(hdr_path, img_path)
    md = img.metadata
    wavelengths = np.array([float(w) for w in md["wavelength"]], dtype=np.float32)
    if np.nanmean(wavelengths) > 10:
        wavelengths *= 1e-3  # nm → µm

    # Usuwanie pasm absorbcyjnych
    if drop_absorption:
        wl = wavelengths
        bad = ((wl >= 0.759) & (wl <= 0.770)) | \
              ((wl >= 1.34) & (wl <= 1.45)) | \
              ((wl >= 1.80) & (wl <= 1.95))
        good = ~bad
    else:
        good = np.ones_like(wavelengths, dtype=bool)

    R, C, B = img.shape
    if subset is not None:
        R = min(R, subset[0])
        C = min(C, subset[1])
        print(f"[INFO] Używam wycinka {R}x{C} pikseli.")

    return img, wavelengths[good], good, (R, C, B)



# 2. Standaryzacja kanałów do średniej 0 i odchylenia 1
def zscore_streaming(img, good_mask, chunk_rows=64):

    nrows, ncols, nbands = img.shape
    nbands_used = np.sum(good_mask)
    scaler = StandardScaler(with_mean=True, with_std=True)

    # 1. Wektorowa aktualizacja globalnych średnich i wariancji dla pasm
    print("[INFO] Faza 1: Standaryzacja do mean = 0 i std = 1")
    for row_start in tqdm(range(0, nrows, chunk_rows), desc="ZScore fit", unit="block"):
        row_end = min(row_start + chunk_rows, nrows)
        chunk = img.read_subregion((row_start, row_end), (0, ncols))
        X = chunk.reshape(-1, nbands)[:, good_mask]
        scaler.partial_fit(X)

    print("[INFO] Zakończono standaryzacje.")
    return scaler

# Zapis wyników
def _open_memmap(path, shape, dtype=np.float32, mode="w+"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)

def _save_sidecar_json(path, shape, dtype_str):
    meta = {"shape": list(shape), "dtype": dtype_str}
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)



# 3. PCA
def reduce_pca_streaming(img, good_mask, scaler,
                         n_components=32,        # liczba komponentów z góry określona
                         batch_rows=64,
                         out_dir=None,
                         out_basename="Xpca"):


    nrows, ncols, nbands = img.shape
    nbands_used = int(np.sum(good_mask))

    print(f"[INFO] Rozpoczynam IncrementalPCA (liczba komponentów = {n_components})...")
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_rows * ncols)

    # Normalizacja oraz uczenie PCA
    for row_start in tqdm(range(0, nrows, batch_rows), desc="PCA fit", unit="block"):
        row_end = min(row_start + batch_rows, nrows)
        chunk = img.read_subregion((row_start, row_end), (0, ncols))
        X = chunk.reshape(-1, nbands)[:, good_mask]
        Xs = scaler.transform(X)
        ipca.partial_fit(Xs)

    print(f"[INFO] PCA dopasowane. Zachowuję {n_components} komponentów.")

    # Tworzenie MemoryMap (wiersze, kolumny, komponenty)
    if out_dir is None:
        out_dir = os.getcwd()
    pca_path = os.path.join(out_dir, f"{out_basename}.dat")

    total_px = nrows * ncols
    mm = _open_memmap(pca_path, shape=(total_px, n_components), dtype=np.float32, mode="w+")

    print("[INFO] Transformacja danych PCA (zapis strumieniowy)...")
    write_ptr = 0
    for row_start in tqdm(range(0, nrows, batch_rows), desc="PCA transform", unit="block"):
        row_end = min(row_start + batch_rows, nrows)
        chunk = img.read_subregion((row_start, row_end), (0, ncols))
        X = chunk.reshape(-1, nbands)[:, good_mask]
        Xs = scaler.transform(X)
        Xp = ipca.transform(Xs).astype(np.float32)
        mm[write_ptr:write_ptr + Xp.shape[0], :] = Xp
        write_ptr += Xp.shape[0]

    mm.flush(); del mm
    _save_sidecar_json(pca_path, (total_px, n_components), "float32")

    print(f"[DONE] Zapisano PCA do pliku: {pca_path}")
    return pca_path, (total_px, n_components), ipca, n_components



# 4. Selekcja pasm (opcjonalne)
def rank_features_rf(X, y):
    rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced")
    rf.fit(X, y)
    imp = rf.feature_importances_
    return np.argsort(-imp)


def rank_features_svm(X, y, n_select=1):
    svc = LinearSVC(dual=False, max_iter=5000, class_weight="balanced")
    rfe = RFE(svc, n_features_to_select=n_select, step=0.1)
    rfe.fit(X, y)
    return np.argsort(rfe.ranking_)


def stratified_sample(X, y, n=100000):
    if len(y) <= n:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n / len(y), random_state=0)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]



# 5. Główna funkcja redukcji wymiarów
def reduce_hsi(scene_dir, hdr_name, labels=None,
               n_pca=32, n_top_bands=64, subset=None,
               out_pca_dir=None, out_pca_name="Xpca"):

    # Redukcja wymiarowości do określonej liczby komponentów PCA.

    img, wavelengths, good_mask, shape = load_aviris_cube(scene_dir, hdr_name, subset=subset)
    print(f"[INFO] Scena: {shape[0]}x{shape[1]}x{shape[2]}, pasm użytecznych: {int(np.sum(good_mask))}")

    scaler = zscore_streaming(img, good_mask)

    if out_pca_dir is None:
        out_pca_dir = os.path.join(scene_dir, "pca_output")
    os.makedirs(out_pca_dir, exist_ok=True)

    pca_path, pca_shape, pca_model, k = reduce_pca_streaming(
        img, good_mask, scaler,
        n_components=n_pca, # ograniczamy do zdefiniowanej liczby
        out_dir=out_pca_dir,
        out_basename=out_pca_name
    )

    result = {
        "X_pca_path": pca_path,
        "X_pca_shape": pca_shape,
        "wavelengths": wavelengths,
        "pca_k": k,
        "shape_rc": (shape[0], shape[1]),
        "scaler": scaler,
        "pca_model": pca_model,
    }

    # opcjonalne automatyczne dobieranie najwazniejszych komponentów
    if labels is not None:
        n_pix = pca_shape[0]
        labels = np.asarray(labels)
        if labels.shape[0] != n_pix:
            print("[WARN] Długość etykiet nie zgadza się z liczbą pikseli po spłaszczaniu – pomijam supervised ranking.")
        else:
            print("[INFO] Supervised ranking (RF) na próbce z memmap)...")
            Xmm = np.memmap(pca_path, dtype=np.float32, mode="r", shape=pca_shape)
            take = min(80000, n_pix)
            rng = np.random.default_rng(0)
            idx = rng.choice(n_pix, size=take, replace=False)
            X_sub = Xmm[idx]
            y_sub = labels[idx]
            del Xmm

            rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced")
            rf.fit(X_sub, y_sub)
            imp = rf.feature_importances_
            rf_idx = np.argsort(-imp)[:n_top_bands]

            result["rf_idx"] = rf_idx
            result["rf_importances"] = imp[rf_idx]

    print(f"[DONE] Redukcja PCA zakończona. Zapisano {n_pca} komponentów.")
    return result

