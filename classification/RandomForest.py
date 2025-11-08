import os
import json
import numpy as np
import geopandas as gpd
import rasterio
from time import time
from affine import Affine
from collections import Counter
from rasterio.crs import CRS
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump
from tqdm import tqdm

# Wczytywanie pliku PCA z georeferencja
def _load_pca_memmap(pca_dir: str):

    dat_files = [f for f in os.listdir(pca_dir) if f.startswith("Xpca_") and f.endswith(".dat")]
    if not dat_files:
        raise FileNotFoundError(f"Nie znaleziono pliku Xpca_*.dat w: {pca_dir}")
    pca_path = os.path.join(pca_dir, dat_files[0])
    meta_path = pca_path + ".json"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n_pix, n_comp = meta["shape"]
    rows, cols = meta["shape_rc"]

    tr = meta["transform"]
    if isinstance(tr, (list, tuple)):
        transform = Affine(*tr)
    elif isinstance(tr, dict):
        transform = Affine(tr["a"], tr["b"], tr["c"], tr["d"], tr["e"], tr["f"])
    else:
        raise ValueError("Niepoprawny format transform w JSON.")

    crs = CRS.from_string(meta["crs"])
    Xpca = np.memmap(pca_path, dtype=np.float32, mode="r", shape=(n_pix, n_comp))
    return Xpca, n_pix, n_comp, rows, cols, transform, crs, pca_path


def train_random_forest_spatial(
    pca_dir: str,
    shp_path: str,
    model_out: str,
    map_out: str = None,
    reflectance_bsq_path: str = None,
    max_samples_per_class: int = 5000,
    test_size: float = 0.2,
    seed: int = 42,
    random_state=42,
    skip_map: bool = False,
    **model_params  # 👈 zmiana: zamiast model_params=None
):

    t0 = time()
    rng = np.random.default_rng(seed)

    # Wczytywanie PCA i georeferencję
    Xpca, n_pix, n_comp, rows, cols, transform, crs, pca_path = _load_pca_memmap(pca_dir)
    print(f"[INFO] PCA: {n_pix} pikseli × {n_comp} komponentów; raster {rows}×{cols}")
    Xpca_img = Xpca.reshape(rows, cols, n_comp)

    # Wczytanie dateset-a
    gdf = gpd.read_file(shp_path)
    if "label_id" not in gdf.columns:
        raise ValueError("Shapefile musi zawierać kolumnę 'label_id'.")
    if gdf.crs is None:
        raise ValueError("Shapefile nie ma CRS — ustaw go w QGIS.")
    gdf = gdf.to_crs(crs)

    if "class" in gdf.columns:
        class_map = (
            gdf[["label_id", "class"]]
            .dropna()
            .drop_duplicates(subset=["label_id"])
            .set_index("label_id")["class"]
            .to_dict()
        )
    else:
        unique_ids = sorted(gdf["label_id"].unique())
        class_map = {int(i): str(int(i)) for i in unique_ids}

    # Rasteryzacja poligonów
    print("[INFO] Rasteryzuję poligony do rastra etykiet...")
    shapes = [(geom, int(lbl)) for geom, lbl in zip(gdf.geometry, gdf["label_id"])]
    label_raster = rasterize(
        shapes=shapes,
        out_shape=(rows, cols),
        transform=transform,
        fill=0,
        dtype="int16",
        all_touched=False,
    )
    unique_labels = sorted([int(l) for l in np.unique(label_raster) if l != 0])
    print(f"[INFO] Klasy w rastrze etykiet: {unique_labels}")

    # Losowanie próbek
    X_list, y_list = [], []
    print("[INFO] Losuję próbki z rastra etykiet...")
    for lbl in tqdm(unique_labels):
        mask = label_raster == lbl
        n_avail = int(mask.sum())
        if n_avail == 0:
            continue
        n_take = min(max_samples_per_class, n_avail)
        idx_lin = rng.choice(n_avail, size=n_take, replace=False)
        rows_idx, cols_idx = np.where(mask)
        feats = Xpca_img[rows_idx[idx_lin], cols_idx[idx_lin], :]
        X_list.append(feats)
        y_list.append(np.full(n_take, lbl, dtype=np.int32))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    print(f"[INFO] Zebrano {len(y)} próbek (X: {X.shape})")

    # Podział train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    print(f"[INFO] Podział danych: {len(y_train)} train / {len(y_test)} test")

    labels_order = sorted(np.unique(y))
    per_class_counts_train = {int(k): int(v) for k, v in Counter(y_train.tolist()).items()}
    per_class_counts_test = {int(k): int(v) for k, v in Counter(y_test.tolist()).items()}

    # Trening RF
    print("[INFO] Trenuję RandomForest...")
    rf = RandomForestClassifier(
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
        **model_params
    )
    rf.fit(X_train, y_train)
    fit_s = time() - t0

    # Ewaluacja
    ytr_pred = rf.predict(X_train)
    yte_pred = rf.predict(X_test)

    metrics = {
        "name": "RandomForest",
        "params": model_params,
        "class_map": class_map,
        "train": {
            "n_samples": int(len(y_train)),
            "accuracy": float(accuracy_score(y_train, ytr_pred)),
            "report_dict": classification_report(y_train, ytr_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_train, ytr_pred, labels=labels_order),
            "labels": list(labels_order),
            "per_class_counts": per_class_counts_train,
        },
        "test": {
            "n_samples": int(len(y_test)),
            "accuracy": float(accuracy_score(y_test, yte_pred)),
            "report_dict": classification_report(y_test, yte_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, yte_pred, labels=labels_order),
            "labels": list(labels_order),
            "per_class_counts": per_class_counts_test,
        },
        "timings": {"fit_s": fit_s},
        "artifacts": {"model_path": model_out, "map_path": map_out},
    }

    dump(rf, model_out)
    print(f"[DONE] Model zapisano: {model_out}")

    if skip_map:
        print("[INFO] Pomijam generowanie mapy klasyfikacji (tryb ewaluacji).")
        return rf, metrics

    # Generowanie mapy klasyfikacji
    if map_out:
        print("[INFO] Generuję mapę klasyfikacji...")
        batch_size = 50000
        preds = np.zeros((rows * cols,), dtype=np.int16)

        valid_mask = ~np.isnan(Xpca).any(axis=1)
        valid_mask &= ~(Xpca.sum(axis=1) == 0)

        if reflectance_bsq_path and os.path.exists(reflectance_bsq_path):
            print("[INFO] Wczytuję maskę z reflektancji (tło)...")
            with rasterio.open(reflectance_bsq_path) as src:
                rfl_band = src.read(1, masked=True)
                scene_mask = (~rfl_band.mask) & (rfl_band > 0)
                valid_mask &= scene_mask.flatten()
        else:
            print("[WARN] Brak reflektancji — maska sceny pominięta.")

        print(f"[INFO] Liczba pikseli w scenie: {rows*cols}, ważnych: {valid_mask.sum()}")

        for start in tqdm(range(0, rows * cols, batch_size), desc="Predykcja batchami"):
            end = min(start + batch_size, rows * cols)
            batch_mask = valid_mask[start:end]
            if not np.any(batch_mask):
                continue
            batch = Xpca[start:end, :][batch_mask]
            preds_chunk = rf.predict(batch)
            preds[start:end][batch_mask] = preds_chunk
            preds[start:end][~batch_mask] = 0

        preds_img = preds.reshape(rows, cols)

        profile = {
            "driver": "GTiff",
            "height": rows,
            "width": cols,
            "count": 1,
            "dtype": "int16",
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
            "nodata": 0,
        }
        with rasterio.open(map_out, "w", **profile) as dst:
            dst.write(preds_img, 1)
        print(f"[DONE] Mapa klasyfikacji zapisana: {map_out}")

    return rf, metrics

