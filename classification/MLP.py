import os
import json
from time import time
from collections import Counter
import geopandas as gpd
import numpy as np
import rasterio
import torch
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import rasterize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layer_sizes,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(hidden_layer_sizes, int):
            hidden_sizes = [hidden_layer_sizes]
        else:
            hidden_sizes = list(hidden_layer_sizes)

        activation_map = {
            "relu": nn.ReLU,        # Funkcje aktywacji
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "gelu": nn.GELU,
        }
        if activation not in activation_map:
            raise ValueError(
                f"Nieobsługiwana funkcja aktywacji '{activation}'. Dostępne: {list(activation_map)}"
            )

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_map[activation]())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

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

# Trening
def train_mlp_classifier(
    pca_dir: str,
    shp_path: str,
    model_out: str,
    map_out: str,
    reflectance_bsq_path: str = None,
    max_samples_per_class: int = 4000,
    test_size: float = 0.2,
    seed: int = 42,
    skip_map: bool = False,
    **model_params,
):
    t0 = time()
    rng = np.random.default_rng(seed)

    Xpca, n_pix, n_comp, rows, cols, transform, crs, _ = _load_pca_memmap(pca_dir)
    print(f"[INFO] PCA: {n_pix} pikseli × {n_comp} komponentów; raster {rows}×{cols}")
    Xpca_img = Xpca.reshape(rows, cols, n_comp)

    # Import dataset-a
    gdf = gpd.read_file(shp_path)
    if "label_id" not in gdf.columns:
        raise ValueError("Shapefile musi zawierać kolumnę 'label_id'.")
    if gdf.crs is None:
        raise ValueError("Shapefile nie ma zdefiniowanego CRS.")
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

    print(f"[INFO] Wczytano {len(gdf)} poligonów.")
    if "class" in gdf.columns:
        print(gdf["class"].value_counts())
    else:
        print(gdf["label_id"].value_counts())

    # Rasteryzacja etykiet
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
    print(f"[INFO] Klasy w rastrze: {unique_labels}")

    X_list, y_list = [], []
    for lbl in tqdm(unique_labels, desc="Losowanie próbek"):
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

    labels_order = sorted(np.unique(y))
    label_to_index = {int(lbl): idx for idx, lbl in enumerate(labels_order)}
    labels_array = np.array(labels_order, dtype=np.int32)
    y_indices = np.array([label_to_index[int(lbl)] for lbl in y], dtype=np.int64)

    (
        X_train,
        X_test,
        y_train_idx,
        y_test_idx,
        y_train_labels,
        y_test_labels,
    ) = train_test_split(
        X,
        y_indices,
        y,
        test_size=test_size,
        stratify=y_indices,
        random_state=seed,
    )
    print(f"[INFO] Podział danych: {len(y_train_idx)} train / {len(y_test_idx)} test")

    per_class_counts_test = {int(k): int(v) for k, v in Counter(y_test_labels.tolist()).items()}

    torch.manual_seed(seed)
    np.random.seed(seed)
    # Parametry MLP
    hidden_layer_sizes = model_params.pop("hidden_layer_sizes", (128, 64))
    activation = model_params.pop("activation", "relu")
    dropout = model_params.pop("dropout", 0.2)
    epochs = model_params.pop("epochs", model_params.pop("max_iter", 100))
    batch_size = model_params.pop("batch_size", 512)
    learning_rate = model_params.pop("learning_rate", 1e-3)
    weight_decay = model_params.pop("weight_decay", model_params.pop("alpha", 0.0))
    validation_fraction = model_params.pop("validation_fraction", 0.1)
    early_stopping = model_params.pop("early_stopping", False)
    patience = model_params.pop("patience", 10)

    if model_params:
        raise ValueError(f"Nieobsługiwane parametry MLP: {sorted(model_params.keys())}")

    device = _get_device()
    training_config = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "validation_fraction": validation_fraction,
        "early_stopping": early_stopping,
        "patience": patience,
        "device": str(device),
    }
    print("[INFO] Trenuję MLP (PyTorch) z parametrami:", training_config)

    scaler_mean = X_train.mean(axis=0)
    scaler_std = X_train.std(axis=0)
    scaler_std[scaler_std == 0] = 1.0
    X_train_norm = (X_train - scaler_mean) / scaler_std
    if validation_fraction and validation_fraction > 0.0:
        (
            X_train_norm,
            X_val_norm,
            y_train_idx,
            y_val_idx,
            X_train_raw,
            _,
            y_train_labels,
            _,
        ) = train_test_split(
            X_train_norm,
            y_train_idx,
            X_train,
            y_train_labels,
            test_size=validation_fraction,
            stratify=y_train_idx,
            random_state=seed,
        )
    else:
        X_val_norm, y_val_idx, _ = None, None, None
        X_train_raw = X_train

    per_class_counts_train = {int(k): int(v) for k, v in Counter(y_train_labels.tolist()).items()}

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_norm.astype(np.float32)),
        torch.from_numpy(y_train_idx.astype(np.int64)),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    if X_val_norm is not None:
        val_dataset = TensorDataset(
            torch.from_numpy(X_val_norm.astype(np.float32)),
            torch.from_numpy(y_val_idx.astype(np.int64)),
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    input_dim = X_train_norm.shape[1]
    num_classes = len(labels_order)
    model = _MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_state = None
    best_metric = float("inf")
    best_epoch = 0
    epochs_without_improve = 0
    epochs_run = 0

    # Trening
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == yb).sum().item()
            total_train += yb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train if total_train > 0 else 0
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        epochs_run = epoch
        val_loss, val_acc = None, None

        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            correct_val, total_val = 0, 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                    total_val_loss += loss.item() * xb.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct_val += (preds == yb).sum().item()
                    total_val += yb.size(0)

            val_loss = total_val_loss / len(val_loader.dataset)
            val_acc = correct_val / total_val if total_val > 0 else 0
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Early stopping
            if val_loss < best_metric:
                best_metric = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if early_stopping and epochs_without_improve >= patience:
                print(f"[INFO] Zatrzymanie wczesne po {epoch} epokach (brak poprawy {patience} epok).")
                break
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        # Logowanie postępu
        if epoch % 10 == 0 or epoch == 1:
            log_msg = f"[EPOCH {epoch}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
            if val_loss is not None:
                log_msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            print(log_msg)

    # Koniec treningu
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epochs_run

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    fit_s = time() - t0

    # Funkcja pomocnicza do predykcji
    def _predict_array(array: np.ndarray) -> np.ndarray:
        array_norm = (array - scaler_mean) / scaler_std
        dataset = TensorDataset(torch.from_numpy(array_norm.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device)
                logits = model(xb)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        if not preds:
            return np.array([], dtype=np.int32)
        preds_idx = np.concatenate(preds, axis=0)
        return labels_array[preds_idx]

    # Predykcje na zbiorze treningowym i testowym
    ytr_pred = _predict_array(X_train_raw)
    yte_pred = _predict_array(X_test)

    # Ewaluacja
    metrics = {
        "name": "MLP",
        "params": training_config,
        "class_map": class_map,
        "train": {
            "n_samples": int(len(y_train_labels)),
            "accuracy": float(accuracy_score(y_train_labels, ytr_pred)),
            "report_dict": classification_report(y_train_labels, ytr_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_train_labels, ytr_pred, labels=labels_order),
            "labels": list(labels_order),
            "per_class_counts": per_class_counts_train,
        },
        "test": {
            "n_samples": int(len(y_test_labels)),
            "accuracy": float(accuracy_score(y_test_labels, yte_pred)),
            "report_dict": classification_report(y_test_labels, yte_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test_labels, yte_pred, labels=labels_order),
            "labels": list(labels_order),
            "per_class_counts": per_class_counts_test,
        },
        "timings": {"fit_s": fit_s},
        "artifacts": {"model_path": model_out, "map_path": map_out},
        # 🔹 Dodajemy dane do raportu PDF
        "training_curves": {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_acc": train_accuracies,
            "val_acc": val_accuracies,
        },
    }

    # Zapis modelu
    artifact = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "scaler": {"mean": scaler_mean, "std": scaler_std},
        "labels": list(labels_order),
        "config": {**training_config, "epochs_trained": epochs_run, "best_epoch": best_epoch},
    }
    torch.save(artifact, model_out)
    print(f"[DONE] Model zapisano: {model_out}")

    if skip_map:
        print("[INFO] Pomijam generowanie mapy klasyfikacji (tryb ewaluacji).")
        return model, metrics

    print("[INFO] Generuję mapę klasyfikacji...")
    map_batch_size = 20000
    preds = np.zeros((rows * cols,), dtype=np.int32)

    # Maska ważnych pikseli
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

    # Predykcja tylko dla pikseli ważnych
    preds[:] = 0
    total_batches = range(0, rows * cols, map_batch_size)
    for start in tqdm(total_batches, desc="Predykcja batchami"):
        end = min(start + map_batch_size, rows * cols)
        batch_mask = valid_mask[start:end]
        if not np.any(batch_mask):
            continue

        batch = Xpca[start:end, :][batch_mask]
        batch_norm = (batch - scaler_mean) / scaler_std
        batch_tensor = torch.from_numpy(batch_norm.astype(np.float32)).to(device)

        with torch.no_grad():
            logits = model(batch_tensor)
            preds_idx = torch.argmax(logits, dim=1).cpu().numpy()
            preds_chunk = labels_array[preds_idx]

        preds[start:end][batch_mask] = preds_chunk
        preds[start:end][~batch_mask] = 0  # poza maską = tło (nie klasyfikowane)

    preds_img = preds.reshape(rows, cols).astype("float32")

    preds_img[~valid_mask.reshape(rows, cols)] = np.nan

    # Usuwanie klasę 0 z metryk
    metrics["test"]["labels"] = [lbl for lbl in metrics["test"]["labels"] if lbl != 0]

    # Ustawianie nodata = 0
    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": 0.0
    }

    with rasterio.open(map_out, "w", **profile) as dst:
        dst.write(preds_img, 1)

    print(f"[DONE] Mapa klasyfikacji zapisana: {map_out}")
    return model, metrics

