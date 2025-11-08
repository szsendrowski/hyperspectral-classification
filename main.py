import os
import json
from datetime import datetime
from preprocessing.RadiometricCorrection import run_atm_corr
from preprocessing.DimensionReduction import reduce_hsi
from preprocessing.Georeference import add_georeference_to_pca
from classification.RandomForest import train_random_forest_spatial
from classification.kNN import train_knn_classifier
from classification.SVM import train_svm_spatial
from classification.MLP import train_mlp_classifier
from analyze.TrainingReport import save_training_report


# Sciezka do glownego katalogu
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
CLASSIFY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "classification"))
ANALYZE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "analyze"))

# NAZWA SCENY
SCENE_NAME = "ang20160917t203013"
SCENE_DIR = os.path.join(BASE_DIR, f"{SCENE_NAME}_rdn_v1n2")
OUT_RFL_DIR = os.path.join(BASE_DIR, f"{SCENE_NAME}_rfl")
OUT_PCA_DIR = os.path.join(BASE_DIR, f"{SCENE_NAME}_pca")

GROUND_TRUTH_SHP = os.path.join(BASE_DIR, "ground_truth.shp")

# Reflektancja
RDN_HDR = f"{SCENE_NAME}_rdn_v1n2_img_bsq.hdr"
REFLECTANCE_HDR = os.path.join(OUT_RFL_DIR, f"{SCENE_NAME}_rdn_v1n2_img_bsq_rfl.hdr")
REFLECTANCE_BSQ = os.path.join(OUT_RFL_DIR, f"{SCENE_NAME}_rdn_v1n2_img_bsq_rfl.bsq")

# PCA
PCA_DAT = os.path.join(OUT_PCA_DIR, "Xpca_32.dat")
PCA_JSON = PCA_DAT + ".json"

# Modele i mapy
RF_MODEL = os.path.join(CLASSIFY_DIR, "models", "random_forest_model.joblib")
KNN_MODEL = os.path.join(CLASSIFY_DIR, "models", "knn_model.joblib")
SVM_MODEL = os.path.join(CLASSIFY_DIR, "models", "svm_model.joblib")
MLP_MODEL = os.path.join(CLASSIFY_DIR, "models", "mlp_model.joblib")

RF_MAP = os.path.join(CLASSIFY_DIR, "maps", "classification_map_RF.tif")
KNN_MAP = os.path.join(CLASSIFY_DIR, "maps", "classification_map_kNN.tif")
SVM_MAP = os.path.join(CLASSIFY_DIR, "maps", "classification_map_SVM.tif")
MLP_MAP = os.path.join(CLASSIFY_DIR, "maps", "classification_map_MLP.tif")

# Raport
REPORT_PDF = os.path.join(ANALYZE_DIR, "training_report.pdf")

# Parametry
LAT, LON, TIMEZONE_OFFSET = 34.035, -117.5, -8.0
COMPONENT_NUMBER = 32
MAX_SAMPLES = 2000
TEST_SIZE = 0.2

# Parametry modeli
MODEL_PARAMS = {
    "RandomForest": {
        "n_estimators": 500,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    "kNN": {
        "n_neighbors": 5,
        "weights": "distance",
        "metric": "minkowski",
        "p": 2,
    },

    "SVM": {
        "kernel": "rbf",
        "C": 10.0,
        "gamma": "scale",
    },
    "MLP": {
        "hidden_layer_sizes": (256, 128, 64),
        "activation": "relu",
        "dropout": 0.2,
        "epochs": 150,
        "batch_size": 512,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "patience": 20,
    },
}

os.makedirs(OUT_RFL_DIR, exist_ok=True)
os.makedirs(OUT_PCA_DIR, exist_ok=True)


def find_reflectance_file(folder):
    for file in os.listdir(folder):
        if file.endswith(("_rfl.hdr", "_rfl_simple.hdr")):
            return os.path.join(folder, file)
    return None


def find_pca_files(folder, component_number):
    basename = f"Xpca_{component_number}"
    dat_path = os.path.join(folder, f"{basename}.dat")
    json_path = dat_path + ".json"
    if os.path.exists(dat_path) and os.path.exists(json_path):
        return dat_path
    return None


# GŁÓWNA FUNKCJA
if __name__ == "__main__":

    print("=== [1/3] Korekcja atmosferyczna AVIRIS-NG ===")

    existing_rfl_hdr = find_reflectance_file(OUT_RFL_DIR)
    if existing_rfl_hdr:
        print(f"[INFO] Znaleziono istniejący plik reflektancji: {existing_rfl_hdr}")
        print("[INFO] Pomijam korekcję atmosferyczną.")
        rfl_hdr_path = existing_rfl_hdr
    else:
        print("[INFO] Nie znaleziono pliku reflektancji — rozpoczynam korekcję.")
        rfl_hdr_path = run_atm_corr(
            scene_dir=SCENE_DIR,
            rdn_hdr=RDN_HDR,
            lat=LAT,
            lon=LON,
            tz_offset=TIMEZONE_OFFSET,
            out_dir=OUT_RFL_DIR
        )
        print(f"[DONE] Reflektancja zapisana: {rfl_hdr_path}")

    print("\n=== [2/3] Redukcja PCA reflektancji (IncrementalPCA) ===")

    existing_pca = find_pca_files(OUT_PCA_DIR, COMPONENT_NUMBER)
    if existing_pca:
        print(f"[INFO] Znaleziono istniejący wynik PCA: {existing_pca}")
        print("[INFO] Pomijam etap redukcji PCA.")
    else:
        print(f"[INFO] Nie znaleziono PCA ({COMPONENT_NUMBER} komponentów) — rozpoczynam obliczenia.")
        pca_result = reduce_hsi(
            scene_dir=OUT_RFL_DIR,
            hdr_name=os.path.basename(rfl_hdr_path),
            n_pca=COMPONENT_NUMBER,
            out_pca_dir=OUT_PCA_DIR,
            out_pca_name=f"Xpca_{COMPONENT_NUMBER}"
        )
        print("\n=== [INFO] Podsumowanie PCA ===")
        print(f" Plik PCA: {pca_result['X_pca_path']}")
        print(f" Kształt: {pca_result['X_pca_shape']}")
        print(f" Liczba komponentów: {pca_result['pca_k']}")
        print(f" Wymiary rastra: {pca_result['shape_rc']}")

    # Ustawianie georeferencji PCA
    add_georeference_to_pca(
        pca_json_path=PCA_JSON,
        reflectance_hdr_path=REFLECTANCE_HDR
    )

    # Trenowanie / ewaluacja modeli
    all_metrics = []

    # Random Forest
    print("\n=== Klasyfikacja gruntów (Random Forest) ===")
    if os.path.exists(RF_MODEL) and os.path.exists(RF_MAP):
        print("[INFO] Pomijam trenowanie — ewaluacja istniejącego modelu.")
        _, rf_metrics = train_random_forest_spatial(
            pca_dir=OUT_PCA_DIR,
            shp_path=GROUND_TRUTH_SHP,
            model_out=RF_MODEL,
            map_out=RF_MAP,
            reflectance_bsq_path=REFLECTANCE_BSQ,
            skip_map=True,
            max_samples_per_class=MAX_SAMPLES,
            test_size=TEST_SIZE,
            **MODEL_PARAMS["RandomForest"]
        )
    else:
        rf_model, rf_metrics = train_random_forest_spatial(
            pca_dir=OUT_PCA_DIR,
            shp_path=GROUND_TRUTH_SHP,
            model_out=RF_MODEL,
            map_out=RF_MAP,
            reflectance_bsq_path=REFLECTANCE_BSQ,
            max_samples_per_class=MAX_SAMPLES,
            test_size=TEST_SIZE,
            **MODEL_PARAMS["RandomForest"]
        )
    all_metrics.append(rf_metrics)

    # kNN
    print("\n=== Klasyfikacja gruntów (kNN) ===")
    if os.path.exists(KNN_MODEL) and os.path.exists(KNN_MAP):
        print("[INFO] Pomijam trenowanie — ewaluacja istniejącego modelu.")
        _, knn_metrics = train_knn_classifier(
            pca_dir=OUT_PCA_DIR,
            shp_path=GROUND_TRUTH_SHP,
            model_out=KNN_MODEL,
            map_out=KNN_MAP,
            reflectance_bsq_path=REFLECTANCE_BSQ,
            skip_map=True,
            max_samples_per_class=MAX_SAMPLES,
            test_size=TEST_SIZE,
            **MODEL_PARAMS["kNN"]
        )
    else:
        knn_model, knn_metrics = train_knn_classifier(
            pca_dir=OUT_PCA_DIR,
            shp_path=GROUND_TRUTH_SHP,
            model_out=KNN_MODEL,
            map_out=KNN_MAP,
            reflectance_bsq_path=REFLECTANCE_BSQ,
            max_samples_per_class=MAX_SAMPLES,
            test_size=TEST_SIZE,
            **MODEL_PARAMS["kNN"]
        )
    all_metrics.append(knn_metrics)

    # SVM
    print("\n=== Klasyfikacja gruntów (SVM) ===")
    if os.path.exists(SVM_MODEL) and os.path.exists(SVM_MAP):
        print("[INFO] Pomijam trenowanie — ewaluacja istniejącego modelu.")
        _, svm_metrics = train_svm_spatial(
            pca_dir=OUT_PCA_DIR,
            shp_path=GROUND_TRUTH_SHP,
            model_out=SVM_MODEL,
            map_out=SVM_MAP,
            reflectance_bsq_path=REFLECTANCE_BSQ,
            skip_map=True,
            max_samples_per_class=MAX_SAMPLES,
            test_size=TEST_SIZE,
            **MODEL_PARAMS["SVM"]
        )
    else:
        svm_model, svm_metrics = train_svm_spatial(
            pca_dir=OUT_PCA_DIR,
            shp_path=GROUND_TRUTH_SHP,
            model_out=SVM_MODEL,
            map_out=SVM_MAP,
            reflectance_bsq_path=REFLECTANCE_BSQ,
            max_samples_per_class=MAX_SAMPLES,
            test_size=TEST_SIZE,
            **MODEL_PARAMS["SVM"]
        )
    all_metrics.append(svm_metrics)

    # MLP
    print("\n=== Klasyfikacja gruntów (MLP) ===")
    if os.path.exists(MLP_MODEL) and os.path.exists(MLP_MAP):
        print("[INFO] Pomijam trenowanie — ewaluacja istniejącego modelu.")
        _, mlp_metrics = train_mlp_classifier(
            pca_dir=OUT_PCA_DIR,
            shp_path=GROUND_TRUTH_SHP,
            model_out=MLP_MODEL,
            map_out=MLP_MAP,
            reflectance_bsq_path=REFLECTANCE_BSQ,
            skip_map=True,
            max_samples_per_class=MAX_SAMPLES,
            test_size=TEST_SIZE,
            **MODEL_PARAMS["MLP"],
        )
    else:
        mlp_model, mlp_metrics = train_mlp_classifier(
            pca_dir=OUT_PCA_DIR,
            shp_path=GROUND_TRUTH_SHP,
            model_out=MLP_MODEL,
            map_out=MLP_MAP,
            reflectance_bsq_path=REFLECTANCE_BSQ,
            max_samples_per_class=MAX_SAMPLES,
            test_size=TEST_SIZE,
            **MODEL_PARAMS["MLP"],
        )
    all_metrics.append(mlp_metrics)

    # Raport PDF
    with open(PCA_JSON, "r", encoding="utf-8") as f:
        pca_meta = json.load(f)

    context = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scene_name": SCENE_NAME,
        "pca_json": PCA_JSON,
        "pca_k": pca_meta.get("shape", ["?", "?"])[1] if isinstance(pca_meta.get("shape"), list) else "?",
        "raster_shape": pca_meta.get("shape_rc", "18127×1439"),
        "crs": pca_meta.get("crs", "EPSG:?"),
        "transform": pca_meta.get("transform", ""),
        "gt_path": GROUND_TRUTH_SHP,
    }

    save_training_report(REPORT_PDF, context, all_metrics)
    print(f"\n[DONE] Raport PDF zapisany: {REPORT_PDF}")

    print("\n=== [DONE] Pipeline ukończony (korekcja + PCA + RF + kNN + SVM + MLP+ RAPORT) ===")
