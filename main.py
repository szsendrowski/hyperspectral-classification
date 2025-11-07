import os
from RadiometricCorrection import run_atm_corr
from DimensionReduction import reduce_hsi


# Input
BASE_DIR = r"C:\Users\sciezka\do\folderu"
SCENE_DIR = os.path.join(BASE_DIR, "ang20160917t203013_rdn_v1n2")
RDN_HDR = "ang20160917t203013_rdn_v1n2_img_bsq.hdr"

# Output
OUT_RFL_DIR = os.path.join(BASE_DIR, "ang20160917t203013_rfl")
OUT_PCA_DIR = os.path.join(BASE_DIR, "ang20160917t203013_pca")

os.makedirs(OUT_RFL_DIR, exist_ok=True)
os.makedirs(OUT_PCA_DIR, exist_ok=True)

# PARAMETRY
LAT = 34.035      # środek sceny
LON = -117.5
TIMEZONE_OFFSET = -8.0   # UTC-8
COMPONENT_NUMBER = 32     # liczba komponentów PCA


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


if __name__ == "__main__":

    print("=== [1/2] Korekcja atmosferyczna AVIRIS-NG ===")

    existing_rfl_hdr = find_reflectance_file(OUT_RFL_DIR)

    if existing_rfl_hdr:
        print(f"[INFO] Znaleziono istniejący plik reflektancji: {existing_rfl_hdr}")
        print("[INFO] Pomijam korekcję atmosferyczną.")
        rfl_hdr_path = existing_rfl_hdr
    else:
        print("[INFO] Nie znaleziono pliku reflektancji — rozpoczynam korekcję atmosferyczną.")
        rfl_hdr_path = run_atm_corr(
            scene_dir=SCENE_DIR,
            rdn_hdr=RDN_HDR,
            lat=LAT,
            lon=LON,
            tz_offset=TIMEZONE_OFFSET,
            out_dir=OUT_RFL_DIR
        )
        print(f"[DONE] Reflektancja zapisana: {rfl_hdr_path}")

    print("\n=== [2/2] Redukcja PCA reflektancji (IncrementalPCA) ===")

    for k in [COMPONENT_NUMBER]:
        existing_pca = find_pca_files(OUT_PCA_DIR, k)

        if existing_pca:
            print(f"[INFO] Znaleziono istniejący wynik PCA: {existing_pca}")
            print("[INFO] Pomijam etap redukcji PCA.")
            continue

        print(f"[INFO] Nie znaleziono pliku PCA dla {k} komponentów — rozpoczynam obliczenia.")
        pca_result = reduce_hsi(
            scene_dir=OUT_RFL_DIR,
            hdr_name=os.path.basename(rfl_hdr_path),
            n_pca=k,
            out_pca_dir=OUT_PCA_DIR,
            out_pca_name=f"Xpca_{k}"
        )

        print("\n=== [INFO] Podsumowanie PCA ===")
        print(f" Plik PCA: {pca_result['X_pca_path']}")
        print(f" Kształt: {pca_result['X_pca_shape']}")
        print(f" Liczba komponentów PCA: {pca_result['pca_k']}")
        print(f" Wymiary rastra: {pca_result['shape_rc']}")
        print(" PCA zakończone pomyślnie")

    print("\n=== [DONE] Pipeline ukończony ===")
