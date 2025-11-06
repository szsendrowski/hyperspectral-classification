import os
from RadiometricCorrection import run_atm_corr


# === ŚCIEŻKI WEJŚCIOWE ===
BASE_DIR = r"C:\Users\sciezka\do\folderu"
SCENE_DIR = os.path.join(BASE_DIR, "ang20160917t203013_rdn_v1n2")

RDN_HDR = "ang20160917t203013_rdn_v1n2_img_bsq.hdr"

# === ŚCIEŻKI WYJŚCIOWE ===
OUT_RFL_DIR = os.path.join(BASE_DIR, "ang20160917t203013_rfl_SIMPLE")

os.makedirs(OUT_RFL_DIR, exist_ok=True)

# === PARAMETRY GEOMETRYCZNE / LOKALIZACYJNE ===
LAT = 34.035  # wpisz dokładne współrzędne sceny (np. centrum obrazu)
LON = -117.5

if __name__ == "__main__":

    print("=== [1/2] Korekcja atmosferyczna AVIRIS-NG ===")

    rfl_hdr_path = run_atm_corr(
        scene_dir=SCENE_DIR,
        rdn_hdr=RDN_HDR,
        lat=LAT,
        lon=LON,
        out_dir=OUT_RFL_DIR
    )

    print(f"✅ Reflektancja zapisana: {rfl_hdr_path}")

