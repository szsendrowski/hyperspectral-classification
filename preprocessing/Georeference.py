import os
import json
import rasterio

def add_georeference_to_pca(pca_json_path, reflectance_hdr_path):

    pca_json_path = os.path.abspath(pca_json_path)
    reflectance_hdr_path = os.path.abspath(reflectance_hdr_path)

    # Znajdź faktyczny plik danych (może być .bsq, .img lub .dat)
    base = reflectance_hdr_path.replace(".hdr", "")
    possible_exts = [".bsq", ".img", ".dat"]
    reflectance_data_path = None
    for ext in possible_exts:
        candidate = base + ext
        if os.path.exists(candidate):
            reflectance_data_path = candidate
            break

    if reflectance_data_path is None:
        raise FileNotFoundError(
            f"Nie znaleziono danych reflektancji (.bsq/.img/.dat) dla: {reflectance_hdr_path}"
        )

    # Odczytaj transform i CRS z reflektancji
    with rasterio.open(reflectance_data_path) as src:
        transform = src.transform
        crs = src.crs.to_string() if src.crs else None
        rows, cols = src.height, src.width

    print(f"[INFO] Odczytano georeferencję z reflektancji:")
    print(f"       Plik danych: {os.path.basename(reflectance_data_path)}")
    print(f"       CRS: {crs}")
    print(f"       Transform: {transform}")
    print(f"       Rozmiar: {rows} × {cols}")

    # Wczytaj meta PCA
    with open(pca_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Dodaj dane przestrzenne
    meta["transform"] = [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f]
    meta["crs"] = crs
    meta["shape_rc"] = [rows, cols]

    # Zapisz zmodyfikowany plik JSON
    with open(pca_json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] Georeferencja dodana do pliku PCA JSON: {pca_json_path}")
    return meta