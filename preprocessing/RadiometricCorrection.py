import os
import re
import numpy as np
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import spectral.io.envi as envi
from pysolar.solar import get_altitude, get_azimuth


def parse_datetime_from_basename(basename: str) -> datetime:
    m = re.search(r"ang(\d{4})(\d{2})(\d{2})t(\d{2})(\d{2})(\d{2})", basename)
    if not m:
        raise ValueError("Nie udało się znaleźć daty w nazwie pliku: " + basename)
    y, mo, d, H, M, S = map(int, m.groups())
    return datetime(y, mo, d, H, M, S, tzinfo=timezone.utc)

def ensure_um_units(wavelengths, fwhm):
    wl = np.array(wavelengths, dtype=np.float32)
    fw = np.array(fwhm, dtype=np.float32) if fwhm is not None else None
    if np.nanmean(wl) > 10:  # nm → µm
        wl *= 1e-3
        if fw is not None:
            fw *= 1e-3
    return wl, fw

def aviris_rdn_to_SI(L_aviris):
    # µW·cm^-2·nm^-1·sr^-1 → W·m^-2·µm^-1·sr^-1
    return L_aviris.astype(np.float32) * 10.0

def compute_sun_angles(lat, lon, when_local):
    try:
        alt = float(get_altitude(lat, lon, when_local))
        zen = 90.0 - alt
        az = float(get_azimuth(lat, lon, when_local))
        if alt < 0:
            print(f"[WARN] Słońce pod horyzontem (alt={alt:.2f}°). Sprawdź strefę czasową.")
        zen = np.clip(zen, 0.0, 89.0)
        return zen, az
    except Exception as e:
        print(f"[ERROR] Nie udało się obliczyć kątów słonecznych: {e}")
        return 45.0, 180.0

def dos_with_mask(band, mask, percentile=1.0):
    """Dark Object Subtraction liczona tylko z ważnych pikseli."""
    if mask is None or not np.any(mask):
        return band
    low_val = np.nanpercentile(band[mask], percentile)
    out = band - low_val
    out[~np.isfinite(out)] = 0
    out[out < 0] = 0
    return out

def run_atm_corr(scene_dir,
                        rdn_hdr,
                        lat,
                        lon,
                        tz_offset=0.0,
                        out_dir=None):
    """
    Krekcja atmosferyczna AVIRIS-NG:
      - DOS (z maską NoData)
      - korekcja Cosine cos(θs)
      - PER-BAND contrast stretch: p50–p95 → [target_low, target_high]
      - maska pasm absorpcji wody
    """

    # import
    in_hdr_path = os.path.join(scene_dir, rdn_hdr)
    possible_suffixes = ["", ".img", ".bil", ".bsq", ".dat", "_img"]
    in_img_path = None
    for suffix in possible_suffixes:
        candidate = in_hdr_path.replace(".hdr", suffix)
        if os.path.exists(candidate):
            in_img_path = candidate
            break
    if in_img_path is None:
        raise FileNotFoundError(f"Nie znaleziono danych binarnych dla {rdn_hdr}")

    if out_dir is None:
        out_dir = os.path.join(scene_dir, "rfl_simple")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Otwieram ENVI: {os.path.basename(in_img_path)}")
    hdr = envi.read_envi_header(in_hdr_path)
    img = envi.open(in_hdr_path, in_img_path)

    nodata = None
    for key in ("data ignore value", "no data value", "nodata", "baddata value"):
        if key in hdr:
            try:
                nodata = float(hdr[key])
                break
            except Exception:
                pass
    if nodata is None:
        nodata = -9999.0

    if "wavelength" in hdr:
        wavelengths = np.array([float(w) for w in hdr["wavelength"]], dtype=np.float32)
    elif "band names" in hdr:
        wavelengths = np.array([float(re.search(r"([\d.]+)", name).group(1))
                                for name in hdr["band names"]], dtype=np.float32)
    else:
        wavelengths = np.arange(img.nbands, dtype=np.float32)
    fwhm = np.array(hdr.get("fwhm", np.full(len(wavelengths), 0.01)), dtype=np.float32)
    wavelengths, fwhm = ensure_um_units(wavelengths, fwhm)

    # Maska wody
    mask_valid = ~(((wavelengths > 1.35) & (wavelengths < 1.45)) |
                   ((wavelengths > 1.80) & (wavelengths < 2.00)))
    valid_idx = np.where(mask_valid)[0]
    print(f"[INFO] Usuwam {np.sum(~mask_valid)} pasm wody — pozostaje {len(valid_idx)} pasm.")

    # Kąt Słońca
    acq_time_utc = parse_datetime_from_basename(os.path.basename(in_hdr_path))
    acq_time_local = acq_time_utc + timedelta(hours=float(tz_offset))
    solar_zen, solar_az = compute_sun_angles(lat=lat, lon=lon, when_local=acq_time_local)
    cos_sun = np.cos(np.deg2rad(solar_zen))
    cos_sun = max(cos_sun, 0.05)
    print(f"[INFO]  Słońce: zen={solar_zen:.2f}°, az={solar_az:.2f}°, cos(θs)={cos_sun:.3f}")

    # Zapis
    base_name = os.path.splitext(os.path.basename(in_img_path))[0]
    tmp_rfl_path = os.path.join(out_dir, f"{base_name}_rfl.bsq")

    total_bands = len(valid_idx)
    block_size = 64
    n_written = 0

    # Zakres reflektancji
    target_low, target_high = 0.0, 1.0

    print(f"[INFO] Korekcja (DOS + cosine + per-band contrast) → {tmp_rfl_path}")

    with open(tmp_rfl_path, "wb") as out_fp:
        for i, b in enumerate(tqdm(valid_idx, desc="Korekcja", unit="pasmo")):
            try:
                band = img.read_band(b).astype(np.float32)

                # Maska ważnych pikseli
                valid_mask = np.isfinite(band) & (band != nodata)

                # Radiancja → SI, DOS, Cosine
                band_si = aviris_rdn_to_SI(band)
                band_dos = dos_with_mask(band_si, valid_mask, percentile=1.0)
                band_corr = np.zeros_like(band_dos, dtype=np.float32)
                band_corr[valid_mask] = band_dos[valid_mask] / cos_sun

                # Percentyle z ważnych pikseli
                if np.any(valid_mask):
                    p50 = np.nanpercentile(band_corr[valid_mask], 50)
                    p95 = np.nanpercentile(band_corr[valid_mask], 95)
                else:
                    p50, p95 = 0.0, 1.0

                if not np.isfinite(p50): p50 = 0.0
                if not np.isfinite(p95) or p95 <= p50: p95 = p50 + 1e-6

                # Rozciąganie
                band_rfl = np.zeros_like(band_corr, dtype=np.float32)
                # normalizacja do 0–1
                band_rfl[valid_mask] = (band_corr[valid_mask] - p50) / (p95 - p50)
                # przeskalowanie do docelowego zakresu
                band_rfl[valid_mask] = target_low + band_rfl[valid_mask] * (target_high - target_low)
                # clip
                np.clip(band_rfl, 0, 1, out=band_rfl)
                # reszta (NoData) = 0
                band_rfl[~valid_mask] = 0.0

                # Zapis blokowy
                for row_start in range(0, band_rfl.shape[0], block_size):
                    row_end = min(row_start + block_size, band_rfl.shape[0])
                    out_fp.write(band_rfl[row_start:row_end, :].astype(np.float32).tobytes())
                n_written += band_rfl.size

                if i % 50 == 0:
                    print(f"[DBG] Band {b}: λ={wavelengths[b]:.3f} µm, p50={p50:.4g}, p95={p95:.4g}")

            except Exception as e:
                print(f"[WARN] Pomijam pasmo {b + 1}: {e}")
                zeros = np.zeros((img.nrows, img.ncols), dtype=np.float32)
                for row_start in range(0, zeros.shape[0], block_size):
                    row_end = min(row_start + block_size, zeros.shape[0])
                    out_fp.write(zeros[row_start:row_end, :].tobytes())
                n_written += zeros.size

    expected = img.nrows * img.ncols * total_bands
    if n_written < expected:
        print(f"[WARN]️ Zapisano mniej pikseli ({n_written}/{expected}).")

    # === Nagłówek ENVI ===
    meta = hdr.copy()
    meta["description"] = "Surface reflectance (DOS + cosine + per-band p50–p95 stretch + water mask)"
    meta["interleave"] = "bsq"
    meta["data type"] = 4
    meta["byte order"] = 0
    meta["bands"] = total_bands
    meta["wavelength"] = [float(w) for w in wavelengths[mask_valid]]
    meta["reflectance scale factor"] = 1.0
    meta["wavelength units"] = "micrometers"

    out_hdr = tmp_rfl_path.replace(".bsq", ".hdr")
    envi.write_envi_header(out_hdr, meta)

    print(f"[DONE] Zapisano wynikowy plik reflektancji: {out_hdr}")
    return out_hdr
