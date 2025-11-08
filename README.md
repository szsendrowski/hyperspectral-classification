# Klasyfikacja użytkowania gruntów oparta na danych hiperspektralnych AVIRIS-NG

Kompletny proces do przetwarzania scen AVIRIS-NG, redukcji ich wymiarowości oraz trenowania wielu klasyfikatorów pokrycia terenu (Random Forest, kNN, SVM i MLP). Projekt łączy korekcję radiometryczną, analizę głównych składowych, trenowanie modeli uczenia maszynowego oraz perceptronu wielowarstwowego i automatyczne generowanie raportu PDF.

Wykonanie: inż. Szymona SENDROWSKI \
KNS ROZPOZNANIE OBRAZOWE ____________ Wojskowa Akademia Techniczna ___________ Warszawa 2025
## Funkcjonalności
- **Korekcja radiometryczna** z wykorzystaniem metody DOS, Cosine oraz maskowania pasm wodnych w pliku [`preprocessing/RadiometricCorrection.py`](preprocessing/RadiometricCorrection.py).
- **Strumieniowa redukcja PCA** do zmniejszenia liczby informacji przy użyciu `IncrementalPCA` w [`preprocessing/DimensionReduction.py`](preprocessing/DimensionReduction.py).
- **Automatyczne georeferencjonowanie** rasterów PCA z dziedziczeniem metadanych z kostki reflektancji dzięki [`preprocessing/Georeference.py`](preprocessing/Georeference.py).
- **Klasyfikatory oparte na uczeniu maszynowym** (Random Forest, kNN, SVM) trenowane na przygotowanym dataset [`preprocessing/ground_truth.shp`], a wyniki są przechowywane w katalogu [`classification/`](classification/).
- **Klasyfikator oparty na perceptronie wielowarstwowym MLP** trenowany również na `ground_truth.shp`
- **Rozbudowana analityka** obejmująca macierze pomyłek, krzywe uczenia i tabele podsumowań eksportowane do raportu PDF przez [`analyze/TrainingReport.py`](analyze/TrainingReport.py).

## Struktura repozytorium
```
├── analyze/                # Narzędzia do generowania raportu
├── classification/         # Modele uczenia maszynowego i perceptron wielowarstwowy
├── data/                   # Wejściowa scena hiperspektralna i plik shape z prawdą terenową
├── preprocessing/          # Korekcja radiometryczna, PCA, georeferencja
├── main.py                 # Główny plik
├──README.md                # Dokumentacja projektu (ten plik)
└──requirements.txt         # Plik z wymaganymmi bibliotekami
```

## Wymagania
- Python 3.10+
- Stos kompatybilny z GDAL (Rasterio, GeoPandas) — na wielu platformach wymaga pakietów systemowych `gdal`, `geos`, `proj`.
- Sugerowane zależności Pythona:
  ```bash
    # Utwórz i aktywuj środowisko wirtualne
    python -m venv .venv
    source .venv/bin/activate      # Linux / macOS
    # lub (Windows):
    .venv\Scripts\activate

    # Zainstaluj wszystkie zależności projektu
    pip install -r requirements.txt
  ```
  > Wskazówka: utwórz środowisko wirtualne (`python -m venv .venv && source .venv/bin/activate`) przed instalacją pakietów.

## Przygotowanie danych
1. Pobierz scenę AVIRIS-NG i rozpakuj ją do katalogu `data/`, tak aby plik nagłówkowy radiancji znajdował się w:
   ```
   data/<NAZWA_SCENY>_rdn_v1n2/<NAZWA_SCENY>_rdn_v1n2_img_bsq.hdr
   na przykład: ang20160917t203013_rdn_v1n2
   ```
2. Udostępnij plik shape z zestawem danych do trenowania zawierający etykiety klas w `data/ground_truth.*` (przykładowe pliki są już dołączone).
3. Dostosuj `SCENE_NAME`, współrzędne geograficzne i inne stałe na początku pliku [`main.py`](main.py), jeśli pracujesz z inną sceną lub układem odniesienia.

## Uruchamianie potoku
1. Aktywuj środowisko Pythona z wymaganymi zależnościami.
2. Wykonaj polecenie:
   ```bash
   python main.py
   ```

Skrypt realizuje następujące etapy (pomijając kroki, dla których istnieją już wyniki):
1. **Korekcja atmosferyczna** kostki radiancji → zapisuje reflektancję (`*_rfl.bsq/.hdr`).
2. **Inkrementalna PCA** → zapisuje `Xpca_<k>.dat` oraz metadane JSON.
3. **Georeferencja** wyników PCA.
4. **Trenowanie i implementacja modeli** Random Forest, kNN, SVM i MLP, w tym eksport map rastrowych.
5. **Generowanie raportu PDF** z wykresami dokładności, macierzami pomyłek w `analyze/training_report.pdf`.

## Artefakty wyjściowe
- `classification/models/*.joblib` – zapisane modele sklearn/MLP.
- `classification/maps/*.tif` – rastrowe mapy klasyfikacji wyrównane z kostką reflektancji.
- `data/<NAZWA_SCENY>_rfl/` – skorygowana kostka reflektancji.
- `data/<NAZWA_SCENY>_pca/` – pamięciowo mapowana PCA wraz z metadanymi.
- `analyze/training_report.pdf` – zbiorczy raport treningowy.

## Rozwiązywanie problemów
- **Brak bibliotek GDAL/PROJ**: zainstaluj odpowiednie pakiety systemowe (`sudo apt install gdal-bin libgdal-dev`) lub użyj `conda install gdal`.
- **Wysokie zużycie pamięci podczas PCA**: dostosuj `COMPONENT_NUMBER`, `MAX_SAMPLES` lub rozmiary paczek (`batch_rows`, `chunk_rows`) w modułach preprocessing.
- **Błędy CUDA w PyTorch**: uruchom MLP na CPU, upewniając się, że CUDA jest wyłączona, lub zainstaluj odpowiednią wersję PyTorch z obsługą GPU.

## Licencja
Projekt udostępniany na licencji MIT. Szczegóły w pliku [`LICENSE`](LICENSE).