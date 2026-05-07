# Detekcja Phishingu na Podstawie Cech URL

Projekt zaliczeniowy — Metody Uczenia Maszynowego, semestr 2.

## Problem

Klasyfikacja binarna URL jako **phishing (0)** lub **legalny (1)** na podstawie
strukturalnych cech adresu URL (długość, obfuskacja, HTTPS, subdomeny itp.).
Brak analizy zawartości strony — tylko cechy wyciągane z samego URL.

**Dataset:** [PhiUSIIL Phishing URL Dataset](https://www.kaggle.com/datasets/ndarvind/phiusiil-phishing-url-dataset)
— 235 795 wierszy, 50+ cech, label: 0=phishing / 1=legalny.

## Wyniki

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 98.43% | 98.89% | 97.43% | 98.15% |
| Random Forest | **99.66%** | **99.83%** | **99.39%** | **99.61%** |

Cross-validation (5-fold, SMOTE wewnątrz foldów):

Uwaga metodyczna: w CV SMOTE wykonywane jest wewnątrz foldów (na treningowej części foldu),
 LR: 98.27% ±0.08%
 RF: 99.67% ±0.02%
żeby uniknąć data leakage.

## Metody

### Preprocessing
1. **Deduplikacja** — usunięcie 425 zduplikowanych URL (`preprocessing.py`)
2. **Feature selection** — zachowanie 10 cech strukturalnych URL (`feature_selection.py`):
   `URLLength`, `DomainLength`, `IsDomainIP`, `TLDLength`, `NoOfSubDomain`,
   `IsHTTPS`, `CharContinuationRate`, `HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio`
3. **Balansowanie klas** — SMOTE na zbiorze treningowym (42.8% → 50% phishing)
4. **Normalizacja** — StandardScaler w Pipeline Logistic Regression

### Modele
- **Logistic Regression** — model bazowy, liniowy, interpretowalny
- **Random Forest** — 100 drzew, ensemble, wyższa skuteczność

### Walidacja
- Podział 80/20 (stratified), `random_state=42`
- 5-fold Stratified Cross-Validation (SMOTE wewnątrz foldów)
- Metryki: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC/AUC

### Analiza odporności (Robustness Analysis)
Testowanie degradacji modelu pod Gaussowskim szumem (`noise_std` 0.0–1.0)
jako przybliżenie adversarial examples — atakujący modyfikujący cechy URL
by obejść detekcję. Wyniki w `outputs/robustness_analysis.png`.

## Struktura projektu

```
phishing-detector/
├── main.ipynb              # Główny notebook z pełną analizą
├── config.py               # Stałe (nazwa datasetu, kolumny)
├── data_io.py              # Ładowanie danych z Kaggle / cache
├── eda.py                  # Eksploracyjna analiza danych
├── preprocessing.py        # Deduplikacja + preprocessing cech
├── features.py             # Przygotowanie macierzy cech
├── feature_selection.py    # Selekcja 10 cech URL
├── models.py               # Trening LR i RF
├── evaluation.py           # Metryki ewaluacji
├── roc_analysis.py         # Krzywe ROC/AUC
├── feature_importance.py   # Ważność cech RF
├── cross_validation.py     # 5-fold CV
├── noise_injection.py      # Robustness analysis (Gaussian noise)
├── requirements.txt        # Zależności
└── outputs/                # Wykresy PNG
    ├── 01_class_distribution.png
    ├── 03_model_evaluation.png
    ├── 05_feature_importance.png
    ├── roc_curve_baseline.png
    └── robustness_analysis.png
```

## Uruchomienie

```bash
pip install -r requirements.txt
# Następnie otwórz main.ipynb i uruchom wszystkie komórki
```

Dane pobierane automatycznie z Kaggle przy pierwszym uruchomieniu
(wymagane konto Kaggle i plik `~/.kaggle/kaggle.json`).

## Wnioski metodyczne

**Dlaczego tylko cechy URL?** Cechy zawartości strony (JavaScript, HTML) są
dostępne dopiero po odwiedzeniu URL — w systemie IDS/SOC musimy klasyfikować
URL *przed* odwiedzeniem. Ograniczenie do 10 cech strukturalnych to świadomy
wybór pod kątem bezpieczeństwa operacyjnego.

**Dlaczego SMOTE tylko na train?** Augmentacja danych testowych fałszuje metryki.
SMOTE musi być stosowany wyłącznie wewnątrz pętli treningowej.

**Dlaczego Random Forest wygrywa?** Cechy URL mają nieliniowe zależności
(np. IsHTTPS samo w sobie nie wystarczy — phishing też używa HTTPS).
Ensemble drzew lepiej modeluje te interakcje niż model liniowy.
