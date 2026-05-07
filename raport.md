# Raport: Detekcja Phishingu na Podstawie Cech URL

**Dataset:** PhiUSIIL Phishing URL Dataset (Kaggle)  
**Metoda:** Logistic Regression vs. Random Forest  
**Cel oceny:** 5.0

---

## 1. Opis problemu

Phishing polega na podszywaniu się pod legalne serwisy w celu wyłudzenia danych
użytkowników. Automatyczna detekcja złośliwych URL na podstawie mierzalnych cech
strukturalnych pozwala wbudować ochronę bezpośrednio w warstwy sieciowe (IDS, proxy,
bramki e-mail) bez potrzeby analizy treści strony, co daje efektywność w czasie
rzeczywistym — zanim użytkownik otworzy stronę.

Dataset **PhiUSIIL** zawiera pre-wyekstrahowane cechy numeryczne URL — długość,
liczbę specjalnych znaków, obecność IP, subdomen, TLD, wskaźniki leksykalne itp.
Nie wymaga to ręcznego parsowania surowych URLi.

Świadomym wyborem projektowym jest ograniczenie cech wyłącznie do **struktury URL**
(bez cech zawartości strony, takich jak liczba obrazów, JS, kod HTML). Uzasadnienie:
cechy HTML są dostępne dopiero *po* odwiedzeniu URL — w systemie IDS/SOC decyzja
musi zapadać *przed* odwiedzeniem.

---

## 2. Analiza danych (EDA)

### 2.1 Podstawowe informacje

| Cecha | Wartość |
|-------|---------|
| Liczba próbek (przed dedupl.) | 235 795 |
| Zduplikowane rekordy usunięte | 425 (URL duplicates) |
| Liczba próbek (po dedupl.) | **235 370** |
| Cechy numeryczne dostępne | 50 |
| Cechy użyte do modelowania | **10 (URL-only)** |
| Kolumna etykiety | `label` (0 = phishing, 1 = legalny) |
| Brakujące wartości | **0** — dataset jest kompletny |

### 2.2 Rozkład klas

| Klasa | Liczba | Odsetek |
|-------|-------:|--------:|
| Phishing (0) | 100 945 | 42.8 % |
| Legalny (1) | 134 850 | 57.2 % |

*Rozkład z EDA (przed deduplikacją, 235 795 wierszy). Po usunięciu 425 duplikatów
URL phishing = 100 520 / 235 370 = 42.7% — różnica pomijalna.*

Dataset jest **lekko niezbalansowany** (~57/43). SMOTE zastosowany wyłącznie
na zbiorze treningowym wyrównuje proporcje do 107 880 / 107 880 przez generację
syntetycznych próbek klasy mniejszościowej (phishing).

### 2.3 Wybrane cechy URL (10)

| # | Cecha | Opis |
|---|-------|------|
| 1 | `URLLength` | Całkowita długość adresu URL |
| 2 | `DomainLength` | Długość samej domeny |
| 3 | `IsDomainIP` | Czy domena to adres IP zamiast nazwy (0/1) |
| 4 | `TLDLength` | Długość TLD (top-level domain) |
| 5 | `NoOfSubDomain` | Liczba subdomen |
| 6 | `IsHTTPS` | Czy protokół to HTTPS (0/1) |
| 7 | `CharContinuationRate` | Spójność ciągu znaków w URL |
| 8 | `HasObfuscation` | Czy URL zawiera obfuskację (0/1) |
| 9 | `NoOfObfuscatedChar` | Liczba zaciemnionych znaków |
| 10 | `ObfuscationRatio` | Stosunek znaków obfuskowanych do całości |

Usunięto 40 cech związanych z zawartością strony (HTML, JS, liczba obrazów, itp.).

---

## 3. Preprocessing

### Kroki wykonane

1. **Usunięcie duplikatów URL** — `drop_duplicates(subset=['URL'])`: usunięto 425 rekordów
2. **Weryfikacja binarności etykiet** — mapowanie do `{0, 1}`, odrzucenie wierszy bez etykiety
3. **Selekcja cech numerycznych** — zachowanie tylko `dtype in ['number']`
4. **Imputacja medianą** — wypełnienie ewentualnych NaN (w tym datasecie brak)
5. **Selekcja 10 cech URL-only** — świadome zawężenie do cech strukturalnych URL
6. **Podział 80/20** — `train_test_split(stratify=y, random_state=42)`
7. **SMOTE** — tylko na `X_train`; zbiór testowy pozostaje niezmieniony
8. **Normalizacja (StandardScaler)** — w Pipeline Logistic Regression; dopasowanie tylko na `X_train_smote`

### Dlaczego SMOTE tylko na zbiorze treningowym?

Augmentacja zbioru testowego fałszuje metryki — test musi odzwierciedlać realne
proporcje klas. Dodatkowo, zastosowanie SMOTE przed CV spowodowałoby data leakage
(syntetyczne próbki z foldów walidacyjnych byłyby "pochodną" danych treningowych).
Dlatego SMOTE stosujemy po splicie, wyłącznie na `X_train`.

### Spójność CV z ewaluacją testową

Cross-Validation uruchamiane jest na zbiorze treningowym `X_train` (przed SMOTE), a SMOTE
wykonywane jest **wewnątrz każdego foldu** w pipeline (tylko na części treningowej danego foldu).
To eliminuje ryzyko data leakage i jest metodologicznie poprawnym odpowiednikiem tego, jak
model byłby trenowany na nowych danych.

---

## 4. Opis modeli

### 4.1 Logistic Regression

Liniowy klasyfikator probabilistyczny. Szacuje prawdopodobieństwo klasy przez funkcję
sigmoid sumy ważonej cech. Wagi są łatwe do interpretacji — wartość dodatnia oznacza
korelację z klasą pozytywną (legalny). Wymaga skalowania cech — zrealizowane przez
StandardScaler w sklearn Pipeline.

**Hiperparametry:** `max_iter=2000`, `solver=lbfgs`, `random_state=42`

### 4.2 Random Forest

Ensemble 100 drzew decyzyjnych trenowanych na losowych podzbiorach cech i próbek
(bagging + feature subsampling). Klasyfikacja przez głosowanie większościowe. Odporny
na wartości odstające, nie wymaga skalowania, generuje Feature Importance.

**Hiperparametry:** `n_estimators=100`, `n_jobs=-1`, `random_state=42`

---

## 5. Wyniki — metryki na zbiorze testowym

Zbiór testowy: **47 074 próbek** (20% datasetu, stratified split, czysty — bez SMOTE).

### 5.1 Tabela zbiorcza

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|:--------:|:---------:|:------:|:--------:|:---:|
| Logistic Regression | 0.9843 | 0.9889 | 0.9743 | 0.9815 | 0.9971 |
| **Random Forest** | **0.9966** | **0.9983** | **0.9939** | **0.9961** | **0.9978** |

*Precision, Recall, F1 liczone dla klasy phishing (pos_label=0).*

### 5.2 Cross-Validation (5-fold, stratified, SMOTE wewnątrz foldów)

| Model | CV Accuracy | CV F1 (mean) | CV F1 (std) |
|-------|:-----------:|:------------:|:-----------:|
| Logistic Regression | 0.9827 | 0.9796 | ±0.0009 |
| **Random Forest** | **0.9967** | **0.9961** | **±0.0002** |

Wyniki CV są spójne z wynikami na zbiorze testowym — brak oznak overfittingu ani
data leakage. Niskie std potwierdza stabilność modeli niezależnie od podziału danych.

### 5.3 Macierz pomyłek

**Logistic Regression:**

|  | Pred. Phishing | Pred. Legalny |
|--|:--------------:|:-------------:|
| **Actual Phishing** | 19 587 (TP) | 517 (FN) |
| **Actual Legalny** | 220 (FP) | 26 750 (TN) |

**Random Forest:**

|  | Pred. Phishing | Pred. Legalny |
|--|:--------------:|:-------------:|
| **Actual Phishing** | 19 981 (TP) | 123 (FN) |
| **Actual Legalny** | 35 (FP) | 26 935 (TN) |

**Interpretacja:** Random Forest przepuszcza tylko **123 phishingowe URL** (FN) na
20 104 rzeczywistych phishingów — Recall = 99.39%. Logistic Regression przepuszcza 517.
Z punktu widzenia bezpieczeństwa FN jest groźniejszy niż FP (przepuszczony atak vs.
fałszywy alarm) — RF jest lepszym wyborem dla środowisk high-security.

### 5.4 Feature Importance — Top 10 (Random Forest)

| Ranga | Cecha | Importance | Uwaga |
|------:|-------|----------:|-------|
| 1 | `IsHTTPS` | 0.3915 | **39%** mocy predykcyjnej |
| 2 | `URLLength` | 0.2402 | Długość URL |
| 3 | `CharContinuationRate` | 0.1553 | Leksykalna spójność |
| 4 | `DomainLength` | 0.0939 | Długość domeny |
| 5 | `NoOfSubDomain` | 0.0925 | Liczba subdomen |
| 6 | `TLDLength` | 0.0257 | Długość TLD |
| 7 | `IsDomainIP` | 0.0006 | Praktycznie bez wpływu |
| 8 | `ObfuscationRatio` | 0.0002 | Praktycznie bez wpływu |
| 9 | `HasObfuscation` | 0.0000 | Nieistotna |
| 10 | `NoOfObfuscatedChar` | 0.0000 | Nieistotna |

**Kluczowa obserwacja:** `IsHTTPS` odpowiada za ~39% mocy predykcyjnej, ale to cechy
leksykalne łącznie (`URLLength` + `CharContinuationRate` + `DomainLength` + `NoOfSubDomain`)
dostarczają kolejnych ~50%. Cechy obfuskacji mają importance ≈ 0 — phishingowe URL
w tym datasecie rzadko używają obfuskacji jako techniki ukrycia.

---

## 6. Analiza statystyczna

### 6.1 Porównanie modeli

Random Forest (F1=0.9961) przewyższa Logistic Regression (F1=0.9815) o ~1.5 punktu
procentowego. W typowych zbiorach phishingowych RF wygrywa dzięki zdolności do
modelowania **nieliniowych interakcji między cechami** — np. kombinacja krótkiego URL
z HTTPS i dużą liczbą subdomen może być phishingiem, czego model liniowy nie uchwwyci
bez ręcznej inżynierii cech.

Logistic Regression jest szybsza w inferencji i w pełni interpretowalna przez wagi —
ważne przy wymaganiach regulacyjnych (GDPR) lub audytach bezpieczeństwa.

### 6.2 Krytyczna analiza dominacji `IsHTTPS`

`IsHTTPS` z importance 39% sygnalizuje, że w tym datasecie phishingowe URL znacznie
rzadziej używają HTTPS niż legalne. To **zależność temporalna** — w latach 2017–2019
phishing rzadko używał certyfikatów. Dziś (2024) Let's Encrypt umożliwia bezpłatne
certyfikaty, więc `IsHTTPS` traci moc predykcyjną w nowszych danych. Model wymaga
regularnego retrainingu.

### 6.3 Niska importance cech obfuskacji

`HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio` mają importance ≈ 0.
Interpretacja: kampanie phishingowe w datasecie PhiUSIIL nie opierają się na obfuskacji
URL jako technice ukrycia — wolą wyglądające-legalnie domeny (typosquatting, subdomenowanie).
To ważna informacja operacyjna: monitorowanie obfuskacji URL nie jest priorytetem dla tego
wektora ataku.

### 6.4 Paradoks AUC vs F1 — LR vs RF

Nieoczekiwana obserwacja: mimo że RF osiąga wyższe F1 (0.9961 vs 0.9815), jego AUC
jest tylko nieznacznie wyższe od LR (0.9978 vs 0.9971). Różnica w AUC wynosi zaledwie
**0.0007**, podczas gdy różnica w F1 wynosi **0.0146**.

Interpretacja: **RF ma lepszy punkt operacyjny** przy domyślnym progu decyzyjnym p=0.5
— jest "trafniejszy" w konkretnej decyzji phishing/nie-phishing. Natomiast **LR ma lepiej
skalibrowane prawdopodobieństwa** na całej krzywej ROC — jego score'y p lepiej odzwierciedlają
rzeczywiste prawdopodobieństwo klasy w szerokim zakresie progów.

Praktyczna implikacja dla SOC:
- Jeśli system używa **stałego progu** (np. p > 0.5 → blokuj) → RF jest lepszy (wyższy F1)
- Jeśli system **kalibruje próg dynamicznie** pod FPR/TPR (np. z krzywej ROC) → LR jest
  konkurencyjny i daje interpretowalniejsze prawdopodobieństwa do rankingu alertów

To klasyczny trade-off ensemble vs. model liniowy: RF optymalizuje lokalnie, LR globalna
separowalność klas.

---

## 7. Analiza Odporności Modelu (Robustness Analysis)

Testowano degradację modeli pod wpływem Gaussowskiego szumu dodawanego do cech URL.
Szum był proporcjonalny do odchylenia standardowego każdej cechy (noise_std ∈ {0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00}).
Celem eksperymentu było zasymulowanie sytuacji, w której atakujący modyfikuje strukturę URL
w celu obejścia detekcji phishingu (adversarial perturbation).

### Wyniki (F1 dla klasy phishing)

| noise_std | LR F1 | RF F1 |
|:---------:|:-----:|:-----:|
| 0.00 (czysty) | 0.9815 | 0.9961 |
| 0.05 | 0.9815 | 0.7481 |
| 0.10 | 0.7601 | 0.7140 |
| 0.20 | 0.6699 | 0.6945 |
| 0.30 | 0.6432 | 0.6821 |
| 0.50 | 0.6217 | 0.6610 |
| 1.00 | 0.5984 | 0.6403 |

### Interpretacja wyników

W przeciwieństwie do standardowej ewaluacji na czystym zbiorze testowym,
oba modele okazały się silnie wrażliwe na perturbacje cech wejściowych.

Najważniejsze obserwacje:

- Już niewielki szum (`noise_std=0.05`) powoduje znaczący spadek F1:
  - Logistic Regression: `0.9815 → 0.8631`
  - Random Forest: `0.9961 → 0.7481`

- Random Forest osiąga wyższe wyniki przy większym poziomie szumu (`noise_std ≥ 0.20`),
  jednak jego degradacja przy małych perturbacjach jest gwałtowniejsza niż dla Logistic Regression.

- Przy bardzo dużym szumie (`noise_std=1.00`) oba modele tracą znaczną część skuteczności:
  - Logistic Regression: `F1 ≈ 0.60`
  - Random Forest: `F1 ≈ 0.64`

Eksperyment pokazuje, że modele trenowane wyłącznie na statycznych cechach URL
są podatne na przesunięcie rozkładu danych (distribution shift) i manipulację cechami.

### Wnioski metodologiczne

Robustness Analysis nie mierzy standardowej jakości klasyfikacji,
lecz odporność modeli na sztucznie wprowadzone perturbacje cech.

Dlatego wyniki tej sekcji nie powinny być porównywane bezpośrednio
z metrykami uzyskanymi na czystym zbiorze testowym.

W praktyce oznacza to, że:
- modele bardzo dobrze działają na danych podobnych do treningowych,
- ale ich skuteczność może silnie spadać przy zmianie charakterystyki URL
  lub celowej manipulacji przez atakującego.

Wykres: `outputs/robustness_analysis.png`

---

## 7.1 Eksperyment kontrolny — ewaluacja na zaszumionym zbiorze testowym

W pierwotnej wersji kodu zbiór testowy był zaszumiany przed ewaluacją tą samą
funkcją co zbiór treningowy (`inject_realistic_noise`: 20% brakujących wartości,
Gaussian noise std=0.30, 5 dodatkowych kolumn `random_noise_0..4`). Poniżej wyniki
tej błędnej metodologii zestawione z poprawną ewaluacją.

### Wyniki — zaszumiony X_test (błędna metodologia)

| Model | Accuracy | F1-Score |
|-------|:--------:|:--------:|
| Logistic Regression | 0.8143 | 0.7674 |
| Random Forest | 0.8502 | 0.8210 |

### Wyniki — czysty X_test (poprawna metodologia)

| Model | Accuracy | F1-Score |
|-------|:--------:|:--------:|
| Logistic Regression | 0.9843 | 0.9815 |
| Random Forest | 0.9966 | 0.9961 |

### Porównanie — wpływ błędu metodologicznego

| Model | F1 czysty | F1 zaszumiony | Różnica |
|-------|:---------:|:-------------:|:-------:|
| Logistic Regression | 0.9815 | 0.7674 | **−0.214** |
| Random Forest | 0.9961 | 0.8210 | **−0.175** |

**Dlaczego to błąd?**

Zbiór testowy musi odzwierciedlać dane produkcyjne w ich oryginalnej postaci.
Zaszumienie X_test przed ewaluacją oznacza, że nie mierzono skuteczności modelu —
mierzono jego odporność na konkretny, arbitralnie dobrany szum. Wynik 0.81 F1
nie jest właściwością modelu, lecz artefaktem metodologii.

Dodatkowo, stary kod dodawał 5 kolumn `random_noise_0..4` do X_train i X_test.
W efekcie `random_noise_2` pojawiła się na **9. miejscu Top 10 Feature Importance**
z importance 0.0307 — wypychając prawdziwą cechę `HasObfuscation`. Model uczył się
na losowym szumie jako informacyjnej cesze, co kompromituje całą analizę ważności cech.

**Poprawne podejście** (zaimplementowane w sekcji 7): szum na danych testowych
pojawia się wyłącznie w dedykowanej analizie odporności, przy jawnym oznaczeniu
poziomu perturbacji i z punktem bazowym `noise_std=0.0` równym standardowej ewaluacji.

---

## 8. Ograniczenia modelu

### 8.1 Statyczność i temporalność cech

`IsHTTPS` — historycznie silny sygnał phishingu — traci moc w miarę jak atakujący
masowo adoptują HTTPS. Model trenowany na starszych danych może generować wysokie FPR
przy nowych kampaniach phishingowych z certyfikatami SSL.

### 8.2 Dataset bias

PhiUSIIL pochodzi z konkretnego okresu. Nowe taktyki phishingowe (IDN homograph attacks,
subdomenowanie Cloud CDN, phishing na OAuth) mogą być niereprezentowane. Model wymaga
regularnego dotreniowania na nowych danych z SOC.

### 8.3 Nierównowaga klas w środowisku produkcyjnym

W realnym ruchu sieciowym procent phishingowych URL jest wielokrotnie niższy niż 42.7%.
Próg decyzyjny `p > 0.5` może wymagać kalibracji (`p > 0.3`) pod kątem akceptowalnego
False Positive Rate w danym systemie.

### 8.4 Brak cech temporalnych

Model nie modeluje wieku domeny, historii rejestracji ani sekwencji czasowej zapytań.
Nowe domeny zarejestrowane dzień wcześniej (często używane w phishingu) są klasyfikowane
wyłącznie na podstawie struktury URL — bez penalizacji za "świeżość".

### 8.5 Zależność od 10 cech URL

Model nie analizuje treści strony. Atakujący może skonstruować URL spełniający wszystkie
"bezpieczne" kryteria strukturalne (krótki URL, HTTPS, znana TLD, brak subdomen) i jednocześnie
phishingowy w treści. Przykład: `https://paypal-secure.com/login` — poprawna struktura, phishing.

---

## 9. Potencjalne wektory obejścia (Adversarial Examples)

### 9.1 Atak na `IsHTTPS` (najważniejsza cecha)

Ponieważ `IsHTTPS` to 39% importance, atakujący wyposażający phishingowy URL w certyfikat
HTTPS (darmowe przez Let's Encrypt) natychmiastowo redukuje sygnał phishingu. To nie
jest atak na model — to zmiana techniki, na którą model nie jest przygotowany.

### 9.2 URL Padding i manipulacja długością

Atakujący znający rozkład `URLLength` legalnych URL może go naśladować:
- Skrócenie phishingowego URL przez redirect (`bit.ly`)
- Lub wydłużenie do "legalnych" rozmiarów przez dodanie path suffix

### 9.3 Subdomain Stuffing

`paypal.com.attacker.xyz` — `NoOfSubDomain` = 3, `DomainLength` krótka, URL zawiera
"paypal.com". Cechy strukturalne zbliżone do legitymnych, choć URL jednoznacznie phishingowy.

### 9.4 IDN Homograph Attack

Unicode umożliwia rejestrację domen wizualnie identycznych z prawdziwymi:
`pаypal.com` (cyryliczne 'а' zamiast łacińskiego 'a'). Cechy leksykalne ASCII nie
wykryją tego bez normalizacji Unicode (Punycode decoding).

### 9.5 Atak na Random Forest (Model Evasion)

Atakujący z dostępem do API może stosować **gradient-free search** (algorytmy genetyczne,
ZOO attack) by iteracyjnie modyfikować cechy URL do momentu zmiany decyzji modelu.
Random Forest jest bardziej odporny niż LR ze względu na nieciągłość granic decyzyjnych,
ale nie jest odporny całkowicie.

---

## 10. Wnioski dotyczące zastosowania w IDS / SOC

### 10.1 Wbudowanie w IDS

Model można wdrożyć jako moduł inline w:
- **Zeek / Suricata** — analiza DNS i HTTP logów w czasie rzeczywistym; cechy obliczane
  on-the-fly z nagłówka `Host` + `URI`
- **Firewalle NGFW / Email Gateway** — punkt wymuszenia przy wyjściu ruchu HTTP/HTTPS

Wymagania produkcyjne:
- Latencja inferencji: RF z 100 drzewami < 5 ms per URL — spełnia wymogi real-time
- Serializacja: `joblib.dump(rf_model, 'rf_phishing.joblib')`

### 10.2 Rola w SOC (Security Operations Center)

W SOC model służy jako **triage layer**:
1. `p > 0.95` → automatyczne blokowanie URL
2. `0.5 < p < 0.95` → kolejka do manualnej analizy analityka
3. False Positives → feedback loop do retrainingu modelu

Model ML redukuje liczbę incydentów do manualnej analizy o ~80–90%, skracając
MTTR (Mean Time to Respond).

### 10.3 Progi decyzyjne i tuning

Domyślny próg `p = 0.5` może nie być optymalny:
- **High Security** (bankowość, ochrona zdrowia): próg `p = 0.3` → wyższy Recall,
  więcej False Positives, ale mniej przepuszczonych ataków
- **Low Friction** (ISP, ogólny użytek): próg `p = 0.7` → mniej False Positives,
  ryzyko przepuszczenia subtelnych ataków

Krzywa ROC (`outputs/roc_curve_baseline.png`) pozwala wybrać punkt operacyjny
(FPR, TPR) odpowiedni dla danego profilu ryzyka organizacji.

### 10.4 Utrzymanie modelu (MLOps)

- **Data drift monitoring** — regularne sprawdzanie rozkładu cech produkcyjnych
  vs. treningowych (np. Evidently AI, Alibi Detect)
- **Model retraining** — co 1–3 miesiące na nowych próbkach z analizy incydentów SOC
- **Ensemble z innymi sygnałami** — wiek domeny (WHOIS), reputacja IP (VirusTotal API),
  certyfikat TLS (Let's Encrypt vs. komercyjna CA) jako dodatkowe cechy

---

## 11. Podsumowanie

### Wyniki końcowe

| Model | Accuracy | Precision | Recall | F1-Score | AUC | CV F1 |
|-------|:--------:|:---------:|:------:|:--------:|:---:|:-----:|
| Logistic Regression | 0.9843 | 0.9889 | 0.9743 | 0.9815 | 0.9971 | 0.9796 ±0.0009 |
| **Random Forest** | **0.9966** | **0.9983** | **0.9939** | **0.9961** | **0.9978** | **0.9961 ±0.0002** |

### Kluczowe wnioski

| Aspekt | Wniosek |
|--------|---------|
| Lepszy model | Random Forest (F1=0.9961, mniej FN — groźniejszy błąd) |
| Interpretowalność | Logistic Regression (auditowalne wagi, szybsza inferencja) |
| Najważniejsza cecha | `IsHTTPS` (39%) — ale temporalnie niestabilna |
| Nieistotne cechy | Obfuskacja URL — phishing w datasecie jej nie używa |
| Główne ograniczenie | Model pre-visit traci skuteczność gdy atakujący używa HTTPS i "legalnej" struktury URL |
| Odporność na evasion | Oba modele silnie degradują pod perturbacją cech; RF osiąga lepsze wyniki przy dużym szumie, ale jest bardzo wrażliwy już przy niewielkich perturbacjach |
| Wdrożenie produkcyjne | DNS-layer / email gateway z URL-only, próg kalibrowany do profilu ryzyka |

---

*Kod źródłowy: `main.ipynb`, `models.py`, `evaluation.py`, `preprocessing.py`,
`feature_selection.py`, `noise_injection.py`*  
*Wykresy: `outputs/01_class_distribution.png`, `outputs/03_model_evaluation.png`,
`outputs/05_feature_importance.png`, `outputs/roc_curve_baseline.png`,
`outputs/robustness_analysis.png`*
