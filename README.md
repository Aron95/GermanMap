# Dahboard

Interaktives [Dash](https://dash.plotly.com/)-Dashboard zur Analyse von Wetter- und
Zugverspätungsdaten. Die Anwendung lädt einen Testdatensatz, zeigt ihn in einer
interaktiven Tabelle an und trainiert auf Wunsch ein **Random-Forest**-Modell
(Klassifikation oder Regression) für eine frei wählbare Zielvariable.

## Features

- Interaktive Datentabelle (sortier- und filterbar) via `dash-ag-grid`
- Random-Forest-Training für eine beliebige Spalte des Datensatzes
  - **Klassifikation** bei kategorialen Zielvariablen (z. B. `Wetterlage`)
  - **Regression** bei numerischen Zielvariablen (z. B. `Distanz_KM`)
- Automatisches Feature-Engineering aus dem Abfahrtsdatum (Monat, Stunde, Wochentag)
- Visualisierung der Top-10-Feature-Importances und Regressionsmetriken (R², RMSE)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Nutzung

```bash
python map.py
```

Anschließend das Dashboard im Browser unter http://127.0.0.1:8050 öffnen.
Eine Zielvariable (Spaltenname) eingeben und **Modell trainieren** klicken.

## Daten

- `wetter_zugverspaetungen_1200.csv` – Testdatensatz mit Wetter- und Verspätungsdaten
- `logistik_prognose_testdaten.csv` – zusätzliche Logistik-Testdaten
- `germany.geo.json` – GeoJSON der deutschen Bundesländer
