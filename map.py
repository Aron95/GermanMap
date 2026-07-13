from typing import Any
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_ag_grid as dag
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from datetime import datetime

# App initialisieren
app = dash.Dash(__name__)

# Datensatz laden (Stelle sicher, dass die Datei im selben Ordner liegt!)
df = pd.read_csv('./wetter_zugverspaetungen_1200.csv')

# Globale Variablen, um das Modell nach dem Trainieren zwischenzuspeichern
trained_pipeline = None
current_dependent_variable = None
trained_features_list = []

features = ['Start_Ort', 'End_Ort', 'Distanz_KM', 'Wetterlage', 'Verkehrsdichte', 'Abfahrt_Monat', 'Abfahrt_Stunde',
            'Abfahrt_Wochentag', 'Ziel_Verspaetet', 'Verspaetung_Minuten']

# App Layout definieren
app.layout = [
    html.Div(children="Geladener Datensatz", style={'fontSize': '18px', 'marginBottom': '10px'}),

    # AG Grid Tabelle zur interaktiven Ansicht der Daten
    dag.AgGrid(
        rowData=df.to_dict(orient='records'),
        columnDefs=[{"field": i} for i in df.columns],
        dashGridOptions={
            "enableCellTextSelection": True,
            "ensureDomOrder": True,
        },
        defaultColDef={"resizable": True, "sortable": True, "filter": True},
        style={"height": 400, "width": "100%", "marginBottom": "20px"}
    ),

    # Eingabebereich für das Modell
    html.Div([
        html.H1(children="Random Forest Einstellungen", style={'fontSize': '24px'}),
        html.P(
            "Gib den Namen einer Spalte ein, die du vorhersagen möchtest (z.B. 'Verspaetung_Minuten' für deine Minutenschätzung):"),

        dcc.Input(
            id='user-input-dependent-variable',
            type='text',
            value='Verspaetung_Minuten',  # Direkt vorausgefüllt für die Regression
            placeholder="Bitte gebe hier die abhängige Variable ein",
            style={'width': '300px', 'padding': '8px', 'marginRight': '10px'}
        ),
        html.Button('Modell trainieren', id='submit-button', n_clicks=0, style={'padding': '8px 15px'})
    ], style={'backgroundColor': '#f9f9f9', 'padding': '20px', 'borderRadius': '5px'}),

    html.Br(),

    # Lade-Animation für das Training
    dcc.Loading(
        id="loading-output",
        type="circle",
        children=html.Div(id='output-container', style={'fontWeight': 'bold', 'marginTop': '10px'})
    ),

    html.Br(),
    html.Hr(),
    html.Br(),

    html.Div([
        html.H2("Live-Vorhersage für eine Zugfahrt"),
        html.P("Hinweis: Trainiere zuerst oben das Modell mit 'Verspaetung_Minuten', um hier die Minuten zu schätzen."),

        html.Div([
            html.Label("Start Ort:"),
            dcc.Input(id='Start-Ort-variable', type='text', value='Berlin', placeholder="z.B. Berlin"),

            html.Label("End Ort:", style={'marginLeft': '20px'}),
            dcc.Input(id='End-Ort-variable', type='text', value='München', placeholder="z.B. München"),

            html.Label("Distanz (KM):", style={'marginLeft': '20px'}),
            dcc.Input(id='Distanz-KM-variable', type='number', value=500, placeholder="z.B. 500"),
        ], style={'marginBottom': '15px'}),

        html.Div([
            html.Label("Wetterlage:"),
            dcc.Input(id='Wetterlage-variable', type='text', value='Schnee',
                      placeholder="z.B. Regnerisch, Schnee, Klar"),

            html.Label("Verkehrsdichte:", style={'marginLeft': '20px'}),
            dcc.Input(id='Verkehrsdichte-variable', type='text', value='Hoch',
                      placeholder="z.B. Niedrig, Normal, Hoch"),

            html.Label("Abfahrtsdatum:", style={'marginLeft': '20px'}),
            dcc.DatePickerSingle(
                id='Datum-picker-variable',
                min_date_allowed=datetime(2020, 1, 1),
                max_date_allowed=datetime(2030, 12, 31),
                date=datetime(2026, 7, 12)
            ),

            html.Label("Uhrzeit (Stunde 0-23):", style={'marginLeft': '20px'}),
            dcc.Input(id='Abfahrt-Stunde-variable', type='number', value=14, min=0, max=23, style={'width': '60px'}),
        ], style={'marginBottom': '20px'}),

        html.Button('Zugverspätung schätzen', id='predict-button', n_clicks=0,
                    style={'padding': '10px 20px', 'backgroundColor': '#2ca02c', 'color': 'white', 'border': 'none',
                           'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '16px'}),

    ], style={'backgroundColor': '#eef2f7', 'padding': '20px', 'borderRadius': '5px'}),

    # Lade-Animation für die Vorhersage
    dcc.Loading(
        id="loading-output-prediction",
        type="circle",
        children=html.Div(id='output-prediction-container', style={'marginTop': '15px'})
    ),
]


# --- CALLBACK FÜR PROGNOSE (REAGIERT NUR AUF BUTTON-KLICK) ---
@app.callback(
    Output('output-prediction-container', 'children'),
    Input('predict-button', 'n_clicks'),  # Löst den Callback aus
    State('Start-Ort-variable', 'value'),  # Werte werden nur als State übergeben
    State('End-Ort-variable', 'value'),
    State('Distanz-KM-variable', 'value'),
    State('Wetterlage-variable', 'value'),
    State('Verkehrsdichte-variable', 'value'),
    State('Datum-picker-variable', 'date'),
    State('Abfahrt-Stunde-variable', 'value'),
    prevent_initial_call=True
)
def predict_current_ride(n_clicks, start, ende, distanz, wetter, verkehr, datum_str, stunde):
    global trained_pipeline, current_dependent_variable, trained_features_list

    # Sicherstellen, dass der Button geklickt wurde und ein Modell da ist
    if n_clicks == 0 or trained_pipeline is None:
        return html.Div("⚠️ Bitte trainiere zuerst oben das Modell, bevor du eine Schätzung startest.",
                        style={'color': 'orange', 'fontWeight': 'bold'})

    # Prüfen, ob das Modell für Minuten trainiert wurde
    if current_dependent_variable != 'Verspaetung_Minuten':
        return html.Div(
            f"⚠️ Das Modell ist aktuell auf '{current_dependent_variable}' trainiert. Bitte trainiere das Modell oben neu mit 'Verspaetung_Minuten'.",
            style={'color': 'red', 'fontWeight': 'bold'})

    try:
        # Features aus dem Datum ziehen
        datum_obj = pd.to_datetime(datum_str)
        monat = datum_obj.month
        wochentag = datum_obj.dayofweek

        # DataFrame für Scikit-Learn erstellen
        input_data = pd.DataFrame([{
            'Start_Ort': str(start),
            'End_Ort': str(ende),
            'Distanz_KM': float(distanz) if distanz is not None else 0.0,
            'Wetterlage': str(wetter),
            'Verkehrsdichte': str(verkehr),
            'Abfahrt_Monat': int(monat),
            'Abfahrt_Stunde': int(stunde) if stunde is not None else 12,
            'Abfahrt_Wochentag': int(wochentag),
            'Ziel_Verspaetet': 0  # Dummy-Wert
        }])

        # Features auf die Trainings-Spalten reduzieren
        input_data_filtered = input_data[trained_features_list]

        # Vorhersage berechnen
        prediction = trained_pipeline.predict(input_data_filtered)[0]
        prediction = max(0.0, prediction)  # Keine negative Verspätung zulassen

        return html.Div([
            html.H3("Ergebnis der Live-Schätzung:"),
            html.Div(f"Voraussichtliche Verspätung: {prediction:.1f} Minuten",
                     style={'fontSize': '22px', 'color': '#2ca02c', 'fontWeight': 'bold'}),
            html.P(f"Berechnet für die Strecke {start} ➔ {ende} bei '{wetter}' und '{verkehr}' Verkehr.",
                   style={'color': 'gray', 'fontSize': '13px', 'marginTop': '5px'})
        ], style={'padding': '15px', 'borderLeft': '5px solid #2ca02c', 'backgroundColor': '#f2fcf2',
                  'borderRadius': '4px'})

    except Exception as e:
        return html.Div(f"❌ Fehler bei der Vorhersage: {str(e)}", style={'color': 'red'})


# --- DASH CALLBACK FÜR TRAINING ---
@app.callback(
    Output('output-container', 'children'),
    Input('submit-button', 'n_clicks'),
    State('user-input-dependent-variable', 'value'),
    prevent_initial_call=True
)
def configurate_random_forest(n_clicks, text_value):
    global trained_pipeline, current_dependent_variable, trained_features_list

    if not text_value or text_value not in features:
        return html.P(f"Bitte gib eine gültige Variable ein! Mögliche Werte: {', '.join(features)}",
                      style={'color': 'orange'})

    try:
        visualisierung, model_pipeline, used_features = train_random_forest_and_return_model(
            dependent_variable=text_value)

        trained_pipeline = model_pipeline
        current_dependent_variable = text_value
        trained_features_list = used_features

        return html.Div([
            html.P(f"✓ Modell erfolgreich trainiert für Zielvariable: '{text_value}' (Durchlauf #{n_clicks})",
                   style={'color': 'green', 'fontSize': '16px'}),
            visualisierung
        ])
    except ValueError as e:
        return html.P(f"❌ Fehler: {str(e)}", style={'color': 'red'})


# --- TRAINING LOGIK ---

def configure_features(features: list[str], dependent_variable: str) -> tuple[list[str], str]:
    features_without_dependent_variable: list[str] = features.copy()
    try:
        features_without_dependent_variable.remove(dependent_variable)
        return features_without_dependent_variable, dependent_variable
    except ValueError:
        raise ValueError(f"Feature '{dependent_variable}' ist nicht vorhanden.")


def regression_training(preprocessor: ColumnTransformer, y_reg, X) -> tuple:
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    reg_pipeline.fit(X_train_r, y_train_r)
    return reg_pipeline, X_train_r, X_test_r, y_train_r, y_test_r


def classification_training(preprocessor: ColumnTransformer, y_clf, X) -> tuple:
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    clf_pipeline.fit(X_train_c, y_train_c)
    return clf_pipeline, X_train_c, X_test_c, y_train_c, y_test_c


def train_random_forest_and_return_model(dependent_variable: str):
    df_ml = pd.read_csv('./wetter_zugverspaetungen_1200.csv')

    if 'Geplante_Abfahrt' in df_ml.columns:
        df_ml['Geplante_Abfahrt'] = pd.to_datetime(df_ml['Geplante_Abfahrt'])
        df_ml['Abfahrt_Monat'] = df_ml['Geplante_Abfahrt'].dt.month
        df_ml['Abfahrt_Stunde'] = df_ml['Geplante_Abfahrt'].dt.hour
        df_ml['Abfahrt_Wochentag'] = df_ml['Geplante_Abfahrt'].dt.dayofweek

    cat_features = ['Start_Ort', 'End_Ort', 'Wetterlage', 'Verkehrsdichte', 'Ziel_Verspaetet']
    num_features = ['Distanz_KM', 'Abfahrt_Monat', 'Abfahrt_Stunde', 'Abfahrt_Wochentag', 'Verspaetung_Minuten']

    if dependent_variable in cat_features:
        cat_features.remove(dependent_variable)
    elif dependent_variable in num_features:
        num_features.remove(dependent_variable)

    features_without_dependent_variable, dependent_variable = configure_features(features, dependent_variable)
    X = df_ml[features_without_dependent_variable]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_features),
            ('num', 'passthrough', num_features)
        ])

    # --- FALL 1: KLASSIFIKATION ---
    if dependent_variable in ['Start_Ort', 'End_Ort', 'Wetterlage', 'Verkehrsdichte', 'Ziel_Verspaetet']:
        y_clf = df_ml[dependent_variable]
        pipeline, X_train, X_test, y_train, y_test = classification_training(preprocessor, y_clf, X)

        feature_imp_df = pd.DataFrame({
            'Feature': list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(
                cat_features)) + num_features,
            'Wichtigkeit': pipeline.named_steps['classifier'].feature_importances_
        }).sort_values(by='Wichtigkeit').tail(10)

        fig = px.bar(feature_imp_df, x='Wichtigkeit', y='Feature', orientation='h',
                     title='Top 10 Feature Importances (Klassifikation)', template='plotly_white')
        return dcc.Graph(figure=fig), pipeline, list(X.columns)

    # --- FALL 2: REGRESSION ---
    else:
        y_reg = df_ml[dependent_variable]
        pipeline, X_train, X_test, y_train, y_test = regression_training(preprocessor, y_reg, X)

        preds = pipeline.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        fig_table = go.Figure(data=[go.Table(
            header=dict(values=['Metrik / Kennzahl', 'Wert'], fill_color='#1f77b4', font=dict(color='white')),
            cells=dict(values=[['R² Score', 'RMSE (Fehler in Min.)'], [f"{r2:.4f}", f"{rmse:.2f}"]],
                       fill_color='#f5f6f9')
        )])
        fig_table.update_layout(title='Modell-Performance (Regression)', height=180, margin=dict(l=0, r=0, t=40, b=10))

        feature_imp_df = pd.DataFrame({
            'Feature': list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(
                cat_features)) + num_features,
            'Wichtigkeit': pipeline.named_steps['regressor'].feature_importances_
        }).sort_values(by='Wichtigkeit').tail(10)

        fig_bar = px.bar(feature_imp_df, x='Wichtigkeit', y='Feature', orientation='h',
                         title='Top 10 Feature Importances (Regression)', template='plotly_white',
                         color_discrete_sequence=['#2ca02c'])
        fig_bar.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))

        return html.Div([dcc.Graph(figure=fig_table), html.Br(), dcc.Graph(figure=fig_bar)]), pipeline, list(X.columns)


if __name__ == '__main__':
    app.run(debug=True, port=8051)