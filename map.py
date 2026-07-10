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

# App initialisieren
app = dash.Dash(__name__)

# Datensatz laden
df = pd.read_csv('./wetter_zugverspaetungen_1200.csv')

# App Layout definieren
app.layout = [
    html.Div(children="Geladener Datensatz", style={'fontSize': '18px', 'marginBottom': '10px'}),

    # AG Grid Tabelle zur interaktiven Ansicht der Daten
    dag.AgGrid(
        rowData=df.to_dict(orient='records'),
        columnDefs=[{"field": i} for i in df.columns],
        defaultColDef={"resizable": True, "sortable": True, "filter": True},
        style={"height": 400, "width": "100%", "marginBottom": "20px"}
    ),

    # Eingabebereich für das Modell
    html.Div([
        html.H1(children="Random Forest Einstellungen", style={'fontSize': '24px'}),
        html.P(
            "Gib den Namen einer Spalte ein, die du vorhersagen möchtest (z.B. 'Wetterlage' für Klassifikation oder 'Distanz_KM' für Regression):"),

        dcc.Input(
            id='user-input-dependent-variable',
            type='text',
            placeholder="Bitte gebe hier die abhängige Variable ein",
            style={'width': '300px', 'padding': '8px', 'marginRight': '10px'}
        ),
        html.Button('Modell trainieren', id='submit-button', n_clicks=0, style={'padding': '8px 15px'})
    ], style={'backgroundColor': '#f9f9f9', 'padding': '20px', 'borderRadius': '5px'}),

    html.Br(),

    # Lade-Animation (Spinner) zeigt sich, während das Modell rechnet
    dcc.Loading(
        id="loading-output",
        type="circle",
        children=html.Div(id='output-container', style={'fontWeight': 'bold', 'marginTop': '10px'})
    )
]


# --- DASH CALLBACK ---
@app.callback(
    Output('output-container', 'children'),
    Input('submit-button', 'n_clicks'),
    State('user-input-dependent-variable', 'value'),
    prevent_initial_call=True
)
def configurate_random_forest(n_clicks, text_value):
    if not text_value:
        return html.P("Bitte gib eine gültige Variable aus dem Datensatz ein!", style={'color': 'orange'})

    try:
        # Startet das Training und holt die fertigen Layout-Elemente/Diagramme ab
        visualisierung = train_random_forest(dependent_variable=text_value)
        return html.Div([
            html.P(f"✓ Modell erfolgreich trainiert für Zielvariable: '{text_value}' (Durchlauf #{n_clicks})",
                   style={'color': 'green', 'fontSize': '16px'}),
            visualisierung
        ])
    except ValueError as e:
        # Fängt falsche Eingaben ab und zeigt sie im Browser an
        return html.P(f"❌ Fehler: {str(e)}", style={'color': 'red'})


# --- HILFSFUNKTIONEN FÜR MACHINE LEARNING ---

def configure_features(features: list[str], dependent_variable: str) -> tuple[list[str], str]:
    features_without_dependent_variable: list[str] = features.copy()
    try:
        features_without_dependent_variable.remove(dependent_variable)
        return features_without_dependent_variable, dependent_variable
    except ValueError:
        raise ValueError(f"Feature '{dependent_variable}' ist nicht in der vordefinierten Feature-Liste vorhanden.")


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


def train_random_forest(dependent_variable: str):
    # Datensatz frisch einlesen
    df_ml = pd.read_csv('./wetter_zugverspaetungen_1200.csv')

    # Feature Engineering aus dem Datum extrahieren
    df_ml['Geplante_Abfahrt'] = pd.to_datetime(df_ml['Geplante_Abfahrt'])
    df_ml['Abfahrt_Monat'] = df_ml['Geplante_Abfahrt'].dt.month
    df_ml['Abfahrt_Stunde'] = df_ml['Geplante_Abfahrt'].dt.hour
    df_ml['Abfahrt_Wochentag'] = df_ml['Geplante_Abfahrt'].dt.dayofweek

    # Definition aller potenziellen Features im verarbeiteten Datensatz
    features = ['Start_Ort', 'End_Ort', 'Distanz_KM', 'Wetterlage', 'Verkehrsdichte', 'Abfahrt_Monat', 'Abfahrt_Stunde',
                'Abfahrt_Wochentag','Ziel_Verspaetet','Verspaetung_Minuten']
    cat_features = ['Start_Ort', 'End_Ort', 'Wetterlage', 'Verkehrsdichte', 'Ziel_Verspaetet']
    num_features = ['Distanz_KM', 'Abfahrt_Monat', 'Abfahrt_Stunde', 'Abfahrt_Wochentag','Verspaetung_Minuten']

    # Zielvariable aus den Eingabe-Features entfernen, damit das Modell nicht schummelt
    if dependent_variable in cat_features:
        cat_features.remove(dependent_variable)
    elif dependent_variable in num_features:
        num_features.remove(dependent_variable)
    else:
        raise ValueError(
            f"Zielvariable '{dependent_variable}' existiert nicht in den Features oder kann nicht verwendet werden.")

    # X-Matrix vorbereiten
    features_without_dependent_variable, dependent_variable = configure_features(features, dependent_variable)
    X = df_ml[features_without_dependent_variable]

    # Präprozessor für One-Hot-Encoding aufbauen
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_features),
            ('num', 'passthrough', num_features)
        ])

    # --- FALL 1: KLASSIFIKATION (Kategoriale Zielvariable) ---
    if dependent_variable in ['Start_Ort', 'End_Ort', 'Wetterlage', 'Verkehrsdichte']:
        y_clf = df_ml[dependent_variable]
        clf_tuple = classification_training(preprocessor=preprocessor, y_clf=y_clf, X=X)
        pipeline, X_train, X_test, y_train, y_test = clf_tuple

        # Feature-Wichtigkeiten extrahieren
        ohe_cols = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_features)
        all_features = list(ohe_cols) + num_features
        importances = pipeline.named_steps['classifier'].feature_importances_

        feature_imp_df = pd.DataFrame({'Feature': all_features, 'Wichtigkeit': importances})
        feature_imp_df = feature_imp_df.sort_values(by='Wichtigkeit', ascending=True).tail(10)  # Top 10

        # HTML Balkendiagramm erzeugen (Blau für Klassifikation)
        fig = px.bar(
            feature_imp_df,
            x='Wichtigkeit',
            y='Feature',
            orientation='h',
            title='Top 10 Feature Importances (Welche Variable erklärt die Klassifikation am besten?)',
            template='plotly_white'
        )
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
        return dcc.Graph(figure=fig)

    # --- FALL 2: REGRESSION (Numerische Zielvariable) ---
    else:
        y_reg = df_ml[dependent_variable]
        reg_tuple = regression_training(preprocessor=preprocessor, y_reg=y_reg, X=X)
        pipeline, X_train, X_test, y_train, y_test = reg_tuple

        # 1. Metriken berechnen
        preds = pipeline.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # HTML Tabelle für Regressionsmetriken erzeugen
        fig_table = go.Figure(data=[go.Table(
            header=dict(values=['Metrik / Kennzahl', 'Errechneter Wert für Testdaten'],
                        fill_color='#1f77b4',
                        align='left',
                        font=dict(color='white', size=14)),
            cells=dict(values=[['R² Score (Erklärte Varianz)', 'RMSE (Durchschnittlicher Abweichungsfehler)'],
                               [f"{r2:.4f}", f"{rmse:.2f}"]],
                       fill_color='#f5f6f9',
                       align='left',
                       font=dict(size=12))
        )])
        fig_table.update_layout(title='Modell-Performance (Regression)', margin=dict(l=0, r=0, t=40, b=10), height=180)

        # 2. Feature-Wichtigkeiten für die Regression extrahieren
        ohe_cols = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_features)
        all_features = list(ohe_cols) + num_features
        importances = pipeline.named_steps['regressor'].feature_importances_

        feature_imp_df = pd.DataFrame({'Feature': all_features, 'Wichtigkeit': importances})
        feature_imp_df = feature_imp_df.sort_values(by='Wichtigkeit', ascending=True).tail(10)  # Top 10

        # HTML Balkendiagramm für Regression erzeugen (Grün zur optischen Unterscheidung)
        fig_bar = px.bar(
            feature_imp_df,
            x='Wichtigkeit',
            y='Feature',
            orientation='h',
            title='Top 10 Feature Importances (Welche Variable erklärt die Regressions-Zielvariable am besten?)',
            template='plotly_white',
            color_discrete_sequence=['#2ca02c']
        )
        fig_bar.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))

        # Kombiniere beide Diagramme (Tabelle + Balkendiagramm) in einer Div
        return html.Div([
            dcc.Graph(figure=fig_table),
            html.Br(),
            dcc.Graph(figure=fig_bar)
        ])


if __name__ == '__main__':
    # Server starten
    app.run(debug=True)