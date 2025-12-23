# -*- coding: utf-8 -*-
"""
app.py
Predicción de Primas y Siniestros (Colombia) — prototipo con XGBoost / fallback
- Soporta carga desde:
  * Google Sheet pública (por URL o Sheet ID + optional GID / sheet name)
  * Google Sheet privada mediante Service Account (st.secrets["gcp_service_account"])
  * Subida de CSV local
- Salida: predicciones de Agosto-Diciembre 2025 por HOMOLOGACIÓN y vistas por ciudad/competidores
"""
import warnings
warnings.filterwarnings("ignore")

import re
import requests
from urllib.parse import quote
from io import BytesIO, StringIO
import json
import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
from typing import Tuple, Dict, List

# Model imports (XGBoost preferido)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# fallback
from sklearn.ensemble import HistGradientBoostingRegressor

# Optional imports for Service Account loader
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except Exception:
    GSPREAD_AVAILABLE = False

# ----------------- Config -----------------
st.set_page_config(page_title="Primas & Siniestros - Forecast XGB", layout="wide")
TARGET_MONTHS = pd.date_range(start="2025-08-01", end="2025-12-01", freq="MS")
TARGET_MONTHS_STR = [d.strftime("%b-%Y") for d in TARGET_MONTHS]

# ----------------- Utilities -----------------
def parse_input_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # Try multiple possible column names
    col_map = {}
    for c in df.columns:
        low = c.strip().lower()
        if low in ("fecha","date"):
            col_map[c] = "FECHA"
        if low in ("valor_mensual","valor mensual","valor"):
            col_map[c] = "Valor_Mensual"
        if low in ("primas/siniestros","primas_siniestros","tipo","primas o siniestros"):
            col_map[c] = "Primas/Siniestros"
        if low in ("homologación","homologacion","homolog"):
            col_map[c] = "HOMOLOGACIÓN"
        if low in ("compañía","compania","company"):
            col_map[c] = "COMPAÑÍA"
        if low in ("ciudad","city"):
            col_map[c] = "CIUDAD"
        if low in ("ramos","ramo"):
            col_map[c] = "RAMOS"
        if low in ("departamento","dept"):
            col_map[c] = "DEPARTAMENTO"
    df = df.rename(columns=col_map)
    # Parse FECHA robustly (dayfirst)
    if 'FECHA' in df.columns:
        df['FECHA'] = pd.to_datetime(df['FECHA'].astype(str).str.replace('\u202f',' '), dayfirst=True, errors='coerce')
        # normalize to month start
        df['FECHA'] = df['FECHA'].dt.to_period("M").dt.to_timestamp()
    else:
        raise RuntimeError("No se encontró columna 'FECHA' en el dataset. Asegúrate de que exista y tenga formato legible.")
    # Ensure numeric
    if 'Valor_Mensual' in df.columns:
        df['Valor_Mensual'] = pd.to_numeric(df['Valor_Mensual'], errors='coerce').fillna(0.0)
    else:
        raise RuntimeError("No se encontró columna 'Valor_Mensual' en el dataset.")
    # Standardize category
    if 'Primas/Siniestros' in df.columns:
        df['Primas/Siniestros'] = df['Primas/Siniestros'].astype(str).str.strip().str.capitalize()
    else:
        raise RuntimeError("No se encontró columna 'Primas/Siniestros' en el dataset.")
    # Fill missing HOMOLOGACIÓN with 'SIN-HOMO'
    if 'HOMOLOGACIÓN' not in df.columns:
        df['HOMOLOGACIÓN'] = "SIN-HOMO"
    else:
        df['HOMOLOGACIÓN'] = df['HOMOLOGACIÓN'].astype(str).str.strip().replace({'nan': 'SIN-HOMO'})
    # Ensure COMPAÑÍA, CIUDAD, RAMOS, DEPARTAMENTO exist
    for c in ['COMPAÑÍA','CIUDAD','RAMOS','DEPARTAMENTO']:
        if c not in df.columns:
            df[c] = "UNKNOWN"
        else:
            df[c] = df[c].astype(str).str.strip()
    return df

def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(['FECHA','HOMOLOGACIÓN','Primas/Siniestros','COMPAÑÍA','CIUDAD','RAMOS'], dropna=False)['Valor_Mensual'].sum().reset_index()
    return agg

def make_time_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=idx)
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['m_sin'] = np.sin(2 * np.pi * (df['month'] / 12.0))
    df['m_cos'] = np.cos(2 * np.pi * (df['month'] / 12.0))
    return df

def prepare_series(df_group: pd.Series) -> pd.Series:
    s = df_group.sort_index().asfreq("MS").fillna(0.0)
    return s

def train_forecast_xgb(series: pd.Series, steps: int = 5, min_months: int = 12, conservative_factor: float = 1.0) -> List[float]:
    if series.isna().all() or len(series.dropna()) == 0:
        return [0.0] * steps
    s = series.copy()
    if len(s.dropna()) < max(6, min_months//2):
        base = float(s.tail(3).mean()) if len(s) >= 3 else float(s.mean())
        return [max(0.0, base * conservative_factor)] * steps

    df = pd.DataFrame({'y': s})
    df = df.reset_index().rename(columns={'index':'ds'})
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.set_index('ds')
    X_time = make_time_features(df.index)
    df = pd.concat([df, X_time], axis=1)

    lags = [1,2,3,6,12]
    for l in lags:
        df[f'lag_{l}'] = df['y'].shift(l)
    df['roll3'] = df['y'].rolling(3, min_periods=1).mean().shift(1)
    df['roll6'] = df['y'].rolling(6, min_periods=1).mean().shift(1)
    df = df.dropna(subset=['y'])
    df = df.dropna()
    if df.empty:
        base = float(s.tail(3).mean()) if len(s) >= 3 else float(s.mean())
        return [max(0.0, base * conservative_factor)] * steps

    feature_cols = [c for c in df.columns if c != 'y']
    X_train = df[feature_cols].values
    y_train = df['y'].values

    if XGBOOST_AVAILABLE:
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1)
    else:
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)

    try:
        model.fit(X_train, y_train)
    except Exception:
        base = float(s.tail(3).mean()) if len(s) >= 3 else float(s.mean())
        return [max(0.0, base * conservative_factor)] * steps

    preds = []
    hist = s.copy()
    for step in range(steps):
        next_idx = hist.index.max() + pd.offsets.MonthBegin(1)
        feat = make_time_features(pd.DatetimeIndex([next_idx]))
        for l in lags:
            lag_date = next_idx - pd.DateOffset(months=l)
            feat[f'lag_{l}'] = hist.get(lag_date, np.nan)
        feat['roll3'] = hist.tail(3).mean() if len(hist) >= 1 else 0.0
        feat['roll6'] = hist.tail(6).mean() if len(hist) >= 1 else 0.0
        feat = feat.fillna(feat.mean(axis=1).iloc[0])
        # ensure feature ordering
        if all(c in feat.columns for c in feature_cols):
            Xp = feat[feature_cols].values
        else:
            Xp = feat.values
        try:
            p = float(model.predict(Xp)[0]) * conservative_factor
        except Exception:
            p = float(np.nanmean(y_train)) * conservative_factor
        p = max(0.0, p)
        preds.append(p)
        hist.loc[next_idx] = p
    return preds

# ----------------- Google Sheets loaders -----------------
def extract_sheet_id(maybe_url: str) -> str:
    if maybe_url is None:
        return ""
    maybe_url = maybe_url.strip()
    if re.fullmatch(r"[A-Za-z0-9_\-]{10,}", maybe_url):
        return maybe_url
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", maybe_url)
    if m:
        return m.group(1)
    m2 = re.search(r"[?&]id=([a-zA-Z0-9-_]+)", maybe_url)
    if m2:
        return m2.group(1)
    return maybe_url

def try_load_public_sheet(sheet_input: str, sheet_name: str = None, gid: str = None, timeout: int = 10) -> pd.DataFrame:
    sheet_id = extract_sheet_id(sheet_input)
    if not sheet_id:
        raise RuntimeError("No se pudo extraer el sheet_id del valor proporcionado.")
    tried = []
    errors = []
    candidates = []
    gid_use = gid if gid and gid.strip() else "0"
    candidates.append(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid_use}")
    if sheet_name and sheet_name.strip():
        candidates.append(f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(sheet_name.strip())}")
    candidates.append(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")
    candidates.append(f"https://docs.google.com/spreadsheets/d/{sheet_id}/pub?output=csv")

    last_status = None
    for url in candidates:
        tried.append(url)
        try:
            resp = requests.get(url, timeout=timeout)
            last_status = resp.status_code
            if resp.status_code == 200:
                text = resp.text
                if ("," in text) and ("\n" in text):
                    try:
                        df = pd.read_csv(StringIO(text))
                        return df
                    except Exception as e:
                        errors.append((url, f"200 OK pero fallo parse CSV: {e}"))
                        continue
                else:
                    errors.append((url, f"200 OK pero el contenido no parece CSV (length {len(text)})"))
                    continue
            else:
                errors.append((url, f"HTTP {resp.status_code}"))
        except Exception as e:
            errors.append((url, f"Exception: {type(e).__name__}: {e}"))
            continue

    msg_lines = ["No se pudo leer la Google Sheet pública. Se intentaron las siguientes URLs:"]
    for u in tried:
        msg_lines.append(f"- {u}")
    msg_lines.append("Errores obtenidos (resumen):")
    for u, e in errors:
        msg_lines.append(f"- {u} -> {e}")
    msg_lines.append("")
    msg_lines.append("Sugerencias:")
    msg_lines.append("1) Asegúrate de que la hoja sea accesible públicamente (Compartir -> 'Cualquiera con el enlace' -> 'Ver').")
    msg_lines.append("2) Si la hoja no funciona con export CSV, publica la hoja en: Archivo -> Publicar en la web -> 'Hoja' -> 'CSV' y prueba la URL:")
    msg_lines.append(f"   https://docs.google.com/spreadsheets/d/{sheet_id}/pub?output=csv")
    msg_lines.append("3) Si la hoja tiene varias pestañas, obtiene el GID de la pestaña activa (mira &gid= en la URL) y úsalo.")
    full_msg = "\n".join(msg_lines)
    raise RuntimeError(full_msg + f"\nÚltimo código HTTP: {last_status}")

def load_gsheet_service_account(sheet_id: str, sheet_name: str = None, creds_info: dict = None) -> pd.DataFrame:
    if not GSPREAD_AVAILABLE:
        raise RuntimeError("gspread/google-auth no están instalados. Añade 'gspread' y 'google-auth' a requirements.txt")
    if creds_info is None:
        # try st.secrets
        creds_info = st.secrets.get("gcp_service_account")
        if creds_info is None:
            # try env var
            env_val = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
            if env_val:
                try:
                    creds_info = json.loads(env_val)
                except Exception:
                    raise RuntimeError("La variable de entorno GCP_SERVICE_ACCOUNT_JSON no contiene JSON válido.")
            else:
                raise RuntimeError("No se proporcionaron credenciales de Service Account (st.secrets['gcp_service_account'] o GCP_SERVICE_ACCOUNT_JSON).")
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly","https://www.googleapis.com/auth/drive.readonly"]
    creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    client = gspread.authorize(creds)
    book = client.open_by_key(sheet_id)
    if sheet_name:
        try:
            ws = book.worksheet(sheet_name)
        except Exception:
            ws = book.get_worksheet(0)
    else:
        ws = book.get_worksheet(0)
    data = ws.get_all_records()
    df = pd.DataFrame(data)
    return df

# ----------------- Caching helpers -----------------
@st.cache_data(show_spinner=False)
def build_aggregates_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_input_dates(df)
    agg = aggregate_monthly(df)
    return agg

@st.cache_data(show_spinner=False)
def compute_predictions_table_from_agg(agg_df: pd.DataFrame, company_filter: str, city_filter: str, ramo_filter: str, conservative_factor: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = agg_df.copy()
    if company_filter and company_filter != "TODAS":
        df = df[df['COMPAÑÍA'] == company_filter]
    if city_filter and city_filter != "TODAS":
        df = df[df['CIUDAD'] == city_filter]
    if ramo_filter and ramo_filter != "TODAS":
        df = df[df['RAMOS'] == ramo_filter]

    primas_rows = []
    sinies_rows = []
    homos = sorted(df['HOMOLOGACIÓN'].dropna().unique())

    for h in homos:
        for target in ['Primas','Siniestros']:
            sub = df[(df['HOMOLOGACIÓN']==h) & (df['Primas/Siniestros']==target)].set_index('FECHA').sort_index()['Valor_Mensual']
            s = prepare_series(sub)
            preds = train_forecast_xgb(s, steps=len(TARGET_MONTHS), conservative_factor=conservative_factor)
            row = {'HOMOLOGACIÓN': h}
            for i, d in enumerate(TARGET_MONTHS):
                row[d.strftime("%Y-%m-%d")] = preds[i]
            if target == 'Primas':
                primas_rows.append(row)
            else:
                sinies_rows.append(row)

    primas_df = pd.DataFrame(primas_rows).set_index('HOMOLOGACIÓN') if primas_rows else pd.DataFrame(columns=['HOMOLOGACIÓN']).set_index('HOMOLOGACIÓN')
    sinies_df = pd.DataFrame(sinies_rows).set_index('HOMOLOGACIÓN') if sinies_rows else pd.DataFrame(columns=['HOMOLOGACIÓN']).set_index('HOMOLOGACIÓN')

    col_map = {d.strftime("%Y-%m-%d"): d.strftime("%b-%Y") for d in TARGET_MONTHS}
    primas_df = primas_df.rename(columns=col_map)
    sinies_df = sinies_df.rename(columns=col_map)

    return primas_df, sinies_df

# ----------------- Streamlit UI -----------------
st.title("Forecast Primas & Siniestros — XGBoost (Prototipo)")

# Default public Google Sheet (user provided earlier)
DEFAULT_SHEET_ID = "1VljNnZtRPDA3TkTUP6w8AviZCPIfILqe"

st.sidebar.header("Fuente de datos")
source = st.sidebar.radio("Selecciona fuente", ["Google Sheet pública", "Google Sheet (service account)", "Subir CSV"], index=0)

csv_text = None
df_loaded = None

if source == "Google Sheet pública":
    st.sidebar.markdown("Introduce el enlace completo o el Sheet ID. Si la hoja es pública la app la cargará automáticamente.")
    sheet_input = st.sidebar.text_input("URL o Sheet ID", value=DEFAULT_SHEET_ID)
    sheet_name_input = st.sidebar.text_input("Nombre de la pestaña (opcional)", value="")
    gid_input = st.sidebar.text_input("GID de la pestaña (opcional)", value="0")
    if st.sidebar.button("Cargar Google Sheet pública"):
        try:
            with st.spinner("Cargando Google Sheet pública..."):
                df_loaded = try_load_public_sheet(sheet_input, sheet_name_input if sheet_name_input else None, gid=gid_input if gid_input else "0")
                st.success("Hoja cargada correctamente.")
                st.write(df_loaded.head(3))
        except Exception as e:
            st.error("No se pudo cargar la Google Sheet pública. Revisa el mensaje de error abajo.")
            st.exception(e)
            df_loaded = None

elif source == "Google Sheet (service account)":
    st.sidebar.markdown("Usa Service Account. Guarda el JSON en st.secrets['gcp_service_account'] (recomendado) o sube el archivo aquí.")
    sheet_input = st.sidebar.text_input("URL o Sheet ID (service account)", value=DEFAULT_SHEET_ID)
    sheet_name_input = st.sidebar.text_input("Nombre de la pestaña (opcional)", value="")
    creds_upload = st.sidebar.file_uploader("Sube JSON credenciales (opcional, si no está en st.secrets)", type=["json"])
    if st.sidebar.button("Cargar Google Sheet (service account)"):
        try:
            creds_info = None
            if creds_upload:
                creds_info = json.loads(creds_upload.getvalue().decode("utf-8"))
            # st.secrets read is inside loader if creds_info is None
            with st.spinner("Cargando Google Sheet con service account..."):
                df_loaded = load_gsheet_service_account(extract_sheet_id(sheet_input), sheet_name=sheet_name_input if sheet_name_input else None, creds_info=creds_info)
                st.success("Hoja cargada correctamente vía service account.")
                st.write(df_loaded.head(3))
        except Exception as e:
            st.error("No se pudo cargar la Google Sheet con service account.")
            st.exception(e)
            df_loaded = None

else:  # Subir CSV
    uploaded = st.sidebar.file_uploader("Sube CSV con datos", type=["csv"], accept_multiple_files=False)
    if uploaded:
        csv_text = uploaded.getvalue().decode("utf-8")
        try:
            df_loaded = pd.read_csv(StringIO(csv_text))
            st.sidebar.success("CSV cargado.")
            st.write(df_loaded.head(3))
        except Exception as e:
            st.sidebar.error("Error leyendo CSV subido.")
            st.exception(e)
            df_loaded = None
    else:
        st.sidebar.info("Sube un CSV si no quieres usar Google Sheet.")

if df_loaded is None:
    st.warning("No hay datos cargados todavía. Usa la barra lateral para cargar desde Google Sheets (pública o service account) o subir un CSV.")
    st.stop()

# Precompute aggregates (cached)
try:
    with st.spinner("Procesando dataset..."):
        agg = build_aggregates_from_dataframe(df_loaded)
except Exception as e:
    st.error("Error procesando el dataset después de la carga.")
    st.exception(e)
    st.stop()

# Sidebar filters
st.sidebar.header("Filtros")
companies = ["TODAS"] + sorted(agg['COMPAÑÍA'].dropna().unique().tolist()) if 'COMPAÑÍA' in agg.columns else ["TODAS"]
cities = ["TODAS"] + sorted(agg['CIUDAD'].dropna().unique().tolist()) if 'CIUDAD' in agg.columns else ["TODAS"]
ramos = ["TODAS"] + sorted(agg['RAMOS'].dropna().unique().tolist()) if 'RAMOS' in agg.columns else ["TODAS"]

company_sel = st.sidebar.selectbox("Compañía", companies)
city_sel = st.sidebar.selectbox("Ciudad", cities)
ramo_sel = st.sidebar.selectbox("Ramo", ramos)

conservative_pct = st.sidebar.slider("Ajuste conservador (%)", -20.0, 20.0, 0.0, step=0.5)
conservative_factor = 1.0 + conservative_pct/100.0

# Tabs / Pages
tabs = st.tabs(["Página 1 — Resumen por HOMOLOGACIÓN", "Página 2 — Ciudades objetivo", "Página 3 — Competidores"])

# --- PAGE 1 ---
with tabs[0]:
    st.header("Histórico y predicciones por HOMOLOGACIÓN")
    df_chart = agg.copy()
    if company_sel != "TODAS":
        df_chart = df_chart[df_chart['COMPAÑÍA'] == company_sel]
    if city_sel != "TODAS":
        df_chart = df_chart[df_chart['CIUDAD'] == city_sel]
    if ramo_sel != "TODAS":
        df_chart = df_chart[df_chart['RAMOS'] == ramo_sel]

    ts_primas = df_chart[df_chart['Primas/Siniestros']=='Primas'].groupby('FECHA')['Valor_Mensual'].sum().sort_index()
    ts_sinies = df_chart[df_chart['Primas/Siniestros']=='Siniestros'].groupby('FECHA')['Valor_Mensual'].sum().sort_index()

    fig = go.Figure()
    if not ts_primas.empty:
        fig.add_trace(go.Scatter(x=ts_primas.index, y=ts_primas.values, name='Primas (Hist)'))
    if not ts_sinies.empty:
        fig.add_trace(go.Scatter(x=ts_sinies.index, y=ts_sinies.values, name='Siniestros (Hist)'))
    fig.update_layout(title="Serie histórica (agrupada)", yaxis_title="Valor Mensual (COP)", xaxis=dict(type="date", rangeslider=dict(visible=True)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tabla de predicciones por HOMOLOGACIÓN (agosto-diciembre 2025)")
    with st.spinner("Entrenando y generando predicciones..."):
        primas_table, sinies_table = compute_predictions_table_from_agg(agg, company_sel, city_sel, ramo_sel, conservative_factor=conservative_factor)

    if not primas_table.empty:
        display_primas = primas_table.copy().fillna(0.0).round(0).astype(float)
        display_primas = display_primas.applymap(lambda x: f"${int(x):,}".replace(",", "."))
        st.subheader("Primas — Predicción (Aug-Dec 2025)")
        st.dataframe(display_primas, use_container_width=True)
    else:
        st.info("No se generaron predicciones de primas para los filtros actuales.")

    if not sinies_table.empty:
        display_sin = sinies_table.copy().fillna(0.0).round(0).astype(float)
        display_sin = display_sin.applymap(lambda x: f"${int(x):,}".replace(",", "."))
        st.subheader("Siniestros — Predicción (Aug-Dec 2025)")
        st.dataframe(display_sin, use_container_width=True)
    else:
        st.info("No se generaron predicciones de siniestros para los filtros actuales.")

    if (not primas_table.empty) or (not sinies_table.empty):
        with BytesIO() as buf:
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                if not primas_table.empty:
                    primas_table.to_excel(writer, sheet_name="Primas_Pred", index=True)
                if not sinies_table.empty:
                    sinies_table.to_excel(writer, sheet_name="Siniestros_Pred", index=True)
            data = buf.getvalue()
        st.download_button("Descargar predicciones (Excel)", data=data, file_name="predicciones_primas_siniestros_2025_Ago-Dic.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- PAGE 2: Cities objetivo ---
TARGET_CITIES = ["BOGOTA","MEDELLIN","CALI","BUCARAMANGA","BARRANQUILLA","CARTAGENA","TUNJA"]
with tabs[1]:
    st.header("Ciudades objetivo")
    mult = st.multiselect("Ciudades a mostrar", options=TARGET_CITIES, default=[c for c in TARGET_CITIES if c in [x.upper() for x in cities if x != "TODAS"]])
    for c in mult:
        st.subheader(f"Ciudad: {c}")
        df_city = agg[agg['CIUDAD'].str.upper() == c.upper()]
        if not df_city.empty:
            ts_pr = df_city[df_city['Primas/Siniestros']=='Primas'].groupby('FECHA')['Valor_Mensual'].sum().sort_index()
            ts_si = df_city[df_city['Primas/Siniestros']=='Siniestros'].groupby('FECHA')['Valor_Mensual'].sum().sort_index()
            figc = go.Figure()
            if not ts_pr.empty:
                figc.add_trace(go.Scatter(x=ts_pr.index, y=ts_pr.values, name='Primas'))
            if not ts_si.empty:
                figc.add_trace(go.Scatter(x=ts_si.index, y=ts_si.values, name='Siniestros'))
            figc.update_layout(title=f"Histórico {c}", yaxis_title="Valor Mensual (COP)", xaxis=dict(type="date"))
            st.plotly_chart(figc, use_container_width=True)

            with st.spinner(f"Prediciendo para {c}..."):
                primas_city, sinies_city = compute_predictions_table_from_agg(agg, "TODAS", c, "TODAS", conservative_factor=conservative_factor)
            if not primas_city.empty:
                agg_pr_city = primas_city.astype(float).sum(axis=0).to_frame(name="Primas")
                agg_si_city = sinies_city.astype(float).sum(axis=0).to_frame(name="Siniestros")
                df_city_pred = pd.concat([agg_pr_city.T, agg_si_city.T])
                df_city_pred = df_city_pred.T.fillna(0.0).round(0).astype(float)
                df_city_pred = df_city_pred.applymap(lambda x: f"${int(x):,}".replace(",", "."))
                st.table(df_city_pred)
            else:
                st.info(f"No se generaron predicciones para {c}.")
        else:
            st.info(f"No hay datos para {c} con los filtros actuales.")

# --- PAGE 3: Competidores ---
COMPETIDORES = ["ESTADO","MAPFRE GENERALES","LIBERTY","AXA GENERALES","MUNDIAL","PREVISORA"]
with tabs[2]:
    st.header("Competidores")
    comp_sel = st.multiselect("Selecciona competidores", options=COMPETIDORES, default=COMPETIDORES)
    for comp in comp_sel:
        st.subheader(comp)
        df_comp = agg[agg['COMPAÑÍA'].str.upper() == comp.upper()]
        if not df_comp.empty:
            ts_pr = df_comp[df_comp['Primas/Siniestros']=='Primas'].groupby('FECHA')['Valor_Mensual'].sum().sort_index()
            ts_si = df_comp[df_comp['Primas/Siniestros']=='Siniestros'].groupby('FECHA')['Valor_Mensual'].sum().sort_index()
            figc = go.Figure()
            if not ts_pr.empty:
                figc.add_trace(go.Scatter(x=ts_pr.index, y=ts_pr.values, name='Primas'))
            if not ts_si.empty:
                figc.add_trace(go.Scatter(x=ts_si.index, y=ts_si.values, name='Siniestros'))
            figc.update_layout(title=f"Histórico {comp}", yaxis_title="Valor Mensual (COP)", xaxis=dict(type="date"))
            st.plotly_chart(figc, use_container_width=True)

            with st.spinner(f"Prediciendo para {comp}..."):
                primas_comp, sinies_comp = compute_predictions_table_from_agg(agg, comp, "TODAS", "TODAS", conservative_factor=conservative_factor)
            if not primas_comp.empty:
                agg_pr = primas_comp.astype(float).sum(axis=0).to_frame(name="Primas")
                agg_si = sinies_comp.astype(float).sum(axis=0).to_frame(name="Siniestros")
                df_comp_pred = pd.concat([agg_pr.T, agg_si.T])
                df_comp_pred = df_comp_pred.T.fillna(0.0).round(0).astype(float)
                df_comp_pred = df_comp_pred.applymap(lambda x: f"${int(x):,}".replace(",", "."))
                st.table(df_comp_pred)
            else:
                st.info(f"No se generaron predicciones para {comp}.")
        else:
            st.info(f"No hay datos para {comp} con los filtros actuales.")

st.markdown("---")
st.caption("Aplicación prototipo — modelos XGBoost / fallback. Predicciones: Agosto a Diciembre 2025.")
