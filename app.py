# -*- coding: utf-8 -*-
"""
app.py
Streamlit app: Predicción de Primas y Siniestros (Agosto-Diciembre 2025)
- Carga desde Google Sheet pública (URL/ID + GID) o permite subir CSV/XLSX
- Normaliza columnas (FECHA, VALOR, HOMOLOGACIÓN, Primas/Siniestros, COMPAÑÍA, CIUDAD, RAMO, DEPARTAMENTO)
- Parser robusto de números (puntos/comas como miles/decimales)
- Entrena XGBoost (si está disponible) por serie (HOMO x TIPO) con fallback a HistGradientBoostingRegressor
- Añade herramienta de diagnóstico para inspeccionar discrepancias en sumas
"""
import warnings
warnings.filterwarnings("ignore")

import re
from io import BytesIO, StringIO
from urllib.parse import urlparse, parse_qs
import requests
import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ML libs
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

from sklearn.ensemble import HistGradientBoostingRegressor

# ---------------- Config ----------------
st.set_page_config(page_title="Primas & Siniestros — Forecast XGBoost", layout="wide")
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1VljNnZtRPDA3TkTUP6w8AviZCPIfILqe/edit?usp=sharing"
DEFAULT_GID = "293107109"

TARGET_START = pd.Timestamp("2025-08-01")
TARGET_MONTHS = pd.date_range(start=TARGET_START, periods=5, freq="MS")  # Aug-Dec 2025
TARGET_MONTHS_STR = [d.strftime("%b-%Y") for d in TARGET_MONTHS]

# ---------------- Sheet helpers ----------------
def extract_sheet_id(url_or_id: str) -> str:
    if not isinstance(url_or_id, str):
        return ""
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_\-]{10,}", s):
        return s
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", s)
    if m:
        return m.group(1)
    parsed = urlparse(s)
    qs = parse_qs(parsed.query)
    if 'id' in qs:
        return qs['id'][0]
    return s

def extract_gid(url: str) -> str:
    if not isinstance(url, str):
        return ""
    m = re.search(r"[?&]gid=(\d+)", url)
    if m:
        return m.group(1)
    m2 = re.search(r"/edit#gid=(\d+)", url)
    if m2:
        return m2.group(1)
    return ""

def export_csv_url(sheet_id: str, gid: str = "0") -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

def try_load_public_sheet(sheet_input: str = None, gid: str = None, timeout: int = 20):
    """
    Try multiple public endpoints (export by gid, pub CSV, gviz).
    Returns DataFrame or raises RuntimeError with details.
    """
    if not sheet_input:
        raise RuntimeError("No sheet_input provided.")
    sheet_id = extract_sheet_id(sheet_input)
    if not sheet_id:
        raise RuntimeError("No sheet id could be extracted.")

    candidates = []
    gid_use = gid if gid and gid.strip() else "0"
    candidates.append(export_csv_url(sheet_id, gid_use))
    candidates.append(export_csv_url(sheet_id, "0"))
    candidates.append(f"https://docs.google.com/spreadsheets/d/{sheet_id}/pub?output=csv")
    candidates.append(f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv")

    last_status = None
    errors = []
    for url in candidates:
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
                        errors.append((url, f"CSV parse failed: {e}"))
                        continue
                else:
                    errors.append((url, f"200 OK but content not CSV-like (len {len(text)})"))
            else:
                errors.append((url, f"HTTP {resp.status_code}"))
        except Exception as e:
            errors.append((url, f"Exception: {type(e).__name__}: {e}"))
            continue

    msg = "Could not load public sheet. Tried URLs and errors:\n" + "\n".join([f"- {u} -> {e}" for u, e in errors])
    raise RuntimeError(msg + f"\nLast status: {last_status}")

# ---------------- Parsing numbers robust ----------------
def parse_number_co(series: pd.Series) -> pd.Series:
    """
    Robust parser for numbers with dots/commas.
    Handles:
      - "1.234.567,89" -> 1234567.89
      - "1,234,567.89" -> 1234567.89
      - "1.249" -> 1249 (if part after dot has 3 digits it's thousands separator)
      - "1249.5" -> 1249.5 (dot decimal)
      - "1249" -> 1249
    """
    import numpy as _np
    s = series.astype(str).fillna("").str.strip()
    s = s.str.replace(r"[^\d\-,\.]", "", regex=True)

    def _parse_one(x: str):
        if x is None:
            return _np.nan
        x = str(x).strip()
        if x == "" or x.lower() in ("nan", "none"):
            return _np.nan
        has_dot = "." in x
        has_comma = "," in x
        try:
            if has_dot and has_comma:
                # decide by last separator
                if x.rfind(",") > x.rfind("."):
                    x2 = x.replace(".", "").replace(",", ".")
                else:
                    x2 = x.replace(",", "")
            elif has_comma and not has_dot:
                parts = x.split(",")
                if len(parts[-1]) == 3 and len(parts) > 1:
                    x2 = x.replace(",", "")
                else:
                    x2 = x.replace(",", ".")
            elif has_dot and not has_comma:
                parts = x.split(".")
                if len(parts[-1]) == 3 and len(parts) > 1:
                    x2 = x.replace(".", "")
                else:
                    x2 = x
            else:
                x2 = x
            if x2 in ("", "-", "--"):
                return _np.nan
            return float(x2)
        except Exception:
            digits = re.sub(r"[^\d\-\.]", "", x)
            try:
                return float(digits) if digits != "" else _np.nan
            except Exception:
                return _np.nan

    parsed = s.map(_parse_one)
    return pd.to_numeric(parsed, errors="coerce")

# ---------------- Normalization ----------------
def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df_raw_backup = df.copy()
    df.columns = [c.strip() for c in df.columns]
    colmap = {}
    orig_val_col = None
    for c in df.columns:
        cn = c.strip().lower()
        if 'homolog' in cn:
            colmap[c] = 'HOMO'
        elif cn in ('año','ano','year'):
            colmap[c] = 'ANIO'
        elif 'compa' in cn or 'compañ' in cn:
            colmap[c] = 'COMPANIA'
        elif 'ciudad' in cn:
            colmap[c] = 'CIUDAD'
        elif 'ram' in cn:
            colmap[c] = 'RAMO'
        elif 'primas' in cn and 'siniest' in cn:
            colmap[c] = 'TIPO'
        elif 'primas/siniestros' in cn or cn == 'primas/siniestros':
            colmap[c] = 'TIPO'
        elif 'fecha' in cn:
            colmap[c] = 'FECHA'
        elif 'valor' in cn:
            colmap[c] = 'VALOR'
            orig_val_col = c
        elif 'depart' in cn:
            colmap[c] = 'DEPARTAMENTO'
    df = df.rename(columns=colmap)

    # keep raw text of value column for diagnosis
    if orig_val_col:
        df['VALOR_RAW'] = df_raw_backup[orig_val_col].astype(str)
    else:
        for alt in ['Valor_Mensual','Valor Mensual','VALOR_MENSUAL','valor_mensual','valor mensual']:
            if alt in df_raw_backup.columns:
                df['VALOR_RAW'] = df_raw_backup[alt].astype(str)
                orig_val_col = alt
                break
        else:
            df['VALOR_RAW'] = ""

    # FECHA cleaning
    if 'FECHA' in df.columns:
        ser = df['FECHA'].astype(str).fillna("").str.replace('\u202f',' ').str.replace('\xa0',' ')
        ser = ser.str.replace(r'(?i)\s*(a\.?\s*m\.?|p\.?\s*m\.?|am|pm)\b', '', regex=True)
        extracted = ser.str.extract(r'(\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4})', expand=False)
        to_parse = extracted.fillna(ser)
        df['FECHA'] = pd.to_datetime(to_parse, dayfirst=True, errors='coerce')
    else:
        df['FECHA'] = pd.NaT
    df['FECHA'] = df['FECHA'].dt.to_period("M").dt.to_timestamp()

    # VALOR numeric
    if 'VALOR' in df.columns:
        df['VALOR'] = parse_number_co(df['VALOR'])
    else:
        for alt in ['Valor_Mensual','Valor Mensual','VALOR_MENSUAL','valor_mensual','valor mensual']:
            if alt in df.columns:
                df['VALOR'] = parse_number_co(df[alt])
                break
        else:
            df['VALOR'] = pd.to_numeric(df.get('VALOR', pd.Series(dtype=float)), errors='coerce')

    # TIPO normalize
    if 'TIPO' in df.columns:
        df['TIPO'] = df['TIPO'].astype(str).str.strip().str.capitalize()
    else:
        found = False
        for c in df.columns:
            sample = df[c].astype(str).str.lower().head(30).tolist()
            if any('primas' in s for s in sample) or any('siniest' in s for s in sample) or any('siniestro' in s for s in sample):
                df['TIPO'] = df[c].astype(str).str.strip().str.capitalize()
                found = True
                break
        if not found:
            df['TIPO'] = 'Primas'

    # Ensure text fields
    for c in ['HOMO','COMPANIA','CIUDAD','RAMO','DEPARTAMENTO']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
        else:
            df[c] = "UNKNOWN"

    if 'ANIO' not in df.columns:
        df['ANIO'] = df['FECHA'].dt.year

    total = len(df)
    valid_dates = int(df['FECHA'].notna().sum())
    st.sidebar.write(f"Registros total: {total:,} — FECHA válida: {valid_dates:,}")
    if valid_dates < total:
        st.sidebar.markdown("Ejemplos filas sin FECHA válida (revisa formato):")
        bad = df[df['FECHA'].isna()].head(6)
        if not bad.empty:
            st.sidebar.dataframe(bad)

    keep = ['ANIO','FECHA','HOMO','COMPANIA','CIUDAD','RAMO','TIPO','VALOR','VALOR_RAW','DEPARTAMENTO']
    keep = [c for c in keep if c in df.columns]
    return df[keep].dropna(subset=['FECHA']).copy()

# ---------------- Features & Model ----------------
def make_lag_features(s: pd.Series, max_lag: int = 12):
    s = s.sort_index().asfreq("MS").fillna(0.0)
    df = pd.DataFrame({'y': s})
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['m_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['m_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    for l in range(1, max_lag+1):
        df[f'lag_{l}'] = df['y'].shift(l)
    df['roll_3'] = df['y'].rolling(3, min_periods=1).mean().shift(1)
    df['roll_6'] = df['y'].rolling(6, min_periods=1).mean().shift(1)
    df['roll_12'] = df['y'].rolling(12, min_periods=1).mean().shift(1)
    return df

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred) + 1e-9)
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100

def train_xgb_on_series(s: pd.Series, steps: int = 5, conservative_factor: float = 1.0, max_lag: int = 12):
    """
    Robust training & iterative forecasting.
    Returns preds (list length=steps) and val_smape (float or nan).
    """
    s = s.sort_index().asfreq("MS").fillna(0.0)
    if s.dropna().sum() == 0 or len(s.dropna()) == 0:
        return [0.0]*steps, np.nan

    if len(s.dropna()) < 12:
        monthly_avg = s.groupby(s.index.month).mean()
        preds = []
        last = s.index.max()
        for i in range(1, steps+1):
            nextm = (last + pd.DateOffset(months=i)).month
            val = float(monthly_avg.get(nextm, s.mean()))
            preds.append(max(0.0, val * conservative_factor))
        return preds, np.nan

    df_feats = make_lag_features(s, max_lag=max_lag).dropna()
    if df_feats.empty or 'y' not in df_feats:
        avg = s.tail(12).mean()
        return [max(0.0, avg*conservative_factor)]*steps, np.nan

    n = len(df_feats)
    val_months = min(6, max(3, int(n * 0.15)))
    train_df = df_feats.iloc[:-val_months].copy()
    val_df = df_feats.iloc[-val_months:].copy()

    # sanitize y
    train_df['y'] = train_df['y'].apply(lambda x: 0.0 if (pd.isna(x) or x < 0) else x)
    val_df['y'] = val_df['y'].apply(lambda x: 0.0 if (pd.isna(x) or x < 0) else x)

    train_df = train_df[np.isfinite(train_df['y'])]
    val_df = val_df[np.isfinite(val_df['y'])]

    if len(train_df) < 6:
        avg = s.tail(12).mean()
        st.warning(f"Serie corta para entrenamiento (n_train={len(train_df)}). Usando promedio fallback.")
        return [max(0.0, avg * conservative_factor)]*steps, np.nan

    X_train_df = train_df.drop(columns=['y']).copy()
    X_val_df = val_df.drop(columns=['y']).copy()

    # fill NaNs in X with column mean (or 0)
    for col in X_train_df.columns:
        col_mean = X_train_df[col].replace([np.inf, -np.inf], np.nan).mean()
        if np.isnan(col_mean):
            col_mean = 0.0
        X_train_df[col] = X_train_df[col].replace([np.inf, -np.inf], np.nan).fillna(col_mean)
    for col in X_val_df.columns:
        if col in X_train_df.columns:
            col_mean = X_train_df[col].mean()
        else:
            col_mean = 0.0
        X_val_df[col] = X_val_df[col].replace([np.inf, -np.inf], np.nan).fillna(col_mean)

    X_train = X_train_df.values.astype(np.float32)
    X_val = X_val_df.values.astype(np.float32)
    y_train = np.log1p(train_df['y'].values.astype(np.float32))
    y_val = np.log1p(val_df['y'].values.astype(np.float32)) if len(val_df)>0 else np.array([])

    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
        avg = s.tail(12).mean()
        st.warning("NaN/Inf en X_train o y_train detectados; usando fallback promedio.")
        return [max(0.0, avg * conservative_factor)]*steps, np.nan

    feature_cols = list(train_df.drop(columns=['y']).columns)

    model = None
    if XGBOOST_AVAILABLE:
        try:
            model = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=1,
                verbosity=0
            )
            if len(y_val) > 0:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
            else:
                model.fit(X_train, y_train)
        except Exception as e:
            st.warning(f"XGBoost fit error: {str(e)[:200]}. Fallback to sklearn model.")
            model = None

    if model is None:
        try:
            model_sk = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=6, random_state=42)
            model_sk.fit(X_train, y_train)
            model = model_sk
        except Exception as e:
            st.error(f"Fallback model training failed: {e}")
            avg = s.tail(12).mean()
            return [max(0.0, avg * conservative_factor)]*steps, np.nan

    # validation smape
    try:
        if len(y_val) > 0:
            y_val_pred_log = model.predict(X_val)
            y_val_pred = np.expm1(y_val_pred_log)
            y_val_true = np.expm1(y_val)
            val_sm = smape(y_val_true, y_val_pred)
        else:
            val_sm = np.nan
    except Exception:
        val_sm = np.nan

    # iterative forecasting
    preds = []
    hist = s.copy()
    for i in range(1, steps+1):
        next_date = hist.index.max() + pd.DateOffset(months=1)
        row = {}
        row['year'] = next_date.year
        row['month'] = next_date.month
        row['m_sin'] = np.sin(2 * np.pi * row['month'] / 12.0)
        row['m_cos'] = np.cos(2 * np.pi * row['month'] / 12.0)
        for l in range(1, max_lag+1):
            lag_date = next_date - pd.DateOffset(months=l)
            row[f'lag_{l}'] = hist.get(lag_date, np.nan)
        row['roll_3'] = hist.tail(3).mean() if len(hist) >= 1 else 0.0
        row['roll_6'] = hist.tail(6).mean() if len(hist) >= 1 else 0.0
        row['roll_12'] = hist.tail(12).mean() if len(hist) >= 1 else 0.0

        feat_df = pd.DataFrame([row])
        for c in feature_cols:
            if c not in feat_df.columns:
                feat_df[c] = 0.0
        feat_df = feat_df[feature_cols]
        feat_df = feat_df.fillna(feat_df.mean(axis=1).iloc[0])

        try:
            y_log_pred = model.predict(feat_df.values.astype(np.float32))
            y_pred = float(np.expm1(y_log_pred[0])) if np.isscalar(y_log_pred[0]) or len(np.atleast_1d(y_log_pred))>0 else float(np.expm1(y_log_pred))
        except Exception:
            y_pred = float(hist.tail(3).mean() if len(hist)>=3 else hist.mean())
        y_pred = max(0.0, y_pred * conservative_factor)
        preds.append(y_pred)
        hist.loc[next_date] = y_pred

    return preds, float(val_sm) if not np.isnan(val_sm) else np.nan

# ----------------- UI ----------------
st.title("Primas & Siniestros — Forecast (Aug–Dec 2025)")
st.markdown("Carga la Google Sheet (pública) o sube un archivo. Usa la herramienta de diagnóstico si las sumas no coinciden.")

# Sidebar: data source
st.sidebar.header("Origen de datos")
sheet_url = st.sidebar.text_input("Google Sheet URL o Sheet ID", value=DEFAULT_SHEET_URL)
gid_input = st.sidebar.text_input("GID (opcional)", value=DEFAULT_GID)
if st.sidebar.button("Forzar recarga desde Google Sheets"):
    st.cache_data.clear()
    st.experimental_rerun()

df_raw = pd.DataFrame()
load_error = None
try:
    extracted_gid = extract_gid(sheet_url) or gid_input or DEFAULT_GID
    df_candidate = try_load_public_sheet(sheet_input=sheet_url, gid=extracted_gid)
    if not df_candidate.empty:
        df_raw = df_candidate
        st.sidebar.success("Google Sheet cargada correctamente.")
except Exception as e:
    load_error = e
    st.sidebar.warning("No se pudo cargar automáticamente la Google Sheet pública. Puedes subir el archivo manualmente.")
    st.sidebar.text(str(e)[:400])

st.sidebar.markdown("---")
st.sidebar.markdown("O sube un archivo")
uploaded_csv = st.sidebar.file_uploader("Subir CSV", type=["csv"])
uploaded_xlsx = st.sidebar.file_uploader("Subir Excel (.xlsx)", type=["xlsx"])
if uploaded_csv:
    try:
        df_raw = pd.read_csv(StringIO(uploaded_csv.getvalue().decode("utf-8")))
        st.sidebar.success("CSV cargado.")
    except Exception as e:
        st.sidebar.error("Error leyendo CSV.")
        st.sidebar.exception(e)
if uploaded_xlsx:
    try:
        df_raw = pd.read_excel(BytesIO(uploaded_xlsx.getvalue()))
        st.sidebar.success("Excel cargado.")
    except Exception as e:
        st.sidebar.error("Error leyendo Excel.")
        st.sidebar.exception(e)

if df_raw.empty:
    if st.sidebar.button("Usar datos de ejemplo"):
        dates = pd.date_range(start='2020-01-01', end='2025-07-01', freq='MS')
        data = []
        comps = ['ESTADO','MAPFRE','LIBERTY']
        ciudades = ['BOGOTA','MEDELLIN','CALI']
        ramos = ['VIDRIOS','INCENDIO','ROBO','SOAT']
        homos = ['GENERALES','ESPECIALES']
        for d in dates:
            for comp in comps:
                for c in ciudades:
                    for r in ramos:
                        data.append({'HOMOLOGACIÓN': np.random.choice(homos), 'Año': d.year, 'COMPAÑÍA': comp, 'CIUDAD': c, 'RAMOS': r, 'Primas/Siniestros': 'Primas', 'FECHA': d, 'Valor_Mensual': max(0, np.random.normal(50000, 20000)), 'DEPARTAMENTO':'ANTIOQUIA'})
                        data.append({'HOMOLOGACIÓN': np.random.choice(homos), 'Año': d.year, 'COMPAÑÍA': comp, 'CIUDAD': c, 'RAMOS': r, 'Primas/Siniestros': 'Siniestros', 'FECHA': d, 'Valor_Mensual': max(0, np.random.normal(8000, 3000)), 'DEPARTAMENTO':'ANTIOQUIA'})
        df_raw = pd.DataFrame(data)
        st.sidebar.success("Datos de ejemplo cargados.")

if df_raw.empty:
    st.warning("No hay datos cargados. Carga la Google Sheet pública o sube un archivo.")
    st.stop()

# show raw columns and sample
st.sidebar.markdown("### Columnas crudas detectadas")
st.sidebar.write(df_raw.columns.tolist())
st.write("### Muestra cruda (primeras filas)")
st.dataframe(df_raw.head(6))

# Normalize
try:
    df = normalize_input(df_raw)
except Exception as e:
    st.error("Error al normalizar datos.")
    st.exception(e)
    st.stop()

st.markdown("### Muestra normalizada (primeras filas)")
st.dataframe(df.head(6))

# Diagnostic UI
st.sidebar.markdown("---")
st.sidebar.markdown("### Diagnóstico rápido")
diag_ramo = st.sidebar.text_input("Ramo a inspeccionar (diagnóstico)", value="SOAT")
diag_tipo = st.sidebar.selectbox("Tipo", options=["Primas", "Siniestros"], index=0)
diag_year = st.sidebar.number_input("Año", min_value=2000, max_value=2100, value=2021, step=1)
diag_month = st.sidebar.selectbox("Mes (num)", options=list(range(1,13)), index=11)
if st.sidebar.button("Ejecutar diagnóstico"):
    df_diag = df.copy()
    mask_ramo = df_diag['RAMO'].str.upper().str.contains(diag_ramo.upper(), na=False)
    mask_tipo = df_diag['TIPO'].str.lower() == diag_tipo.lower()
    mask_fecha = (df_diag['FECHA'].dt.year == int(diag_year)) & (df_diag['FECHA'].dt.month == int(diag_month))
    sel = df_diag[mask_ramo & mask_tipo & mask_fecha]
    st.markdown(f"## Diagnóstico: {diag_ramo} · {diag_tipo} · {diag_month:02d}/{diag_year}")
    st.write(f"Filas encontradas: {len(sel)}")
    if sel.empty:
        st.info("No se encontraron filas con esos filtros.")
    else:
        cols_show = [c for c in ['FECHA','HOMO','COMPANIA','CIUDAD','RAMO','TIPO','VALOR_RAW','VALOR'] if c in sel.columns]
        st.dataframe(sel[cols_show].sort_values(by=['CIUDAD','FECHA']).head(500))
        st.write("Suma VALOR (parsed):", float(sel['VALOR'].sum()))
        st.write("Valores crudos ejemplo (hasta 50):")
        st.write(sel['VALOR_RAW'].astype(str).unique()[:50].tolist())
        st.write("Filas con VALOR NaN:", int(sel['VALOR'].isna().sum()))
        if 'CIUDAD' in sel.columns:
            st.write("Suma por CIUDAD:")
            st.dataframe(sel.groupby('CIUDAD')['VALOR'].sum().sort_values(ascending=False).head(200))

# Filters
st.sidebar.markdown("---")
st.sidebar.header("Filtros para predicción")
company_opts = ["TODAS"] + sorted(df['COMPANIA'].dropna().unique().tolist())
city_opts = ["TODAS"] + sorted(df['CIUDAD'].dropna().unique().tolist())
ramo_opts = ["TODAS"] + sorted(df['RAMO'].dropna().unique().tolist())

company_sel = st.sidebar.selectbox("Compañía", company_opts)
city_sel = st.sidebar.selectbox("Ciudad", city_opts)
ramo_sel = st.sidebar.selectbox("Ramo", ramo_opts)
conservative_pct = st.sidebar.slider("Ajuste conservador (%)", -20.0, 20.0, 0.0, step=0.5)
conservative_factor = 1.0 + conservative_pct/100.0

# Apply filters
df_f = df.copy()
if company_sel != "TODAS":
    df_f = df_f[df_f['COMPANIA'] == company_sel]
if city_sel != "TODAS":
    df_f = df_f[df_f['CIUDAD'] == city_sel]
if ramo_sel != "TODAS":
    df_f = df_f[df_f['RAMO'] == ramo_sel]

if df_f.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

st.write(f"Registros después de filtros: {len(df_f):,} — Periodo: {df_f['FECHA'].min().date()} to {df_f['FECHA'].max().date()}")

# Global plot
ts_pr = df_f[df_f['TIPO'].str.lower() == 'primas'].groupby('FECHA')['VALOR'].sum().sort_index()
ts_si = df_f[df_f['TIPO'].str.lower() == 'siniestros'].groupby('FECHA')['VALOR'].sum().sort_index()

fig = go.Figure()
if not ts_pr.empty:
    fig.add_trace(go.Scatter(x=ts_pr.index, y=ts_pr.values, name='Primas'))
if not ts_si.empty:
    fig.add_trace(go.Scatter(x=ts_si.index, y=ts_si.values, name='Siniestros'))
fig.update_layout(title="Serie histórica agregada", yaxis_title="Valor mensual (COP)", xaxis=dict(type="date", rangeslider=dict(visible=True)))
st.plotly_chart(fig, use_container_width=True)

# Predictions by HOMO x TIPO
st.markdown("## Predicciones por HOMOLOGACIÓN (Agosto-Diciembre 2025)")
groups = df_f.groupby(['HOMO','TIPO'])

primas_rows = []
sini_rows = []
total = len(groups)
pbar = st.progress(0)
idx = 0

for (homo, tipo), g in groups:
    idx += 1
    pbar.progress(int(idx/total*100))
    s = g.set_index('FECHA').sort_index().groupby('FECHA')['VALOR'].sum()
    s = s.asfreq("MS").fillna(0.0)
    preds, val_sm = train_xgb_on_series(s, steps=len(TARGET_MONTHS), conservative_factor=conservative_factor)
    row = {'HOMOLOGACIÓN': homo}
    for dt, p in zip(TARGET_MONTHS, preds):
        row[dt.strftime("%b-%Y")] = round(p, 0)
    row['SMAPE_val'] = round(val_sm, 2) if not np.isnan(val_sm) else None
    if tipo.lower().startswith('p'):
        primas_rows.append(row)
    else:
        sini_rows.append(row)

pbar.empty()

primas_df = pd.DataFrame(primas_rows).set_index('HOMOLOGACIÓN') if primas_rows else pd.DataFrame()
sini_df = pd.DataFrame(sini_rows).set_index('HOMOLOGACIÓN') if sini_rows else pd.DataFrame()

st.subheader("Primas — predicción (COP)")
if not primas_df.empty:
    st.dataframe(primas_df.fillna(0).style.format(lambda v: f"${int(v):,}".replace(",", ".") if pd.notna(v) and v!=0 else "-"), use_container_width=True)
else:
    st.info("No hay predicciones de primas.")

st.subheader("Siniestros — predicción (COP)")
if not sini_df.empty:
    st.dataframe(sini_df.fillna(0).style.format(lambda v: f"${int(v):,}".replace(",", ".") if pd.notna(v) and v!=0 else "-"), use_container_width=True)
else:
    st.info("No hay predicciones de siniestros.")

# Aggregate by city
st.markdown("### Agregado por CIUDAD — Predicción (Agosto-Diciembre 2025)")
if not primas_df.empty:
    homo_city = df_f.groupby('HOMO')['CIUDAD'].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "UNKNOWN")
    tmp = primas_df.reset_index().merge(homo_city.rename('CIUDAD'), left_on='HOMOLOGACIÓN', right_index=True, how='left')
    agg_city_pr = tmp.groupby('CIUDAD')[TARGET_MONTHS_STR].sum().round(0)
    st.dataframe(agg_city_pr.style.format(lambda v: f"${int(v):,}".replace(",", ".")), use_container_width=True)
if not sini_df.empty:
    homo_city = df_f.groupby('HOMO')['CIUDAD'].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "UNKNOWN")
    tmp = sini_df.reset_index().merge(homo_city.rename('CIUDAD'), left_on='HOMOLOGACIÓN', right_index=True, how='left')
    agg_city_si = tmp.groupby('CIUDAD')[TARGET_MONTHS_STR].sum().round(0)
    st.dataframe(agg_city_si.style.format(lambda v: f"${int(v):,}".replace(",", ".")), use_container_width=True)

# Download
if (not primas_df.empty) or (not sini_df.empty):
    with BytesIO() as out:
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            if not primas_df.empty:
                primas_df.to_excel(writer, sheet_name="Primas_HOMO")
            if not sini_df.empty:
                sini_df.to_excel(writer, sheet_name="Siniestros_HOMO")
            try:
                if 'agg_city_pr' in locals():
                    agg_city_pr.to_excel(writer, sheet_name="Primas_CIUDAD")
                if 'agg_city_si' in locals():
                    agg_city_si.to_excel(writer, sheet_name="Siniestros_CIUDAD")
            except Exception:
                pass
        data = out.getvalue()
    st.download_button("⬇️ Descargar predicciones (Excel)", data=data, file_name="predicciones_Ago-Dic-2025.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.sidebar.markdown("---")
st.sidebar.write(f"Última fecha en datos: {df['FECHA'].max().date()}")
st.sidebar.write(f"Series históricas (meses): {df['FECHA'].nunique()}")
