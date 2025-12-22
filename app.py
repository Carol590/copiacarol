# -*- coding: utf-8 -*-
"""
Streamlit app: Industria — Nowcast & Forecast por Ciudad/Ramo (incluye proyección mes a mes 2026)
- Carga automáticamente la Hoja1 del spreadsheet (ID/GID definidos) usando la URL de export CSV.
  Si la hoja no es pública la app mostrará instrucciones (no hace upload).
- Normaliza columnas mínimas (FECHA, VALOR, TIPO_VALOR, CIUDAD, RAMO, COMPANIA, ESTADO, etc).
- Forecast por ciudad o por ramo (configurable). Calcula:
    * Nowcast / cierre estimado del año seleccionado por ciudad/ramo
    * Proyección mes-a-mes para 2026 por ciudad/ramo (y total industria)
    * Comparativa "Solo ESTADO" (mi empresa) vs Industria
- Descarga Excel con resultados.
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
from typing import Optional, Dict, List

import urllib.request
from urllib.error import HTTPError, URLError

# Time series models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# ---------- Config ----------
DEFAULT_SHEET_ID = "1VljNnZtRPDA3TkTUP6w8AviZCPIfILqe"
DEFAULT_GID = "293107109"
DEFAULT_SHEET_NAME = "Hoja1"

st.set_page_config(page_title="Industria · Forecast 2026 por Ciudad / Ramo", layout="wide")

# ---------- Utilities ----------
def export_csv_url(sheet_id: str, gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

def parse_number_co(series: pd.Series) -> pd.Series:
    s = series.astype(str).fillna("")
    s = s.str.replace(r"[^\d,.\-]", "", regex=True)
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def ensure_monthly(ts: pd.Series) -> pd.Series:
    ts = ts.asfreq("MS")
    ts = ts.interpolate(method="linear", limit_area="inside")
    return ts

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100

def sanitize_trailing_zeros(ts: pd.Series, ref_year: int) -> pd.Series:
    ts = ensure_monthly(ts.copy())
    year_series = ts[ts.index.year == ref_year]
    if year_series.empty:
        return ts.dropna()
    mask = (year_series[::-1] == 0)
    run, flag = [], True
    for v in mask:
        if flag and bool(v):
            run.append(True)
        else:
            flag = False
            run.append(False)
    trailing_zeros = pd.Series(run[::-1], index=year_series.index)
    ts.loc[trailing_zeros.index[trailing_zeros]] = np.nan
    if ts.last_valid_index() is not None:
        ts = ts.loc[:ts.last_valid_index()]
    return ts.dropna()

def split_series_excluding_partial_current(ts: pd.Series, ref_year: int, today_like: pd.Timestamp):
    ts = ensure_monthly(ts.copy())
    cur_m = pd.Timestamp(year=today_like.year, month=today_like.month, day=1)
    if len(ts) == 0:
        return ts, None, False
    end_of_month = (cur_m + pd.offsets.MonthEnd(0)).day
    if today_like.day < end_of_month:
        ts.loc[cur_m] = np.nan
        return ts.dropna(), cur_m, True
    return ts.dropna(), None, False

def fmt_cop(x):
    try:
        if pd.isna(x):
            return "-"
    except Exception:
        return "-"
    try:
        return "$" + f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return x

# ---------- Forecast helper ----------
def fit_forecast(ts_m: pd.Series, steps: int, eval_months:int=6, conservative_factor: float = 1.0):
    """Forecast historical series ts_m and return hist_df, fc_df, smape_last"""
    if steps < 1:
        steps = 1
    ts = ensure_monthly(ts_m.copy())
    if ts.empty:
        return pd.DataFrame(columns=["FECHA","Mensual","ACUM"]), pd.DataFrame(columns=["FECHA","Forecast_mensual","Forecast_acum","IC_lo","IC_hi"]), np.nan
    y = np.log1p(ts.replace(0, np.nan).dropna())  # log1p on nonzero series
    if y.empty:
        return pd.DataFrame({"FECHA":ts.index, "Mensual":ts.values}), pd.DataFrame(), np.nan
    smapes = []
    start = max(len(y)-eval_months, 12)
    if len(y) >= start+1:
        for t in range(start, len(y)):
            y_tr = y.iloc[:t]
            y_te = y.iloc[t:t+1]
            try:
                m = SARIMAX(y_tr, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
                r = m.fit(disp=False)
                p = r.get_forecast(steps=1).predicted_mean
            except Exception:
                r = ARIMA(y_tr, order=(1,1,1)).fit()
                p = r.get_forecast(steps=1).predicted_mean
            smapes.append(smape(np.expm1(y_te.values), np.expm1(p.values)))
    smape_last = float(np.mean(smapes)) if smapes else np.nan
    def _adj(arr):
        return np.expm1(arr) * conservative_factor
    try:
        m_full = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
        r_full = m_full.fit(disp=False)
        pred = r_full.get_forecast(steps=steps)
        mean = _adj(pred.predicted_mean)
        ci = np.expm1(pred.conf_int(alpha=0.05)) * conservative_factor
    except Exception:
        r_full = ARIMA(y, order=(1,1,1)).fit()
        pred = r_full.get_forecast(steps=steps)
        mean = _adj(pred.predicted_mean)
        ci = np.expm1(pred.conf_int(alpha=0.05)) * conservative_factor
    future_idx = pd.date_range(ts.index.max() + pd.offsets.MonthBegin(), periods=steps, freq="MS")
    hist_acum = ts.cumsum()
    forecast_acum = np.cumsum(mean) + (hist_acum.iloc[-1] if len(hist_acum) > 0 else 0.0)
    fc_df = pd.DataFrame({"FECHA": future_idx, "Forecast_mensual": mean.values.clip(min=0), "Forecast_acum": forecast_acum.values.clip(min=0)})
    # attach IC if ci available
    if hasattr(ci, 'iloc'):
        try:
            fc_df["IC_lo"] = ci.iloc[:,0].values.clip(min=0)
            fc_df["IC_hi"] = ci.iloc[:,1].values.clip(min=0)
        except Exception:
            fc_df["IC_lo"] = np.nan
            fc_df["IC_hi"] = np.nan
    hist_df = pd.DataFrame({"FECHA": ts.index, "Mensual": ts.values, "ACUM": hist_acum.values if len(ts) > 0 else []})
    return hist_df, fc_df, smape_last

def forecast_year_monthly_for_series(series: pd.Series, target_year: int = 2026, conservative_factor: float = 1.0):
    """
    Forecast monthly values for target_year (Jan..Dec) given historical monthly series.
    Returns a Series indexed by month start dates for the target year.
    """
    if series is None or series.empty:
        # return zeros for 12 months
        idx = pd.date_range(start=f"{target_year}-01-01", periods=12, freq="MS")
        return pd.Series([0.0]*12, index=idx)
    last = series.index.max()
    last_year = last.year
    last_month = last.month
    steps = (target_year - last_year) * 12 + (12 - last_month)
    # steps is number of months from month after last to Dec target_year inclusive
    steps = int(max(1, steps))
    hist_df, fc_df, _ = fit_forecast(series, steps=steps, eval_months=6, conservative_factor=conservative_factor)
    if fc_df.empty:
        # fallback: use last 12-month average
        avg = series.tail(12).mean() if len(series) > 0 else 0.0
        idx = pd.date_range(start=f"{target_year}-01-01", periods=12, freq="MS")
        return pd.Series([avg]*12, index=idx)
    # fc_df starts at month after last
    fc_df = fc_df.copy()
    # Filter fc_df rows corresponding to target_year
    fc_df['YEAR'] = fc_df['FECHA'].dt.year
    sel = fc_df[fc_df['YEAR'] == target_year]
    if sel.empty:
        # maybe fc_df covers beyond target_year; produce zeros
        idx = pd.date_range(start=f"{target_year}-01-01", periods=12, freq="MS")
        return pd.Series([0.0]*12, index=idx)
    # ensure months Jan..Dec present (if some missing, fill with 0)
    idx = pd.date_range(start=f"{target_year}-01-01", periods=12, freq="MS")
    out = pd.Series(0.0, index=idx)
    for _, r in sel.iterrows():
        d = pd.Timestamp(r['FECHA']).to_period('M').to_timestamp()
        if d.year == target_year:
            out.loc[d] = float(r['Forecast_mensual'])
    return out

# ---------- Data loading & normalization ----------
def normalize_industria(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # Flexible mapping
    colmap = {}
    for c in df.columns:
        cn = c.strip().lower()
        if 'homolog' in cn:
            colmap[c] = 'HOMO'
        elif cn in ['año','ano','year']:
            colmap[c] = 'ANIO'
        elif 'compañ' in cn or 'compania' in cn:
            colmap[c] = 'COMPANIA'
        elif 'ciudad' in cn:
            colmap[c] = 'CIUDAD'
        elif 'ram' in cn:
            colmap[c] = 'RAMO'
        elif ('primas' in cn and 'siniest' in cn) or 'primas/siniestros' in cn:
            colmap[c] = 'TIPO_VALOR'
        elif cn in ['primas','siniestros']:
            colmap[c] = 'TIPO_VALOR'
        elif 'fecha' in cn:
            colmap[c] = 'FECHA'
        elif 'valor' in cn or 'valor_mensual' in cn:
            colmap[c] = 'VALOR'
        elif 'depart' in cn:
            colmap[c] = 'DEPARTAMENTO'
        elif 'estado' in cn:
            colmap[c] = 'ESTADO'
    df = df.rename(columns=colmap)
    # Parse fecha
    if 'FECHA' in df.columns:
        df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
    else:
        if 'ANIO' in df.columns and 'MES' in df.columns:
            try:
                df['FECHA'] = pd.to_datetime(dict(year=df['ANIO'].astype(int), month=df['MES'].astype(int), day=1), errors='coerce')
            except Exception:
                df['FECHA'] = pd.NaT
        elif 'ANIO' in df.columns:
            try:
                df['FECHA'] = pd.to_datetime(df['ANIO'].astype(int).astype(str) + "-01-01", errors='coerce')
            except Exception:
                df['FECHA'] = pd.NaT
        else:
            df['FECHA'] = pd.NaT
    df['FECHA'] = df['FECHA'].dt.to_period("M").dt.to_timestamp()
    # Valor numeric
    if 'VALOR' in df.columns:
        df['VALOR'] = parse_number_co(df['VALOR'])
    else:
        for alt in ['Valor_Mensual','Valor Mensual','VALOR_MENSUAL','VALOR_MES']:
            if alt in df.columns:
                df['VALOR'] = parse_number_co(df[alt])
                break
        else:
            df['VALOR'] = pd.to_numeric(df.get('VALOR', pd.Series(dtype=float)), errors='coerce')
    # Tipo normalized
    if 'TIPO_VALOR' in df.columns:
        df['TIPO_VALOR'] = df['TIPO_VALOR'].astype(str).str.strip().str.lower()
    else:
        df['TIPO_VALOR'] = 'primas'
    # text clean
    for c in ['HOMO','COMPANIA','CIUDAD','RAMO','DEPARTAMENTO','ESTADO']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if 'ANIO' not in df.columns:
        df['ANIO'] = df['FECHA'].dt.year
    keep = ['ANIO','FECHA','HOMO','COMPANIA','CIUDAD','RAMO','TIPO_VALOR','VALOR','DEPARTAMENTO','ESTADO']
    keep = [c for c in keep if c in df.columns]
    return df[keep].dropna(subset=['FECHA']).copy()

def try_load_industria_direct(sheet_id: str, gid: str):
    url = export_csv_url(sheet_id, gid)
    try:
        resp = urllib.request.urlopen(url)
        # if reachable, load via pandas
        df = pd.read_csv(url)
        df = normalize_industria(df)
        return df
    except HTTPError as e:
        st.error(f"No fue posible acceder al Google Sheet (HTTP {e.code}: {e.reason}).")
        st.markdown("Asegúrate de que la hoja sea pública: *Compartir → Cambiar a 'Cualquiera con el enlace' → Lector*")
        st.markdown("O publica la hoja: *Archivo → Publicar en la web*.")
        st.markdown(f"Prueba abrir en el navegador esta URL (debe descargar un CSV): `{url}`")
        st.stop()
    except URLError as e:
        st.error("Error de red al intentar acceder al Google Sheet.")
        st.exception(e)
        st.stop()
    except Exception as e:
        st.error("Error al leer o normalizar el CSV del Google Sheet.")
        st.exception(e)
        st.stop()

# ---------- App UI ----------
st.title("Industria · Forecast 2026 — por Ciudad / Ramo")
st.markdown("La app carga automáticamente la Hoja1 indicada y calcula previsiones mes-a-mes para 2026.")

# Try to load automatically (no uploader; user requested direct load)
df_ind = try_load_industria_direct(DEFAULT_SHEET_ID, DEFAULT_GID)

# Basic sanity check
if df_ind.empty:
    st.error("La hoja se cargó pero no contiene filas válidas con FECHA.")
    st.stop()

# Controls: scope and segmentation
col_a, col_b, col_c = st.columns([2,2,2])
with col_a:
    year_analysis = st.selectbox("Año de análisis (nowcast/cierre)", options=sorted(df_ind['ANIO'].dropna().unique().astype(int).tolist()), index=len(df_ind['ANIO'].dropna().unique().astype(int).tolist())-1)
    target_year = st.number_input("Año objetivo para monthly forecast", min_value=2023, max_value=2030, value=2026, step=1)
with col_b:
    seg_mode = st.radio("Agrupar proyección por", options=["CIUDAD", "RAMO"], index=0, horizontal=True)
    top_n = st.number_input("Top N segmentos para proyectar (si no seleccionas manualmente)", min_value=1, max_value=50, value=15)
with col_c:
    tipo_options = sorted(df_ind['TIPO_VALOR'].dropna().unique().astype(str).tolist())
    tipos_sel = st.multiselect("Tipos a incluir (Primas / Siniestros)", options=tipo_options, default=tipo_options)
    scope_estado_only = st.checkbox("Solo ESTADO (mi empresa) en la proyección", value=False)
    # detect ESTADO values if available
    estado_values = sorted(df_ind['ESTADO'].dropna().unique().astype(str).tolist()) if 'ESTADO' in df_ind.columns else []
    my_company = st.selectbox("Mi compañía (si no usa ESTADO)", options=["(ninguna)"] + sorted(df_ind['COMPANIA'].dropna().unique().astype(str).tolist()), index=0)

conservative_adj = st.slider("Ajuste conservador forecast (%) (aplica a modelos)", min_value=-20.0, max_value=20.0, value=0.0, step=0.5)
conservative_factor = 1.0 + (conservative_adj / 100.0)

st.markdown("### Selección de segmentos a proyectar")
# Build list of segments (top N by selected year & tipos)
seg_col = seg_mode
df_sel_for_top = df_ind[(df_ind['FECHA'].dt.year == int(year_analysis)) & (df_ind['TIPO_VALOR'].isin(tipos_sel))]
seg_sums = df_sel_for_top.groupby(seg_col)['VALOR'].sum().reset_index().sort_values('VALOR', ascending=False)
default_segments = seg_sums.head(top_n)[seg_col].tolist()
segments_sel = st.multiselect(f"Selecciona {seg_col}(s) (dejar vacío para usar Top N)", options=sorted(df_ind[seg_col].dropna().unique().astype(str).tolist()), default=default_segments)

if not segments_sel:
    segments = default_segments
else:
    segments = segments_sel

st.info(f"Proyectando {len(segments)} segmentos ({seg_col}) para el año {target_year}. Esto puede tardar unos segundos por segmento (modelo SARIMAX).")

# Helper to aggregate series given a df and a segment value and tipo
def agg_monthly_series(df: pd.DataFrame, tipo: str, seg_col: str, seg_value: str, year_limit: Optional[int] = None) -> pd.Series:
    df2 = df.copy()
    df2 = df2[df2['TIPO_VALOR'].isin([tipo])] if tipo else df2
    df2 = df2[df2[seg_col] == seg_value]
    if year_limit:
        df2 = df2[df2['FECHA'].dt.year <= year_limit]
    s = df2.groupby('FECHA')['VALOR'].sum().sort_index()
    s.index = pd.to_datetime(s.index)
    return s

# Function to produce monthly 2026 forecast pivot for list of segments and given tipo (or combined tipos)
def produce_forecast_2026_by_segments(df: pd.DataFrame, segments: List[str], seg_col: str, tipos: List[str], target_year: int, conservative_factor: float, scope_estado_only: bool, my_company_choice: str):
    rows = []
    # Filter scope: if scope_estado_only True and ESTADO exists, filter df accordingly; else if my_company selected use COMPANIA
    df_scope = df.copy()
    if scope_estado_only and 'ESTADO' in df.columns:
        # interpret ESTADO truthy values (common patterns)
        truthy = df_scope['ESTADO'].astype(str).str.strip().str.lower().isin(['true','si','yes','1','mi_empresa','miempresa'])
        if truthy.sum() > 0:
            df_scope = df_scope[truthy]
        elif my_company_choice and my_company_choice != "(ninguna)":
            df_scope = df_scope[df_scope['COMPANIA'] == my_company_choice]
        else:
            # fallback: no ESTADO matches; do empty
            df_scope = df_scope[df_scope['COMPANIA'] == "###__NO_MATCH__###"]
    else:
        if my_company_choice and my_company_choice != "(ninguna)" and not scope_estado_only:
            # offer option to restrict to company if user selected it (but default is industry)
            pass

    # For each segment and each tipo, compute forecast vector for target_year
    for seg in segments:
        row = {"SEGMENT": seg}
        # combined tipos: sum forecasts across tipos
        monthly_total = pd.Series(0.0, index=pd.date_range(start=f"{target_year}-01-01", periods=12, freq="MS"))
        for tipo in tipos:
            ser = agg_monthly_series(df_scope, tipo, seg_col, seg, year_limit=None)
            ser = sanitize_trailing_zeros(ser, target_year-1)  # clean trailing zeros for previous year
            fc_yr = forecast_year_monthly_for_series(ser, target_year=target_year, conservative_factor=conservative_factor)
            monthly_total = monthly_total.add(fc_yr, fill_value=0.0)
            # store per-tipo columns if desired (optional)
            for d, v in fc_yr.items():
                mon_label = d.strftime("%b-%Y")
                row[f"{tipo}_{mon_label}"] = v
        # store combined months
        for d, v in monthly_total.items():
            mon_label = d.strftime("%b-%Y")
            row[f"TOTAL_{mon_label}"] = v
        row["TOTAL_2026"] = monthly_total.sum()
        rows.append(row)
    df_out = pd.DataFrame(rows)
    # pivot make months columns in order Jan..Dec
    month_idx = pd.date_range(start=f"{target_year}-01-01", periods=12, freq="MS")
    month_labels = [d.strftime("%b-%Y") for d in month_idx]
    # ensure columns order
    cols_order = ["SEGMENT"] + [f"TOTAL_{m}" for m in month_labels] + ["TOTAL_2026"]
    # if missing TOTAL_* columns create them
    for m in month_labels:
        colm = f"TOTAL_{m}"
        if colm not in df_out.columns:
            df_out[colm] = 0.0
    df_out = df_out[cols_order]
    return df_out

# Produce forecasts on button click
if st.button("Calcular proyección 2026 mes a mes"):
    with st.spinner("Generando forecasts 2026 por segmento..."):
        df_forecast_2026 = produce_forecast_2026_by_segments(df_ind, segments, seg_col, tipos_sel, target_year, conservative_factor, scope_estado_only, my_company)
        # show result pivot
        st.markdown(f"## Proyección mes a mes {target_year} por {seg_col}")
        # format numbers
        def fmt_val(x): return fmt_cop(x) if pd.notna(x) and x != 0 else "-"
        display = df_forecast_2026.copy()
        # show totals descending
        display = display.sort_values("TOTAL_2026", ascending=False).reset_index(drop=True)
        st.dataframe(display, use_container_width=True)
        # show top segments chart
        top_chart = display.head(20)
        # sum of all segments (industry) per month
        month_idx = pd.date_range(start=f"{target_year}-01-01", periods=12, freq="MS")
        month_labels = [d.strftime("%b-%Y") for d in month_idx]
        industry_month_totals = df_forecast_2026[[f"TOTAL_{m}" for m in month_labels]].sum().values
        fig_ind = go.Figure([go.Bar(x=month_labels, y=industry_month_totals, marker_color='#0ea5e9')])
        fig_ind.update_layout(title=f"Industria - Proyección mes a mes {target_year} (suma de segmentos)", yaxis_title="VALOR")
        st.plotly_chart(fig_ind, use_container_width=True)
        # If scope_estado_only selected, compute industry vs estado totals
        if scope_estado_only and 'ESTADO' in df_ind.columns:
            st.markdown("### Nota: Proyección realizada solo sobre registros marcados en ESTADO (mi empresa).")
        elif my_company != "(ninguna)" and scope_estado_only is False:
            st.markdown(f"### Nota: Resultados corresponden a industria total; para filtrar solo tu compañía selecciona la casilla 'Solo ESTADO' o indica ESTADO en datos.")
        # Export to Excel
        try:
            with BytesIO() as output:
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df_forecast_2026.to_excel(writer, sheet_name=f"forecast_{target_year}", index=False)
                    # also save raw input filtered for reference
                    df_ind.to_excel(writer, sheet_name="raw_industria", index=False)
                data_xls = output.getvalue()
            st.download_button("⬇️ Descargar proyección 2026 (Excel)", data=data_xls, file_name=f"industria_forecast_2026_by_{seg_col}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.warning("No fue posible generar el archivo Excel.")
            st.exception(e)

st.markdown("---")
st.markdown("Instrucciones y notas:")
st.markdown("- La app intenta leer directamente el Google Sheet mediante la URL de export CSV. Asegúrate de que la hoja sea pública (Compartir -> 'Cualquiera con el enlace' -> Lector) o usa 'Archivo -> Publicar en la web'.")
st.markdown("- Forecast por segmento usa SARIMAX con fallback ARIMA y aplica el ajuste conservador que especifiques.")
st.markdown("- Si la columna ESTADO está presente y marcas 'Solo ESTADO', la proyección se hará solo sobre los registros marcados como pertenecientes a tu empresa (valores truthy en ESTADO).")
st.markdown("- Si prefieres que la proyección combine Primas y Siniestros, selecciona ambos en 'Tipos a incluir'.")
