import strea# -*- coding: utf-8 -*-
"""
AseguraView · Primas & Siniestros Colombia
Forecast por Homologación, Ciudades Principales y Competidores
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from datetime import datetime, date
from typing import Dict, List, Optional

# Time series models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# XGBoost fallback
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# ----------------- CONFIG -----------------
st.set_page_config(page_title="AseguraView · Primas & Siniestros", layout="wide")

# CSS Styles (Dark theme with cards)
st.markdown("""
<style>
:root { 
    --bg:#071428; --fg:#f8fafc; --accent:#38bdf8; --muted:#9fb7cc; 
    --card:rgba(255,255,255,0.03); --up:#16a34a; --down:#ef4444; --near:#f59e0b; 
}
body,.stApp {background:var(--bg);color:var(--fg);}
.block-container{padding-top:.6rem;}
.card{background:var(--card);border:1px solid rgba(255,255,255,0.04);border-radius:12px;padding:12px;margin-bottom:12px}
.table-wrap{overflow:auto;border:1px solid rgba(255,255,255,0.04);border-radius:12px;background:transparent;padding:6px}
.tbl{width:100%;border-collapse:collapse;font-size:14px;color:var(--fg)}
.tbl thead th{position:sticky;top:0;background:#033b63;color:#ffffff;padding:10px;border-bottom:1px solid rgba(255,255,255,0.06);text-align:left}
.tbl td{padding:8px;border-bottom:1px dashed rgba(255,255,255,0.03);white-space:nowrap;color:var(--fg)}
.bad{color:var(--down);font-weight:700}
.ok{color:var(--up);font-weight:700}
.near{color:var(--near);font-weight:700}
.muted{color:var(--muted)}
.small{font-size:12px;color:var(--muted)}
.vertical-summary{display:flex;gap:12px;flex-wrap:wrap}
.vert-left{flex:0 0 360px}
.vert-right{flex:1;min-height:160px}
.vrow{display:flex;justify-content:space-between;padding:8px 10px;border-bottom:1px dashed rgba(255,255,255,0.03)}
.vtitle{color:var(--muted)}
.vvalue{font-weight:700;color:var(--fg)}
.badge{padding:3px 6px;border-radius:6px}

/* Cards for segments */
.lplus-cards-wrap{display:flex;gap:10px;flex-wrap:nowrap;overflow:auto;padding:6px}
.lplus-card{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.04);border-radius:10px;padding:10px;min-width:240px;max-width:260px;flex:0 0 auto}
.lplus-title{font-weight:800;margin-bottom:6px;color:var(--fg);font-size:13px}
.lplus-row{display:flex;justify-content:space-between;padding:6px 2px;border-bottom:1px dashed rgba(255,255,255,0.03);font-size:13px}
.lplus-row .vtitle{color:var(--muted);font-size:12px}
.lplus-row .vvalue{font-weight:700;color:var(--fg);font-size:13px}
</style>
""", unsafe_allow_html=True)

# ----------------- DATA SOURCE -----------------
# TU GOOGLE SHEETS ID (extraído de tu enlace)
SHEET_ID_DEFAULT = "1VljNnZtRPDA3TkTUP6w8AviZCPIfILqe"  # ✅ CORRECTO
SHEET_NAME_DATOS_DEFAULT = "Hoja1"  # Cambia si tu hoja tiene otro nombre
# Si tu hoja está en otra pestaña, descomenta y usa el GID:
# GID_DATOS = "293107109"  # Reemplaza con el gid de tu pestaña

# URL de exportación robusta
def gsheet_csv(sheet_id: str, sheet_name: str) -> str:
    """URL para exportar hoja específica por nombre"""
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&sheet={sheet_name}"

def csv_by_gid(sheet_id: str, gid: str) -> str:
    """URL para exportar hoja específica por GID"""
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

# ----------------- UTILITIES -----------------
@st.cache_data(show_spinner=False)
def load_cutoff_date(sheet_id: str, gid: str) -> pd.Timestamp:
    """Carga fecha de corte desde celda específica"""
    try:
        url = csv_by_gid(sheet_id, gid)
        df = pd.read_csv(url, header=None)
        raw = str(df.iloc[0, 0]).strip() if not df.empty else ""
        ts = pd.to_datetime(raw, dayfirst=True, errors='coerce')
        return pd.Timestamp(ts.date()) if pd.notna(ts) else pd.Timestamp.today().normalize()
    except Exception as e:
        st.sidebar.warning(f"No se pudo cargar fecha de corte: {e}")
        return pd.Timestamp.today().normalize()

def parse_number_co(series: pd.Series) -> pd.Series:
    """Parsea números en formato colombiano"""
    s = series.astype(str).fillna("")
    s = s.str.replace(r"[^\d,.\-]", "", regex=True)
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def fmt_cop(x):
    """Formatea valor en pesos colombianos"""
    try:
        return "$" + f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return x

def badge_pct_html(val):
    """Badge para porcentajes con colores"""
    try:
        f = float(val)
        cls = "ok" if f >= 100 else "near" if f >= 95 else "bad"
        return f'<span class="{cls}">{f:.1f}%</span>'
    except Exception:
        return "-"

def badge_growth_cop_html(val):
    """Badge para crecimiento en COP"""
    try:
        f = float(val)
        cls = "ok" if f >= 0 else "bad"
        return f'<span class="{cls}">{fmt_cop(f)}</span>'
    except Exception:
        return "-"

def df_to_html(df_in: pd.DataFrame):
    """Convierte DataFrame a HTML con estilos"""
    html = '<div class="table-wrap"><table class="tbl"><thead><tr>'
    for c in df_in.columns:
        html += f'<th>{c}</th>'
    html += '</tr></thead><tbody>'
    for _, r in df_in.iterrows():
        html += '<tr>'
        for c in df_in.columns:
            html += f'<td>{r[c]}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    return html

# ----------------- LOAD & NORMALIZE DATA -----------------
@st.cache_data(show_spinner=False)
def load_gsheet(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    """Carga datos desde Google Sheets con manejo de errores"""
    try:
        url = gsheet_csv(sheet_id, sheet_name)
        df = pd.read_csv(url)
        return normalize_columns(df)
    except Exception as e:
        st.error(f"❌ Error cargando Google Sheets: {e}")
        st.info("💡 **Solución:** Haz tu hoja pública: Compartir → 'Cualquiera con el enlace' → Lector")
        
        # Botón para usar datos de ejemplo
        if st.button("▶️ Usar Datos de Ejemplo para Probar"):
            return generate_sample_data()
        return pd.DataFrame()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas y tipos de datos"""
    df.columns = [c.strip() for c in df.columns]
    
    # Map columns (flexible para diferentes nombres)
    rename_map = {
        'HOMOLOGACIÓN': 'HOMOLOGACION', 'HOMOLOGACION': 'HOMOLOGACION',
        'Año': 'ANIO', 'ANO': 'ANIO', 'YEAR': 'ANIO',
        'COMPAÑÍA': 'COMPANIA', 'COMPANIA': 'COMPANIA', 'COMPAÑIA': 'COMPANIA',
        'CIUDAD': 'CIUDAD', 'CIUDADES': 'CIUDAD',
        'RAMOS': 'RAMO', 'RAMO': 'RAMO',
        'Primas/Siniestros': 'TIPO_VALOR', 'Primas_Siniestros': 'TIPO_VALOR', 'TIPO': 'TIPO_VALOR',
        'FECHA': 'FECHA', 'FECHA_MES': 'FECHA',
        'Valor_Mensual': 'VALOR', 'VALOR_MENSUAL': 'VALOR', 'VALOR': 'VALOR',
        'DEPARTAMENTO': 'DEPARTAMENTO'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Validar columnas mínimas
    required_cols = ['FECHA', 'VALOR']
    if not all(col in df.columns for col in required_cols):
        st.error("❌ Columnas requeridas faltantes: FECHA y VALOR")
        return pd.DataFrame()
    
    # Parse fecha
    if 'FECHA' in df.columns:
        df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
    else:
        # Intentar construir desde Año/Mes si existe
        if 'ANIO' in df.columns and 'MES' in df.columns:
            try:
                df['FECHA'] = pd.to_datetime(dict(year=df['ANIO'].astype(int), 
                                                 month=df['MES'].astype(int), day=1), errors='coerce')
            except:
                df['FECHA'] = pd.to_datetime(df.get('ANIO', pd.Series()).astype(str)+"-01-01", errors='coerce')
    
    df['FECHA'] = df['FECHA'].dt.to_period("M").dt.to_timestamp()
    
    # Parse valor
    df['VALOR'] = parse_number_co(df['VALOR'])
    
    # Normalizar texto
    for col in ['HOMOLOGACION', 'COMPANIA', 'CIUDAD', 'RAMO', 'TIPO_VALOR']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    
    if 'ANIO' not in df.columns:
        df['ANIO'] = df['FECHA'].dt.year
    
    # Seleccionar columnas disponibles
    keep = ['HOMOLOGACION', 'ANIO', 'COMPANIA', 'CIUDAD', 'RAMO', 'TIPO_VALOR', 'FECHA', 'VALOR', 'DEPARTAMENTO']
    keep = [c for c in keep if c in df.columns]
    
    # Eliminar filas sin fecha o valor
    df = df.dropna(subset=['FECHA', 'VALOR'])
    
    if df.empty:
        st.warning("⚠️ No hay datos válidos después de la normalización")
        return pd.DataFrame()
    
    return df[keep].copy()

def generate_sample_data():
    """Genera datos de ejemplo realistas para pruebas"""
    st.warning("⚠️ **Usando datos de ejemplo** - Configura tu Google Sheets para datos reales")
    
    dates = pd.date_range(start='2020-01-01', end='2025-07-31', freq='M')
    companias = ['ESTADO', 'MAPFRE', 'LIBERTY', 'AXA', 'MUNDIAL', 'PREVISORA', 'ALFA', 'ALLIANZ']
    ciudades = ['BOGOTA', 'MEDELLIN', 'CALI', 'BUCARAMANGA', 'BARRANQUILLA', 'CARTAGENA', 'TUNJA', 'BUENAVENTURA']
    ramos = ['VIDRIOS', 'INCENDIO', 'ROBO', 'RESPONSABILIDAD CIVIL', 'VEHICULOS', 'VIDA', 'SALUD']
    homologaciones = ['GENERALES', 'ESPECIALES', 'EXCLUIDOS']
    
    data = []
    for date in dates:
        for compania in companias[:4]:  # Reducido para velocidad
            for ciudad in ciudades[:5]:
                for ramo in ramos[:4]:
                    base_valor = np.random.normal(50000, 15000)
                    data.append({
                        'HOMOLOGACION': np.random.choice(homologaciones),
                        'ANIO': date.year,
                        'COMPANIA': compania,
                        'CIUDAD': ciudad,
                        'RAMO': ramo,
                        'TIPO_VALOR': 'PRIMAS',
                        'FECHA': date,
                        'VALOR': max(0, base_valor),
                        'DEPARTAMENTO': 'VALLE DEL CAUCA' if ciudad == 'BUENAVENTURA' else 'ANTIOQUIA'
                    })
                    # Siniestros (20% de primas)
                    data.append({
                        'HOMOLOGACION': np.random.choice(homologaciones),
                        'ANIO': date.year,
                        'COMPANIA': compania,
                        'CIUDAD': ciudad,
                        'RAMO': ramo,
                        'TIPO_VALOR': 'SINIESTROS',
                        'FECHA': date,
                        'VALOR': max(0, base_valor * np.random.normal(0.2, 0.05)),
                        'DEPARTAMENTO': 'VALLE DEL CAUCA' if ciudad == 'BUENAVENTURA' else 'ANTIOQUIA'
                    })
    
    return pd.DataFrame(data)

# ----------------- PROPORTIONS & SEGMENT SUMMARY -----------------
def proporciones_segmento_mes(df_scope: pd.DataFrame, col_segmento: str, mes_ts: pd.Timestamp, ventana_meses: int = 11) -> Dict[str, float]:
    """Calcula proporción de cada segmento en el mes"""
    if col_segmento not in df_scope.columns:
        return {}
    ventana_ini = mes_ts - pd.DateOffset(months=ventana_meses)
    tmp = df_scope[(df_scope["FECHA"] >= ventana_ini) & (df_scope["FECHA"] <= mes_ts)].groupby(col_segmento)["VALOR"].sum()
    segs = sorted(df_scope[col_segmento].dropna().unique())
    if tmp.sum() > 0:
        prop = tmp / tmp.sum()
    else:
        prop = pd.Series([1/len(segs)]*len(segs), index=segs) if segs else pd.Series(dtype=float)
    return prop.to_dict()

def resumen_segmentado_df(df_scope: pd.DataFrame, col_segmento: str, ref_year: int, mes_ref: int, mes_ts: pd.Timestamp,
                          forecast_mes_total: float, habiles_restantes_mes: int, vista_select: str,
                          forecast_annual_total: Optional[float] = None,
                          df_scope_full: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Genera resumen segmentado con forecast"""
    if col_segmento not in df_scope.columns:
        return pd.DataFrame()
    
    df = df_scope.copy()
    mask_mes_actual = (df["FECHA"].dt.year == ref_year) & (df["FECHA"].dt.month == mes_ref)
    mask_mes_prev = (df["FECHA"].dt.year == ref_year-1) & (df["FECHA"].dt.month == mes_ref)
    mask_ytd = (df["FECHA"].dt.year == ref_year) & (df["FECHA"].dt.month <= mes_ref)
    mask_ytd_prev = (df["FECHA"].dt.year == ref_year-1) & (df["FECHA"].dt.month <= mes_ref)

    # Usar dataset completo para presupuesto y previos
    if df_scope_full is not None and col_segmento in df_scope_full.columns:
        prod_prev_full_series = df_scope_full[df_scope_full["FECHA"].dt.year == (ref_year-1)].groupby(col_segmento)["VALOR"].sum()
    else:
        prod_prev_full_series = df[df["FECHA"].dt.year == (ref_year-1)].groupby(col_segmento)["VALOR"].sum()

    prod_mes = df[mask_mes_actual].groupby(col_segmento)["VALOR"].sum()
    prod_prev = df[mask_mes_prev].groupby(col_segmento)["VALOR"].sum()
    prod_ytd = df[mask_ytd].groupby(col_segmento)["VALOR"].sum()
    prod_ytd_prev = df[mask_ytd_prev].groupby(col_segmento)["VALOR"].sum()

    segmentos = sorted(df[col_segmento].dropna().unique())
    prop = proporciones_segmento_mes(df_scope, col_segmento, mes_ts, ventana_meses=11)
    rows = []
    
    for seg in segmentos:
        prod_seg = float(prod_mes.get(seg, 0.0))
        prev_seg = float(prod_prev.get(seg, 0.0))
        ytd_seg = float(prod_ytd.get(seg, 0.0))
        ytd_prev_seg = float(prod_ytd_prev.get(seg, 0.0))
        share = float(prop.get(seg, 0.0))
        fc_seg = share * forecast_mes_total

        if vista_select == "Mes":
            growth_abs = fc_seg - prev_seg
            growth_pct = (fc_seg / prev_seg - 1.0) * 100.0 if prev_seg > 0 else np.nan
            row = {
                col_segmento: seg,
                "Previo": fmt_cop(prev_seg),
                "Actual": fmt_cop(prod_seg),
                "Forecast (mes)": fmt_cop(fc_seg),
                "Crec. Fc (COP)": badge_growth_cop_html(growth_abs),
                "Crec. Fc (%)": badge_growth_pct_html(growth_pct),
            }
        elif vista_select == "Año":
            prod_prev_label = float(prod_prev_full_series.get(seg, 0.0))
            prod_act_label = ytd_seg
            fc_ann = share * forecast_annual_total if forecast_annual_total else fc_seg * 12.0
            growth_abs = fc_ann - prod_prev_label
            growth_pct = (fc_ann / prod_prev_label - 1.0) * 100.0 if prod_prev_label > 0 else np.nan
            row = {
                col_segmento: seg,
                "Previo (año prev.)": fmt_cop(prod_prev_label),
                "Actual (YTD)": fmt_cop(prod_act_label),
                "Forecast (anual est.)": fmt_cop(fc_ann),
                "Crec. Fc (COP)": badge_growth_cop_html(growth_abs),
                "Crec. Fc (%)": badge_growth_pct_html(growth_pct),
            }
        else:
            growth_abs = fc_seg - ytd_prev_seg
            growth_pct = (fc_seg / ytd_prev_seg - 1.0) * 100.0 if ytd_prev_seg > 0 else np.nan
            row = {
                col_segmento: seg,
                "Previo (YTD)": fmt_cop(ytd_prev_seg),
                "Actual (YTD)": fmt_cop(ytd_seg),
                "Forecast (YTD est.)": fmt_cop(fc_seg),
                "Crec. Fc (COP)": badge_growth_cop_html(growth_abs),
                "Crec. Fc (%)": badge_growth_pct_html(growth_pct),
            }
        rows.append(row)

    df_out = pd.DataFrame(rows)
    cols_order = [col_segmento] + [c for c in rows[0].keys() if c != col_segmento]
    return df_out[cols_order]

# ----------------- FORECAST MODELS -----------------
def fit_forecast_prophet(ts: pd.Series, steps: int, conservative_factor: float = 1.0):
    """Prophet forecast with seasonal components"""
    if ts.empty or len(ts) < 3:
        return pd.DataFrame(), np.nan
    
    df_prophet = pd.DataFrame({'ds': ts.index, 'y': ts.values}).dropna()
    
    if len(df_prophet) < 3:
        return pd.DataFrame(), np.nan
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=steps, freq='MS')
    forecast = model.predict(future)
    
    forecast_future = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps)
    forecast_future['yhat'] = forecast_future['yhat'] * conservative_factor
    
    return forecast_future, np.nan

def fit_forecast_xgboost(ts: pd.Series, steps: int, conservative_factor: float = 1.0):
    """XGBoost forecast using time features"""
    if not XGBOOST_AVAILABLE or ts.empty or len(ts) < 6:
        return pd.DataFrame(), np.nan
    
    df = pd.DataFrame({'year': ts.index.year, 'month': ts.index.month, 'y': ts.values})
    
    if len(np.unique(df['y'])) <= 1:
        return pd.DataFrame(), np.nan
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(df[['year', 'month']], df['y'])
    
    future_dates = pd.date_range(start=ts.index.max() + pd.DateOffset(months=1), periods=steps, freq='MS')
    future_df = pd.DataFrame({'year': future_dates.year, 'month': future_dates.month})
    
    predictions = model.predict(future_df) * conservative_factor
    
    return pd.DataFrame({'ds': future_dates, 'yhat': predictions, 'yhat_lower': predictions*0.8, 'yhat_upper': predictions*1.2}), np.nan

# ----------------- MAIN APP -----------------
# Load data
df = load_gsheet(SHEET_ID_DEFAULT, SHEET_NAME_DATOS_DEFAULT)
fecha_corte = load_cutoff_date(SHEET_ID_DEFAULT, GID_FECHA_CORTE)

# Header
st.markdown(f"""
<div style="display:flex;align-items:center;gap:18px;margin-bottom:6px">
  <div style="font-size:26px;font-weight:800;color:#f3f4f6">AseguraView · Primas & Siniestros</div>
  <div style="opacity:.85;color:var(--muted);">Corte: {fecha_corte.strftime('%d/%m/%Y')}</div>
</div>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("🔧 Filtros Globales")

# Compañías
comp_opts = ["TODAS"] + sorted(df['COMPANIA'].dropna().unique()) if 'COMPANIA' in df.columns and not df.empty else ["TODAS"]
compania_sel = st.sidebar.selectbox("Compañía", comp_opts)

# Ciudades
ciudad_opts = ["TODAS"] + sorted(df['CIUDAD'].dropna().unique()) if 'CIUDAD' in df.columns and not df.empty else ["TODAS"]
ciudad_sel = st.sidebar.selectbox("Ciudad", ciudad_opts)

# Ramos
ramo_opts = ["TODAS"] + sorted(df['RAMO'].dropna().unique()) if 'RAMO' in df.columns and not df.empty else ["TODAS"]
ramo_sel = st.sidebar.selectbox("Ramo", ramo_opts)

# Año de análisis
anio_analisis = st.sidebar.number_input("Año de análisis", min_value=2018, max_value=2100, 
                                        value=fecha_corte.year, step=1)

# Ajuste conservador
st.sidebar.markdown("#### Ajuste Conservador")
ajuste_pct = st.sidebar.slider("Ajuste forecast (%)", min_value=-20.0, max_value=20.0, 
                               value=0.0, step=0.5)
conservative_factor = 1.0 + (ajuste_pct / 100.0)

# Apply filters
df_sel_full = df.copy()
if df.empty:
    st.stop()

if compania_sel != "TODAS" and 'COMPANIA' in df_sel_full.columns:
    df_sel_full = df_sel_full[df_sel_full['COMPANIA'] == compania_sel]
if ciudad_sel != "TODAS" and 'CIUDAD' in df_sel_full.columns:
    df_sel_full = df_sel_full[df_sel_full['CIUDAD'] == ciudad_sel]
if ramo_sel != "TODAS" and 'RAMO' in df_sel_full.columns:
    df_sel_full = df_sel_full[df_sel_full['RAMO'] == ramo_sel]

# Dataset truncado para análisis (hasta fecha de corte)
df_sel = df_sel_full.copy()
df_sel = df_sel[(df_sel['FECHA'].dt.year <= anio_analisis)]
df_sel = df_sel[~((df_sel['FECHA'].dt.year == anio_analisis) & 
                  (df_sel['FECHA'] > pd.Timestamp(anio_analisis, fecha_corte.month, 1)))]

# Tabs
tabs = st.tabs(["🏠 Inicio", "📋 Homologación", "🏙️ Ciudades", "🏢 Competidores"])

# -------- TAB 0: INICIO --------
with tabs[0]:
    st.header("Bienvenido a AseguraView")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Compañías", len(df_sel_full['COMPANIA'].unique()))
    with col2:
        st.metric("Ciudades", len(df_sel_full['CIUDAD'].unique()))
    with col3:
        st.metric("Ramos", len(df_sel_full['RAMO'].unique()))
    with col4:
        st.metric("Registros", f"{len(df_sel_full):,}")
    
    if not df_sel_full.empty:
        # Evolución temporal
        st.subheader("Evolución Temporal")
        df_temp = df_sel_full.groupby(['FECHA', 'TIPO_VALOR'])['VALOR'].sum().reset_index()
        
        if not df_temp.empty:
            fig = px.line(df_temp, x='FECHA', y='VALOR', color='TIPO_VALOR',
                         title="Primas vs Siniestros", 
                         color_discrete_map={'PRIMAS': '#38bdf8', 'SINIESTROS': '#ef4444'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Top 5 segmentos
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Compañías")
            top_comp = df_sel_full.groupby('COMPANIA')['VALOR'].sum().nlargest(5).reset_index()
            top_comp['VALOR'] = top_comp['VALOR'].apply(fmt_cop)
            st.dataframe(top_comp, use_container_width=True)
        
        with col2:
            st.subheader("Top Ciudades")
            top_ciu = df_sel_full.groupby('CIUDAD')['VALOR'].sum().nlargest(5).reset_index()
            top_ciu['VALOR'] = top_ciu['VALOR'].apply(fmt_cop)
            st.dataframe(top_ciu, use_container_width=True)

# -------- TAB 1: HOMOLOGACION --------
with tabs[1]:
    st.header("📋 Predicciones por Homologación")
    
    if df_sel.empty:
        st.warning("No hay datos para analizar")
    else:
        # Preparar datos por homologación
        df_homo = df_sel.groupby(['HOMOLOGACION', 'FECHA', 'TIPO_VALOR'])['VALOR'].sum().reset_index()
        
        col1, col2 = st.columns([1, 2])
        with col1:
            tipo_vista = st.selectbox("Tipo de Valor", ["Primas", "Siniestros"])
            modelo_sel = st.selectbox("Modelo Predictivo", ["Prophet", "XGBoost", "Promedio"])
        
        # Filtrar por tipo
        df_tipo = df_homo[df_homo['TIPO_VALOR'] == tipo_vista.upper()]
        
        if df_tipo.empty:
            st.info(f"No hay datos de {tipo_vista.lower()} para los filtros seleccionados")
        else:
            # Calcular proyecciones
            homologaciones = df_tipo['HOMOLOGACION'].unique()
            resultados = []
            
            for homo in homologaciones[:10]:  # Limitar a 10 para velocidad
                df_h = df_tipo[df_tipo['HOMOLOGACION'] == homo]
                
                if len(df_h) > 6:
                    serie = df_h.set_index('FECHA')['VALOR']
                    steps = 12 - fecha_corte.month + 1
                    
                    if modelo_sel == "Prophet":
                        fc_df, _ = fit_forecast_prophet(serie, steps, conservative_factor)
                        forecast_6m = fc_df['yhat'].iloc[5] if len(fc_df) >= 6 else 0
                    elif modelo_sel == "XGBoost" and XGBOOST_AVAILABLE:
                        fc_df, _ = fit_forecast_xgboost(serie, steps, conservative_factor)
                        forecast_6m = fc_df['yhat'].iloc[5] if len(fc_df) >= 6 else 0
                    else:
                        forecast_6m = serie.tail(6).mean() if len(serie) >= 6 else serie.mean()
                    
                    ultimo_valor = serie.iloc[-1] if not serie.empty else 0
                    crecimiento = ((forecast_6m - ultimo_valor) / ultimo_valor * 100) if ultimo_valor > 0 else 0
                    
                    resultados.append({
                        'HOMOLOGACION': homo,
                        'Último Valor': fmt_cop(ultimo_valor),
                        'Forecast 6M': fmt_cop(forecast_6m),
                        'Crecimiento %': f"{crecimiento:.1f}%",
                        'Tendencia': '📈' if crecimiento > 0 else '📉'
                    })
            
            if resultados:
                df_result = pd.DataFrame(resultados)
                st.dataframe(df_result, use_container_width=True)
                
                # Gráfico
                df_result['Crecimiento Num'] = df_result['Crecimiento %'].str.replace('%', '').astype(float)
                fig = px.bar(df_result, x='HOMOLOGACION', y='Crecimiento Num',
                           title=f"Crecimiento Estimado - {modelo_sel}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay suficientes datos para generar predicciones")

# -------- TAB 2: CIUDADES --------
with tabs[2]:
    st.header("🏙️ Análisis de Ciudades Principales")
    
    # Ciudades objetivo
    ciudades_objetivo = ['BOGOTA', 'MEDELLIN', 'CALI', 'BUCARAMANGA', 'BARRANQUILLA', 'CARTAGENA', 'TUNJA']
    ciudades_disponibles = [c for c in ciudades_objetivo if c in df_sel['CIUDAD'].unique()]
    
    if not ciudades_disponibles:
        st.warning("No hay datos para las ciudades principales con los filtros seleccionados")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            ciudad_foco = st.selectbox("Selecciona ciudad para detalle:", ciudades_disponibles)
        with col2:
            vista_ciudad = st.radio("Vista", ["Mes", "Año"], horizontal=True)
        
        # Resumen por ciudad
        df_ciudad = df_sel[df_sel['CIUDAD'].isin(ciudades_disponibles)]
        df_ciudad_resumen = df_ciudad.groupby(['CIUDAD', 'FECHA', 'TIPO_VALOR'])['VALOR'].sum().reset_index()
        
        # Calcular proyección para ciudad seleccionada
        df_ciudad_foco = df_ciudad_resumen[df_ciudad_resumen['CIUDAD'] == ciudad_foco]
        df_ciudad_foco_primas = df_ciudad_foco[df_ciudad_foco['TIPO_VALOR'] == 'PRIMAS']
        
        if not df_ciudad_foco_primas.empty:
            serie_ciudad = df_ciudad_foco_primas.set_index('FECHA')['VALOR']
            steps = 12 - fecha_corte.month + 1
            
            fc_df, _ = fit_forecast_prophet(serie_ciudad, steps, conservative_factor)
            
            # Mostrar métricas
            ultimo_valor = serie_ciudad.iloc[-1] if not serie_ciudad.empty else 0
            forecast_6m = fc_df['yhat'].iloc[5] if len(fc_df) >= 6 else 0
            crecimiento = ((forecast_6m - ultimo_valor) / ultimo_valor * 100) if ultimo_valor > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Último Valor", fmt_cop(ultimo_valor))
            with col2:
                st.metric("Forecast 6M", fmt_cop(forecast_6m))
            with col3:
                st.metric("Crecimiento", f"{crecimiento:.1f}%")
            
            # Gráfico
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=serie_ciudad.index, y=serie_ciudad.values, mode='lines+markers', 
                                   name='Histórico', line=dict(color='#38bdf8')))
            if not fc_df.empty:
                fig.add_trace(go.Scatter(x=fc_df['ds'], y=fc_df['yhat'], mode='lines+markers', 
                                       name='Forecast', line=dict(dash='dash', color='#16a34a')))
            fig.update_layout(title=f"Proyección para {ciudad_foco}", yaxis_title="COP")
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabla comparativa
        st.markdown("### Comparativa entre Ciudades")
        rows_ciudad = []
        for ciudad in ciudades_disponibles:
            df_c = df_ciudad_resumen[(df_ciudad_resumen['CIUDAD'] == ciudad) & (df_ciudad_resumen['TIPO_VALOR'] == 'PRIMAS')]
            if not df_c.empty:
                serie_c = df_c.set_index('FECHA')['VALOR']
                total_anio = serie_c[serie_c.index.year == anio_analisis].sum()
                total_prev = serie_c[serie_c.index.year == anio_analisis-1].sum()
                growth = ((total_anio - total_prev) / total_prev * 100) if total_prev > 0 else 0
                rows_ciudad.append({
                    'Ciudad': ciudad,
                    f'{anio_analisis}': fmt_cop(total_anio),
                    f'{anio_analisis-1}': fmt_cop(total_prev),
                    'Crecimiento': badge_growth_pct_html(growth)
                })
        
        if rows_ciudad:
            df_comp = pd.DataFrame(rows_ciudad)
            st.markdown(df_to_html(df_comp), use_container_width=True)

# -------- TAB 3: COMPETIDORES --------
with tabs[3]:
    st.header("🏢 Vista de Competidores Principales")
    
    # Competidores objetivo
    competidores_objetivo = ['ESTADO', 'MAPFRE GENERALES', 'LIBERTY', 'AXA GENERALES', 'MUNDIAL', 'PREVISORA']
    df_sel['COMP_NORMALIZADO'] = df_sel['COMPANIA'].str.strip().str.upper()
    competidores_disponibles = [c for c in competidores_objetivo if c in df_sel['COMP_NORMALIZADO'].unique()]
    
    if not competidores_disponibles:
        st.warning("No hay datos para los competidores principales")
        st.info("Compañías disponibles: " + ", ".join(sorted(df_sel['COMP_NORMALIZADO'].unique()[:10])))
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            competidor_foco = st.selectbox("Selecciona competidor:", competidores_disponibles)
        with col2:
            vista_comp = st.radio("Vista", ["Mes", "Año"], horizontal=True)
        
        # Análisis del competidor seleccionado
        df_competidor = df_sel[df_sel['COMP_NORMALIZADO'] == competidor_foco]
        df_competidor_primas = df_competidor[df_competidor['TIPO_VALOR'] == 'PRIMAS']
        
        if not df_competidor_primas.empty:
            serie_comp = df_competidor_primas.groupby('FECHA')['VALOR'].sum().sort_index()
            steps = 12 - fecha_corte.month + 1
            
            fc_df, _ = fit_forecast_prophet(serie_comp, steps, conservative_factor)
            
            # KPIs
            ultimo_valor = serie_comp.iloc[-1] if not serie_comp.empty else 0
            forecast_6m = fc_df['yhat'].iloc[5] if len(fc_df) >= 6 else 0
            crecimiento = ((forecast_6m - ultimo_valor) / ultimo_valor * 100) if ultimo_valor > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                market_share = df_competidor_primas['VALOR'].sum() / df_sel[df_sel['TIPO_VALOR'] == 'PRIMAS']['VALOR'].sum() * 100
                st.metric("Market Share Est.", f"{market_share:.1f}%")
            with col2:
                st.metric("Forecast 6M", fmt_cop(forecast_6m))
            with col3:
                st.metric("Crecimiento", f"{crecimiento:.1f}%")
            
            # Gráfico
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=serie_comp.index, y=serie_comp.values, mode='lines+markers', 
                                   name='Histórico', line=dict(color='#38bdf8')))
            if not fc_df.empty:
                fig.add_trace(go.Scatter(x=fc_df['ds'], y=fc_df['yhat'], mode='lines+markers', 
                                       name='Forecast', line=dict(dash='dash', color='#16a34a')))
            fig.update_layout(title=f"Proyección - {competidor_foco}", yaxis_title="COP")
            st.plotly_chart(fig, use_container_width=True)
        
        # Comparativa de competidores
        st.markdown("### Comparativa de Competidores")
        rows_comp = []
        for comp in competidores_disponibles:
            df_c = df_sel[(df_sel['COMP_NORMALIZADO'] == comp) & (df_sel['TIPO_VALOR'] == 'PRIMAS')]
            if not df_c.empty:
                serie_c = df_c.groupby('FECHA')['VALOR'].sum()
                total_anio = serie_c[serie_c.index.year == anio_analisis].sum()
                total_prev = serie_c[serie_c.index.year == anio_analisis-1].sum()
                growth = ((total_anio - total_prev) / total_prev * 100) if total_prev > 0 else 0
                rows_comp.append({
                    'Competidor': comp,
                    f'{anio_analisis}': fmt_cop(total_anio),
                    f'{anio_analisis-1}': fmt_cop(total_prev),
                    'Crecimiento': badge_growth_pct_html(growth)
                })
        
        if rows_comp:
            df_comp_table = pd.DataFrame(rows_comp)
            st.markdown(df_to_html(df_comp_table), use_container_width=True)
        
        # Market Share Pie
        st.markdown("##### Market Share - Primas")
        df_market = df_sel[df_sel['TIPO_VALOR'] == 'PRIMAS'].groupby('COMP_NORMALIZADO')['VALOR'].sum().reset_index()
        df_market = df_market[df_market['COMP_NORMALIZADO'].isin(competidores_disponibles)]
        
        if not df_market.empty:
            fig_pie = go.Figure(data=[go.Pie(
                labels=df_market['COMP_NORMALIZADO'], 
                values=df_market['VALOR'],
                hole=0.4,
                marker_colors=['#38bdf8', '#16a34a', '#f59e0b', '#ef4444', '#a855f7', '#0ea5e9']
            )])
            fig_pie.update_layout(title="Market Share Competidores")
            st.plotly_chart(fig_pie, use_container_width=True)

# ----------------- FOOTER & DOWNLOADS -----------------
st.sidebar.markdown("---")
st.sidebar.info(f"""
📊 **Conexión:** Google Sheets  
🔮 **Modelos:** Prophet, XGBoost  
📅 **Corte:** {fecha_corte.strftime('%d/%m/%Y')}  
📈 **Registros:** {len(df_sel_full):,}
""")

# Download raw data
if not df_sel_full.empty:
    csv_data = df_sel_full.to_csv(index=False)
    st.sidebar.download_button(
        "⬇️ Descargar Datos Filtrados (CSV)",
        data=csv_data,
        file_name="primas_siniestros_colombia.csv",
        mime="text/csv"
    )

# Recargar datos
if st.sidebar.button("🔄 Recargar Datos"):
    st.cache_data.clear()
    st.rerun()
