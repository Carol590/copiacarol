import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SARIMA Predicción Primas/Siniestros", layout="wide")

@st.cache_data(ttl=300)
def cargar_datos():
    url = "https://docs.google.com/spreadsheets/d/1VljNnZtRPDA3TkTUP6w8AviZCPIfILqe/export?format=csv&gid=293107109"
    try:
        df = pd.read_csv(url)
        st.success(f"✅ {len(df):,} filas cargadas")
        return df
    except:
        return pd.DataFrame()

def preparar_datos(df):
    df.columns = df.columns.str.strip()
    
    if 'FECHA' in df.columns:
        df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
        df['YEAR'] = df['FECHA'].dt.year
        df['MONTH'] = df['FECHA'].dt.month
        df['FECHA_MENSUAL'] = df['FECHA'].dt.to_period('M').dt.to_timestamp()
        df = df.sort_values('FECHA')
    
    if 'Valor_Mensual' in df.columns:
        df['Valor_Mensual'] = pd.to_numeric(df['Valor_Mensual'], errors='coerce').fillna(0)
    
    if 'Primas/Siniestros' in df.columns:
        df['Primas'] = np.where(df['Primas/Siniestros'] == 'Primas', df['Valor_Mensual'], 0)
        df['Siniestros'] = np.where(df['Primas/Siniestros'] == 'Siniestros', df['Valor_Mensual'], 0)
    
    for col in ['HOMOLOGACIÓN', 'CIUDAD', 'COMPAÑÍA', 'RAMOS']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    return df.dropna(subset=['YEAR', 'MONTH'])

def sarima_por_grupo(df_filt, grupo_col, grupo_valor, target_col, steps=5):
    mask = df_filt[grupo_col] == grupo_valor
    if mask.sum() < 12:
        return np.full(steps, df_filt.loc[mask, target_col].mean())
    
    series = df_filt.loc[mask].groupby('FECHA_MENSUAL')[target_col].sum()
    
    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12))
        fitted = model.fit(disp=False)
        forecast = fitted.get_forecast(steps=steps)
        return forecast.predicted_mean.values.round(0)
    except:
        return np.full(steps, series.tail(6).mean())

def calcular_sarima_homologacion(df_filt, target, steps=5):
    resultados = []
    homologaciones = df_filt['HOMOLOGACIÓN'].unique()
    
    for homologacion in homologaciones:
        pred = sarima_por_grupo(df_filt, 'HOMOLOGACIÓN', homologacion, target, steps)
        
        for i, mes in enumerate([8,9,10,11,12]):
            resultados.append({
                'HOMOLOGACIÓN': homologacion,
                'Mes_Nombre': ['Agosto','Septiembre','Octubre','Noviembre','Diciembre'][i],
                'Predicción': pred[i]
            })
    
    return pd.DataFrame(resultados)

def calcular_sarima_compania(df_filt, target_col='Primas', steps=5):
    resultados = []
    companias = df_filt['COMPAÑÍA'].unique()
    
    for compania in companias:
        pred = sarima_por_grupo(df_filt, 'COMPAÑÍA', compania, target_col, steps)
        
        for i, mes in enumerate([8,9,10,11,12]):
            resultados.append({
                'COMPAÑÍA': compania,
                'Mes_Nombre': ['Agosto','Septiembre','Octubre','Noviembre','Diciembre'][i],
                'Predicción': pred[i]
            })
    
    return pd.DataFrame(resultados)

# === CARGAR DATOS ===
df = cargar_datos()
if df.empty:
    st.stop()

df_clean = preparar_datos(df)

# === TABS ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 SARIMA Homologación", "🚗 AUTOMÓVILES", "✅ CUMPLIMIENTO", 
    "🏢 GENERALES", "⚠️ RC", "🚑 SOAT", "💚 VIDA", "❌ NO SDE"
])

# === TAB 1: SARIMA por HOMOLOGACIÓN ===
with tab1:
    st.header("🔮 SARIMA por HOMOLOGACIÓN")
    target = st.radio("Predecir", ["Primas", "Siniestros"], horizontal=True, key="homologacion")
    
    if st.button("🚀 Generar SARIMA Homologación", type="primary", use_container_width=True, key="btn_homologacion"):
        with st.spinner("Entrenando
