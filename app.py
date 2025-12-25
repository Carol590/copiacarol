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

def sarima_por_homologacion(df_filt, homologacion, target_col, steps=5):
    """SARIMA ORIGINAL - FUNCIONAL"""
    mask = df_filt['HOMOLOGACIÓN'] == homologacion
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

def calcular_sarima_completo(df_filt, target, steps=5):
    resultados = []
    homologaciones = df_filt['HOMOLOGACIÓN'].unique()
    
    for homologacion in homologaciones:
        pred = sarima_por_homologacion(df_filt, homologacion, target, steps)
        
        for i, mes in enumerate([8,9,10,11,12]):
            resultados.append({
                'HOMOLOGACIÓN': homologacion,
                'Mes_Nombre': ['Agosto','Septiembre','Octubre','Noviembre','Diciembre'][i],
                'Predicción': pred[i]
            })
    
    return pd.DataFrame(resultados)

def calcular_promedio_mensual(df):
    mensual = df.groupby(['HOMOLOGACIÓN', 'YEAR', 'MONTH']).agg({
        'Primas': 'sum', 'Siniestros': 'sum'
    }).round(0)
    
    promedio_mensual = mensual.groupby(['HOMOLOGACIÓN', 'MONTH']).mean().round(0)
    promedio_mensual.columns = ['Promedio_Total_Primas', 'Promedio_Total_Siniestros']
    promedio_mensual = promedio_mensual.reset_index()
    
    mes_map = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
               7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
    promedio_mensual['Mes_Nombre'] = promedio_mensual['MONTH'].map(mes_map)
    
    return promedio_mensual.sort_values(['HOMOLOGACIÓN', 'MONTH'])

def calcular_promedio_compania(df):
    """🎯 NUEVA: Promedio por COMPAÑÍA"""
    mensual_comp = df.groupby(['COMPAÑÍA', 'YEAR', 'MONTH']).agg({
        'Primas': 'sum', 'Siniestros': 'sum'
    }).round(0)
    
    promedio_comp = mensual_comp.groupby(['COMPAÑÍA', 'MONTH']).mean().round(0)
    promedio_comp.columns = ['Promedio_Total_Primas', 'Promedio_Total_Siniestros']
    promedio_comp = promedio_comp.reset_index()
    
    mes_map = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
               7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
    promedio_comp['Mes_Nombre'] = promedio_comp['MONTH'].map(mes_map)
    
    return promedio_comp.sort_values(['COMPAÑÍA', 'MONTH'])

# === APP ORIGINAL ===
st.title("🔥 SARIMA Predicción 2025")
st.markdown("**SARIMA ORIGINAL + Tabla por COMPAÑÍA**")

df = cargar_datos()
if df.empty:
    st.stop()

df_clean = preparar_datos(df)

# === FILTROS SIMPLES ===
st.sidebar.header("🔍 Filtros")
homologacion_opts = sorted(df_clean['HOMOLOGACIÓN'].dropna().unique())
homologacion = st.sidebar.multiselect("Homologación", homologacion_opts, default=homologacion_opts[:5])

df_filt = df_clean[df_clean['HOMOLOGACIÓN'].isin(homologacion)]

# === MÉTRICAS GLOBALES ===
st.header("📊 Métricas Globales")
tabla_promedios = calcular_promedio_mensual(df_filt)

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Promedio Primas", f"${df_filt['Primas'].mean():,.0f}")
col2.metric("💰 Promedio Siniestros", f"${df_filt['Siniestros'].mean():,.0f}")
col3.metric("📈 Homologaciones", len(tabla_promedios['HOMOLOGACIÓN'].unique()))
col4.metric("📅 Años", f"{df_filt['YEAR'].min()}-{df_filt['YEAR'].max()}")

# === SARIMA ORIGINAL ===
st.header("🔮 SARIMA Predicción Agosto-Diciembre 2025")
target = st.radio("Predecir", ["Primas", "Siniestros"], horizontal=True)

if st.button("🚀 Generar SARIMA", type="primary", use_container_width=True):
    with st.spinner("Entrenando SARIMA..."):
        st.session_state.pred_sarima = calcular_sarima_completo(df_filt, target)
        st.session_state.target = target
        st.session_state.df_filt = df_filt
        st.success("✅ SARIMA listo!")

if 'pred_sarima' in st.session_state:
    st.subheader("📈 Predicciones SARIMA 2025")
    
    pivot_sarima = st.session_state.pred_sarima.pivot(
        index='HOMOLOGACIÓN', 
        columns='Mes_Nombre', 
        values='Predicción'
    ).fillna(0).round(0)
    
    st.dataframe(pivot_sarima, use_container_width=True)

# === NUEVA TABLA: PROMEDIOS POR COMPAÑÍA ===
st.header("🏢 Promedios por COMPAÑÍA")
tabla_compania = calcular_promedio_compania(df_filt)

pivot_compania = tabla_compania.pivot(
    index='COMPAÑÍA', 
    columns='Mes_Nombre', 
    values='Promedio_Total_Primas'
).fillna(0).round(0)

st.dataframe(pivot_compania, use_container_width=True)

# === PROMEDIOS HISTÓRICOS HOMOLOGACIÓN ===
st.header("📊 Promedios Históricos Homologación")
pivot_hist = tabla_promedios.pivot(
    index='HOMOLOGACIÓN', 
    columns='Mes_Nombre', 
    values='Promedio_Total_Primas'
).fillna(0).round(0)
st.dataframe(pivot_hist, use_container_width=True)

# === GRÁFICO ===
if 'pred_sarima' in st.session_state:
    fig = px.line(
        st.session_state.pred_sarima, 
        x='Mes_Nombre', 
        y='Predicción',
        color='HOMOLOGACIÓN',
        title="SARIMA Predicciones Agosto-Diciembre 2025",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

# DESCARGA
if 'pred_sarima' in st.session_state:
    csv_sarima = pivot_sarima.to_csv().encode('utf-8')
    csv_compania = pivot_compania.to_csv().encode('utf-8')
    
    col1, col2 = st.columns(2)
    col1.download_button("📥 SARIMA Homologación", csv_sarima, f"sarima_{pd.Timestamp.now().strftime('%Y%m%d')}.csv")
    col2.download_button("📥 Promedios Compañía", csv_compania, f"compania_{pd.Timestamp.now().strftime('%Y%m%d')}.csv")
