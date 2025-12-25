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

def sarima_por_compania(df_filt, compania, target_col, steps=5):
    mask = df_filt['COMPAÑÍA'] == compania
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

def calcular_sarima_completo(df_filt, target, steps=5, agrupacion='HOMOLOGACIÓN'):
    resultados = []
    grupos = df_filt[agrupacion].unique()
    
    for grupo in grupos:
        if agrupacion == 'HOMOLOGACIÓN':
            pred = sarima_por_homologacion(df_filt, grupo, target, steps)
        else:
            pred = sarima_por_compania(df_filt, grupo, target, steps)
        
        for i, mes in enumerate([8,9,10,11,12]):
            resultados.append({
                agrupacion: grupo,
                'Mes_Nombre': ['Agosto','Septiembre','Octubre','Noviembre','Diciembre'][i],
                'Predicción': pred[i]
            })
    
    return pd.DataFrame(resultados)

def calcular_promedio_mensual(df, agrupacion='HOMOLOGACIÓN'):
    mensual = df.groupby([agrupacion, 'YEAR', 'MONTH']).agg({
        'Primas': 'sum', 'Siniestros': 'sum'
    }).round(0)
    
    promedio_mensual = mensual.groupby([agrupacion, 'MONTH']).mean().round(0)
    promedio_mensual.columns = ['Promedio_Total_Primas', 'Promedio_Total_Siniestros']
    promedio_mensual = promedio_mensual.reset_index()
    
    mes_map = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
               7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
    promedio_mensual['Mes_Nombre'] = promedio_mensual['MONTH'].map(mes_map)
    
    return promedio_mensual.sort_values([agrupacion, 'MONTH'])

# === APP ===
st.title("🔥 SARIMA Predicción 2025")
st.markdown("**SARIMA por Homologación + Compañía**")

df = cargar_datos()
if df.empty:
    st.stop()

df_clean = preparar_datos(df)

# FILTROS SIMPLES
st.sidebar.header("🔍 Filtros")
homologacion_opts = sorted(df_clean['HOMOLOGACIÓN'].dropna().unique())
homologacion = st.sidebar.multiselect("Homologación", homologacion_opts, default=homologacion_opts[:5])

df_filt = df_clean[df_clean['HOMOLOGACIÓN'].isin(homologacion)]

# === TABS CON BOTONES INDEPENDIENTES ===
tab1, tab2 = st.tabs(["📊 Homologación", "🏢 Compañía"])

# TAB 1: HOMOLOGACIÓN
with tab1:
    st.header("🔮 SARIMA Homologación Agosto-Diciembre 2025")
    target_homo = st.radio("Predecir", ["Primas", "Siniestros"], horizontal=True, key="target_homo")
    
    if st.button("🚀 SARIMA Homologación", type="primary", key="btn_homo"):
        with st.spinner("Entrenando SARIMA Homologación..."):
            pred_homo = calcular_sarima_completo(df_filt, target_homo, agrupacion='HOMOLOGACIÓN')
            st.session_state.pred_homo = pred_homo
            st.session_state.target_homo = target_homo
            st.success("✅ SARIMA Homologación listo!")
            st.rerun()
    
    if 'pred_homo' in st.session_state:
        st.subheader("📈 Predicciones Homologación")
        pivot_homo = st.session_state.pred_homo.pivot(
            index='HOMOLOGACIÓN', columns='Mes_Nombre', values='Predicción'
        ).fillna(0).round(0)
        st.dataframe(pivot_homo, use_container_width=True)
        
        fig_homo = px.line(
            st.session_state.pred_homo, x='Mes_Nombre', y='Predicción',
            color='HOMOLOGACIÓN', title="SARIMA Homologación", markers=True
        )
        st.plotly_chart(fig_homo, use_container_width=True)

# TAB 2: COMPAÑÍA  
with tab2:
    st.header("🔮 SARIMA Compañía Agosto-Diciembre 2025")
    target_comp = st.radio("Predecir", ["Primas", "Siniestros"], horizontal=True, key="target_comp")
    
    if st.button("🚀 SARIMA Compañía", type="primary", key="btn_comp"):
        with st.spinner("Entrenando SARIMA Compañía..."):
            pred_comp = calcular_sarima_completo(df_filt, target_comp, agrupacion='COMPAÑÍA')
            st.session_state.pred_comp = pred_comp
            st.session_state.target_comp = target_comp
            st.success("✅ SARIMA Compañía listo!")
            st.rerun()
    
    if 'pred_comp' in st.session_state:
        st.subheader("📈 Predicciones Compañía")
        pivot_comp = st.session_state.pred_comp.pivot(
            index='COMPAÑÍA', columns='Mes_Nombre', values='Predicción'
        ).fillna(0).round(0)
        st.dataframe(pivot_comp, use_container_width=True)
        
        fig_comp = px.line(
            st.session_state.pred_comp, x='Mes_Nombre', y='Predicción',
            color='COMPAÑÍA', title="SARIMA Compañía", markers=True
        )
        st.plotly_chart(fig_comp, use_container_width=True)

# === PROMEDIOS HISTÓRICOS ===
st.header("📊 Promedios Históricos")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Homologación")
    tabla_homo = calcular_promedio_mensual(df_filt, 'HOMOLOGACIÓN')
    pivot_homo_hist = tabla_homo.pivot(index='HOMOLOGACIÓN', columns='Mes_Nombre', values='Promedio_Total_Primas').fillna(0).round(0)
    st.dataframe(pivot_homo_hist, use_container_width=True)

with col2:
    st.subheader("Compañía")
    tabla_comp = calcular_promedio_mensual(df_filt, 'COMPAÑÍA')
    pivot_comp_hist = tabla_comp.pivot(index='COMPAÑÍA', columns='Mes_Nombre', values='Promedio_Total_Primas').fillna(0).round(0)
    st.dataframe(pivot_comp_hist, use_container_width=True)

# === DESCARGAS ===
st.header("📥 Descargas")
col_dl1, col_dl2 = st.columns(2)

if 'pred_homo' in st.session_state:
    csv_homo = st.session_state.pred_homo.pivot(index='HOMOLOGACIÓN', columns='Mes_Nombre', values='Predicción').round(0).to_csv()
    col_dl1.download_button("SARIMA Homologación", csv_homo.encode(), f"sarima_homo_{pd.Timestamp.now().strftime('%Y%m%d')}.csv")

if 'pred_comp' in st.session_state:
    csv_comp = st.session_state.pred_comp.pivot(index='COMPAÑÍA', columns='Mes_Nombre', values='Predicción').round(0).to_csv()
    col_dl2.download_button("SARIMA Compañía", csv_comp.encode(), f"sarima_comp_{pd.Timestamp.now().strftime('%Y%m%d')}.csv")
