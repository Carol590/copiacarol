import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SARIMA Predicción Primas/Siniestros", layout="wide")

# === FILTRO GLOBAL DE CIUDAD EN SIDEBAR ===
if 'ciudad_filtro' not in st.session_state:
    st.session_state.ciudad_filtro = []

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

def aplicar_filtro_ciudad(df, ciudades_seleccionadas):
    """Aplica filtro de ciudad al dataframe"""
    if not ciudades_seleccionadas:
        return df
    return df[df['CIUDAD'].isin(ciudades_seleccionadas)]

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

# === SIDEBAR CON FILTRO DE CIUDAD ===
with st.sidebar:
    st.header("🔍 Filtros Globales")
    ciudades_disponibles = sorted(df_clean['CIUDAD'].unique())
    st.session_state.ciudad_filtro = st.multiselect(
        "Seleccionar Ciudad(es):",
        opciones=ciudades_disponibles,
        default=st.session_state.ciudad_filtro,
        key="filtro_ciudad_global"
    )
    
    if st.button("🔄 Limpiar Filtros", type="secondary"):
        st.session_state.ciudad_filtro = []
        st.rerun()

    st.info(f"📍 Ciudad(es) filtrada(s): {len(st.session_state.ciudad_filtro)}")
    if st.session_state.ciudad_filtro:
        st.caption(", ".join(st.session_state.ciudad_filtro))

# === APLICAR FILTRO DE CIUDAD A DATOS GLOBALES ===
df_filtrado = aplicar_filtro_ciudad(df_clean, st.session_state.ciudad_filtro)

# === TABS ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 SARIMA Homologación", "🚗 AUTOMÓVILES", "✅ CUMPLIMIENTO", 
    "🏢 GENERALES", "⚠️ RC", "🚑 SOAT", "💚 VIDA", "❌ NO SDE"
])

# === TAB 1: SARIMA por HOMOLOGACIÓN ===
with tab1:
    st.header("🔮 SARIMA por HOMOLOGACIÓN")
    st.info(f"📊 Datos filtrados: {len(df_filtrado):,} filas")
    target = st.radio("Predecir", ["Primas", "Siniestros"], horizontal=True, key="homologacion")
    
    if st.button("🚀 Generar SARIMA Homologación", type="primary", use_container_width=True, key="btn_homologacion"):
        with st.spinner("Entrenando SARIMA..."):
            st.session_state.pred_sarima = calcular_sarima_homologacion(df_filtrado, target)
            st.session_state.target = target
            st.success("✅ SARIMA Homologación listo!")

    if 'pred_sarima' in st.session_state:
        st.subheader("📈 Predicciones Agosto-Diciembre 2025")
        pivot_sarima = st.session_state.pred_sarima.pivot(
            index='HOMOLOGACIÓN', 
            columns='Mes_Nombre', 
            values='Predicción'
        ).fillna(0).round(0)
        st.dataframe(pivot_sarima, use_container_width=True)

        fig = px.line(
            st.session_state.pred_sarima, 
            x='Mes_Nombre', 
            y='Predicción',
            color='HOMOLOGACIÓN',
            title="SARIMA Predicciones Homologación 2025",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

# === TAB 2: AUTOMÓVILES ===
with tab2:
    df_auto = df_filtrado[df_filtrado['HOMOLOGACIÓN'] == 'AUTOMOVILES']
    st.header("🚗 SARIMA por COMPAÑÍA - AUTOMÓVILES")
    st.info(f"📊 Datos: {len(df_auto):,} filas")
    
    target_auto = st.radio("Predecir", ["Primas", "Siniestros"], horizontal=True, key="auto")
    
    if st.button("🚀 Generar SARIMA AUTOMÓVILES", type="primary", use_container_width=True, key="btn_auto"):
        with st.spinner("Entrenando SARIMA AUTOMÓVILES..."):
            st.session_state.pred_auto = calcular_sarima_compania(df_auto, target_auto)
            st.success("✅ SARIMA AUTOMÓVILES listo!")
    
    if 'pred_auto' in st.session_state:
        pivot_auto = st.session_state.pred_auto.pivot(
            index='COMPAÑÍA', 
            columns='Mes_Nombre', 
            values='Predicción'
        ).fillna(0).round(0)
        st.dataframe(pivot_auto, use_container_width=True)

# === Resto de tabs siguen igual pero usando df_filtrado ===
with tab3:
    df_cumpl = df_filtrado[df_filtrado['HOMOLOGACIÓN'] == 'CUMPLIMIENTO']
    st.header("✅ SARIMA por COMPAÑÍA - CUMPLIMIENTO")
    st.info(f"📊 Datos: {len(df_cumpl):,} filas")
    
    target_cumpl = st.radio("Predecir", ["Primas", "Siniestros"], horizontal=True, key="cumpl")
    
    if st.button("🚀 Generar SARIMA CUMPLIMIENTO", type="primary", use_container_width=True, key="btn_cumpl"):
        with st.spinner("Entrenando SARIMA CUMPLIMIENTO..."):
            st.session_state.pred_cumpl = calcular_sarima_compania(df_cumpl, target_cumpl)
            st.success("✅ SARIMA CUMPLIMIENTO listo!")
    
    if 'pred_cumpl' in st.session_state:
        pivot_cumpl = st.session_state.pred_cumpl.pivot(
            index='COMPAÑÍA', 
            columns='Mes_Nombre', 
            values='Predicción'
        ).fillna(0).round(0)
        st.dataframe(pivot_cumpl, use_container_width=True)

with tab4:
    df_gen = df_filtrado[df_filtrado['HOMOLOGACIÓN'] == 'GENERALES']
    st.header("🏢 SARIMA por COMPAÑÍA - GENERALES")
    st.info(f"📊 Datos: {len(df_gen):,} filas")
    
    target_gen = st.radio("Predecir", ["Primas", "Siniestros"], horizontal=True, key="gen")
    
    if st.button("🚀 Generar SARIMA GENERALES", type="primary", use_container_width=True, key="btn_gen"):
        with st.spinner("Entrenando SARIMA GENERALES..."):
            st.session_state.pred_gen = calcular_sarima_compania(df_gen, target_gen)
            st.success("✅ SARIMA GENERALES listo!")
    
    if 'pred_gen' in st.session_state:
        pivot_gen = st.session_state.pred_gen.pivot(
            index='COMPAÑÍA', 
            columns='Mes_Nombre', 
            values='Predicción'
        ).fillna(0).round(0)
        st.dataframe(pivot_gen, use_container_width=True)

with tab5:
    df_rc = df_filtrado[df_filtrado['HOMOLOGACIÓN'] == 'RC']
    st.header("⚠️ SARIMA por COMPAÑÍA - RC")
    st.info(f"📊 Datos: {len(df_rc):,} filas")
    
    target_rc = st.radio("Predecir", ["Primas", "Siniestros"], horizontal=True, key="rc")
    
    if st.button("🚀 Generar SARIMA RC", type="primary", use_container_width=True, key="btn_rc"):
        with st.spinner("Entrenando SARIMA RC..."):
            st.session_state.pred_rc = calcular_sarima_compania(df_rc, target_rc)
            st.success("✅ SARIMA RC listo!")
    
    if 'pred_rc' in st.session_state:
        pivot_rc = st.session_state.pred_rc.pivot(
            index='COMPAÑÍA', 
            columns='Mes_Nombre', 
            values='Predicción'
        ).fillna(0).round(0)
        st.dataframe(pivot_rc, use_container_width=True)

with tab6:
    df_soat = df_filtrado[df_filtrado['HOMOLOGACIÓN'] == 'SOAT']
    st.header("🚑 SARIMA por COMPAÑÍA - SOAT")
    st.info(f"📊 Datos: {len(df_soat):,} filas")
    
    target_soat = st.radio("Predecir", ["Primas", "Siniestros"], horizontal=True, key="soat")
    
    if st.button("🚀 Generar SARIMA SOAT", type="primary", use_container_width=True, key="btn_soat"):
        with st.spinner("Entrenando SARIMA SOAT..."):
            st.session_state.pred_soat = calcular_sarima_compania(df_soat, target_soat)
            st.success("✅ SARIMA SOAT listo!")
    
    if 'pred_soat' in st.session_state:
        pivot_soat = st.session_state.pred_soat.pivot(
            index='COMPAÑÍA', 
            columns='Mes_Nombre', 
            values='Predicción'
        ).fillna(0).round(0)
        st.dataframe(pivot_soat, use_container_width=True)

with tab7:
    df_vida = df_filtrado[df_filtrado['HOMOLOGACIÓN'] == 'VIDA']
    st.header("💚 SARIMA por COMPAÑÍA - VIDA")
    st.info(f"📊 Datos: {len(df_vida):,} filas")
    
    target_vida = st.radio("Predecir", ["Primas", "Siniestros"], horizontal=True, key="vida")
    
    if st.button("🚀 Generar SARIMA VIDA", type="primary", use_container_width=True, key="btn_vida"):
        with st.spinner("Entrenando SARIMA VIDA..."):
            st.session_state.pred_vida = calcular_sarima_compania(df_vida, target_vida)
            st.success("✅ SARIMA VIDA listo!")
    
    if 'pred_vida' in st.session_state:
        pivot_vida = st.session_state.pred_vida.pivot(
            index='COMPAÑÍA', 
            columns='Mes_Nombre', 
            values='Predicción'
        ).fillna(0).round(0)
        st.dataframe(pivot_vida, use_container_width=True)

with tab8:
    df_nosde = df_filtrado[df_filtrado['HOMOLOGACIÓN'] == 'NO SDE']
    st.header("❌ SARIMA por COMPAÑÍA - NO SDE")
    st.info(f"📊 Datos: {len(df_nosde):,} filas")
    
    target_nosde = st.radio("Predecir", ["Primas", "Siniestros"], horizontal=True, key="nosde")
    
    if st.button("🚀 Generar SARIMA NO SDE", type="primary", use_container_width=True, key="btn_nosde"):
        with st.spinner("Entrenando SARIMA NO SDE..."):
            st.session_state.pred_nosde = calcular_sarima_compania(df_nosde, target_nosde)
            st.success("✅ SARIMA NO SDE listo!")
    
    if 'pred_nosde' in st.session_state:
        pivot_nosde = st.session_state.pred_nosde.pivot(
            index='COMPAÑÍA', 
            columns='Mes_Nombre', 
            values='Predicción'
        ).fillna(0).round(0)
        st.dataframe(pivot_nosde, use_container_width=True)
