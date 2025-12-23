import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Datos Seguros", layout="wide")

@st.cache_data(ttl=300)
def cargar_datos():
    """Carga datos del Google Sheet"""
    url = "https://docs.google.com/spreadsheets/d/1VljNnZtRPDA3TkTUP6w8AviZCPIfILqe/export?format=csv&gid=293107109"
    
    try:
        df = pd.read_csv(url)
        st.success(f"✅ {len(df):,} filas cargadas")
        return df
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return pd.DataFrame()

def preparar_datos(df):
    """Prepara YEAR, MONTH, Valor_Mensual"""
    df.columns = df.columns.str.strip()
    
    # FECHA
    if 'FECHA' in df.columns:
        df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
        df['YEAR'] = df['FECHA'].dt.year
        df['MONTH'] = df['FECHA'].dt.month
    else:
        df['YEAR'] = 2023
        df['MONTH'] = 1
    
    # Valor numérico
    if 'Valor_Mensual' in df.columns:
        df['Valor_Mensual'] = pd.to_numeric(df['Valor_Mensual'], errors='coerce').fillna(0)
    else:
        df['Valor_Mensual'] = 0
    
    # Homologación
    if 'HOMOLOGACIÓN' in df.columns:
        df['HOMOLOGACIÓN'] = df['HOMOLOGACIÓN'].astype(str).str.strip()
    else:
        df['HOMOLOGACIÓN'] = 'SIN_HOMOLOGACION'
    
    return df.dropna(subset=['YEAR', 'MONTH'])

# === APP ===
st.title("📊 Promedios por Mes/Año/Homologación")
st.markdown("**Datos Seguros Colombia 2020-2025**")

# CARGAR Y LIMPIAR
df = cargar_datos()
if df.empty:
    st.stop()

df_clean = preparar_datos(df)
st.success(f"✅ Datos limpios: {len(df_clean):,} filas")

# Sidebar filtros
st.sidebar.header("🔍 Filtros")
homologacion_opts = sorted(df_clean['HOMOLOGACIÓN'].unique())
homologacion = st.sidebar.multiselect(
    "Homologación", 
    homologacion_opts, 
    default=homologacion_opts
)

df_filt = df_clean[df_clean['HOMOLOGACIÓN'].isin(homologacion)].copy()

# === 1. PROMEDIOS POR MES/AÑO/HOMOLOGACIÓN ===
st.header("📈 Promedios Mensuales por Homologación")

# Agrupar: PROMEDIO por YEAR, MONTH, HOMOLOGACIÓN
promedios = df_filt.groupby(['YEAR', 'MONTH', 'HOMOLOGACIÓN'])['Valor_Mensual'].agg(['mean', 'sum', 'count']).round(0)
promedios.columns = ['Promedio', 'Total', 'N_Filas']
promedios = promedios.reset_index()
promedios['Mes_Año'] = promedios['YEAR'].astype(str) + '-' + promedios['MONTH'].astype(str).str.zfill(2)

st.dataframe(promedios, use_container_width=True)

# === 2. GRÁFICO PROMEDIOS ===
import plotly.express as px
fig = px.line(
    promedios, 
    x='Mes_Año', 
    y='Promedio',
    color='HOMOLOGACIÓN',
    title="📊 Promedio Mensual por Homologación",
    markers=True
)
fig.update_layout(xaxis_tickangle=-45, height=500)
st.plotly_chart(fig, use_container_width=True)

# === 3. RESUMEN POR HOMOLOGACIÓN ===
st.header("🏢 Resumen por Homologación")

resumen_homo = df_filt.groupby('HOMOLOGACIÓN')['Valor_Mensual'].agg([
    'count', 'mean', 'sum', 'std'
]).round(0)
resumen_homo.columns = ['N_Filas', 'Promedio', 'Total', 'Desviación']
resumen_homo = resumen_homo.sort_values('Total', ascending=False)
st.dataframe(resumen_homo, use_container_width=True)

# Métricas principales
col1, col2, col3, col4 = st.columns(4)
total_general = df_filt['Valor_Mensual'].sum()
col1.metric("💰 Total General", f"${total_general:,.0f}")
col2.metric("📊 Promedio", f"${df_filt['Valor_Mensual'].mean():,.0f}")
col3.metric("📈 Homologaciones", len(df_filt['HOMOLOGACIÓN'].unique()))
col4.metric("📅 Años", f"{df_filt['YEAR'].min()}-{df_filt['YEAR'].max()}")

# === 4. TOP 10 MESES MÁS ALTOS ===
st.header("🔥 Top 10 Meses (Total)")
top_meses = promedios.nlargest(10, 'Total')[['Mes_Año', 'HOMOLOGACIÓN', 'Total', 'Promedio']]
st.dataframe(top_meses, use_container_width=True)

# === 5. TABLA COMPLETA FILTRADA ===
with st.expander("📋 Ver TODOS los datos filtrados"):
    st.dataframe(df_filt[['YEAR', 'MONTH', 'HOMOLOGACIÓN', 'Valor_Mensual']].sort_values('YEAR'), height=400)

# === DESCARGA ===
csv = df_filt.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Descargar CSV filtrado",
    data=csv,
    file_name=f"seguros_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
    mime='text/csv'
)

st.markdown("---")
st.caption("✅ Paso 2 completado | Siguiente: Modelos XGBoost")
