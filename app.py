import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
    """Prepara columnas EXACTAS que necesitas"""
    df.columns = df.columns.str.strip()
    
    # FECHA → YEAR, MONTH
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
    
    # SEPARAR PRIMAS vs SINIETROS
    if 'Primas/Siniestros' in df.columns:
        df['Primas'] = np.where(df['Primas/Siniestros'] == 'Primas', df['Valor_Mensual'], 0)
        df['Siniestros'] = np.where(df['Primas/Siniestros'] == 'Siniestros', df['Valor_Mensual'], 0)
    else:
        df['Primas'] = df['Valor_Mensual']
        df['Siniestros'] = 0
    
    # Homologación (OPCIONAL para filtro)
    if 'HOMOLOGACIÓN' in df.columns:
        df['HOMOLOGACIÓN'] = df['HOMOLOGACIÓN'].astype(str).str.strip()
    else:
        df['HOMOLOGACIÓN'] = 'SIN_HOMOLOGACION'
    
    return df.dropna(subset=['YEAR', 'MONTH'])

# === APP ===
st.title("📊 Primas y Siniestros por Año-Mes")
st.markdown("**Año | Mes | Promedio Primas | Total Primas | Promedio Siniestros | Total Siniestros**")

# CARGAR Y LIMPIAR
df = cargar_datos()
if df.empty:
    st.stop()

df_clean = preparar_datos(df)
st.success(f"✅ Datos listos: {len(df_clean):,} filas")

# Sidebar filtros
st.sidebar.header("🔍 Filtros")
homologacion_opts = sorted(df_clean['HOMOLOGACIÓN'].unique())
homologacion = st.sidebar.multiselect(
    "Homologación", 
    homologacion_opts, 
    default=homologacion_opts
)

df_filt = df_clean[df_clean['HOMOLOGACIÓN'].isin(homologacion)].copy()

# === TABLA EXACTA QUE PIDES ===
st.header("📈 Tabla: Año | Mes | Prom_Prim | Tot_Prim | Prom_Sin | Tot_Sin")

# AGRUPAR por YEAR, MONTH → LAS 6 COLUMNAS EXACTAS
tabla = df_filt.groupby(['YEAR', 'MONTH']).agg({
    'Primas': ['mean', 'sum'],
    'Siniestros': ['mean', 'sum']
}).round(0)

# RENOMBRAR EXACTAMENTE como pides
tabla.columns = [
    'Promedio_Primas', 'Total_Primas', 
    'Promedio_Siniestros', 'Total_Siniestros'
]
tabla = tabla.reset_index()

# ORDENAR por AÑO y MES
tabla = tabla.sort_values(['YEAR', 'MONTH'])

st.dataframe(tabla, use_container_width=True, height=600)

# === GRÁFICO ===
fig_line = px.line(
    tabla, 
    x='MONTH', 
    y=['Promedio_Primas', 'Promedio_Siniestros'],
    color_discrete_sequence=['#1f77b4', '#ff7f0e'],
    facet_col='YEAR',
    facet_col_wrap=4,
    title="📊 Promedio Mensual Primas vs Siniestros (por Año)",
    labels={'value': 'Promedio ($)', 'MONTH': 'Mes'}
)
fig_line.update_traces(line_shape="linear")
st.plotly_chart(fig_line, use_container_width=True)

# === GRÁFICO TOTALES ===
fig_bar = px.bar(
    tabla, 
    x='MONTH', 
    y=['Total_Primas', 'Total_Siniestros'],
    color_discrete_sequence=['#2ca02c', '#d62728'],
    facet_col='YEAR',
    facet_col_wrap=4,
    title="💰 Total Mensual Primas vs Siniestros (por Año)",
    labels={'value': 'Total ($)', 'MONTH': 'Mes'}
)
st.plotly_chart(fig_bar, use_container_width=True)

# === RESUMEN ANUAL ===
st.header("📅 Resumen Anual")
resumen_anual = df_filt.groupby('YEAR').agg({
    'Primas': ['mean', 'sum'],
    'Siniestros': ['mean', 'sum']
}).round(0)

resumen_anual.columns = [
    'Promedio_Primas', 'Total_Primas', 
    'Promedio_Siniestros', 'Total_Siniestros'
]
resumen_anual = resumen_anual.reset_index()

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Primas", f"${resumen_anual['Total_Primas'].sum():,.0f}")
col2.metric("💰 Total Siniestros", f"${resumen_anual['Total_Siniestros'].sum():,.0f}")
col3.metric("📊 Promedio Primas", f"${resumen_anual['Promedio_Primas'].mean():,.0f}")
col4.metric("📊 Promedio Siniestros", f"${resumen_anual['Promedio_Siniestros'].mean():,.0f}")

st.dataframe(resumen_anual, use_container_width=True)

# === TOP MESES ===
st.header("🔥 Top 10 Meses (Total Primas + Siniestros)")
tabla['Total_General'] = tabla['Total_Primas'] + tabla['Total_Siniestros']
top_meses = tabla.nlargest(10, 'Total_General')[
    ['YEAR', 'MONTH', 'Total_Primas', 'Total_Siniestros', 'Total_General']
]
st.dataframe(top_meses, use_container_width=True)

# === DESCARGA ===
csv = tabla.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Descargar TABLA (Año|Mes|Prim|Sini)",
    data=csv,
    file_name=f"tabla_primas_siniestros_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
    mime='text/csv'
)

# Vista datos originales
with st.expander("🔎 Datos originales"):
    st.dataframe(
        df_filt[['YEAR', 'MONTH', 'HOMOLOGACIÓN', 'Primas/Siniestros', 'Primas', 'Siniestros']]
        .head(1000), 
        height=400
    )

st.markdown("---")
st.caption("✅ TABLA EXACTA: Año | Mes | Prom_Primas | Total_Primas | Prom_Siniestros | Total_Siniestros")
