import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Datos Seguros", layout="wide")

@st.cache_data(ttl=300)  # Cache 5min
def cargar_datos():
    """Carga SOLO los datos del Google Sheet"""
    url = "https://docs.google.com/spreadsheets/d/1VljNnZtRPDA3TkTUP6w8AviZCPIfILqe/export?format=csv&gid=293107109"
    
    try:
        df = pd.read_csv(url)
        st.success(f"✅ ¡Datos cargados! {len(df):,} filas")
        return df
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return pd.DataFrame()

st.title("📊 Datos Seguros Colombia")
st.markdown("**Paso 1: Solo cargar y ver datos del Google Sheet**")

# CARGAR DATOS
df = cargar_datos()

if df.empty:
    st.stop()

# MOSTRAR INFO BÁSICA
col1, col2, col3 = st.columns(3)
col1.metric("Filas", f"{len(df):,}")
col2.metric("Columnas", len(df.columns))
col3.metric("No nulos", f"{df.count().sum():,}")

# PREVIEW TABLA
st.subheader("🔍 Vista previa")
st.dataframe(df.head(10), use_container_width=True)

# INFO COLUMNAS
st.subheader("📋 Columnas encontradas")
column_info = pd.DataFrame({
    'Columna': df.columns,
    'Tipo': [str(df[col].dtype) for col in df.columns],
    'No nulos': [df[col].count() for col in df.columns],
    'Valores únicos': [df[col].nunique() for col in df.columns]
})
st.dataframe(column_info, use_container_width=True)

# ESTADÍSTICAS BÁSICO
st.subheader("📈 Estadísticas")
st.dataframe(df.describe(), use_container_width=True)

# MUESTRA CRUDA (para debug)
with st.expander("🔎 Ver DATOS CRUDOS (primeras 50 filas)"):
    st.dataframe(df.head(50), height=400)
