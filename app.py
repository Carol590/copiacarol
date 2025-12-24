import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Promedio Mensual por Homologación", layout="wide")

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
    """Prepara datos con Primas y Siniestros separados"""
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
    
    # Homologación
    if 'HOMOLOGACIÓN' in df.columns:
        df['HOMOLOGACIÓN'] = df['HOMOLOGACIÓN'].astype(str).str.strip()
    else:
        df['HOMOLOGACIÓN'] = 'SIN_HOMOLOGACION'
    
    return df.dropna(subset=['YEAR', 'MONTH'])

def calcular_promedio_mensual(df):
    """🎯 PROMEDIO del TOTAL mensual por Homologación (todos los años)"""
    
    # PASO 1: Total mensual POR AÑO y Homologación
    mensual = (
        df.groupby(['HOMOLOGACIÓN', 'YEAR', 'MONTH'], as_index=False)
        .agg({
            'Primas': 'sum',
            'Siniestros': 'sum'
        })
        .round(0)
    )
    mensual.rename(columns={
        'Primas': 'Total_Primas_mensual',
        'Siniestros': 'Total_Siniestros_mensual'
    }, inplace=True)
    
    # PASO 2: PROMEDIO de esos totales mensuales (todos los años)
    promedio_mensual = (
        mensual.groupby(['HOMOLOGACIÓN', 'MONTH'], as_index=False)
        .agg({
            'Total_Primas_mensual': 'mean',
            'Total_Siniestros_mensual': 'mean'
        })
        .round(0)
    )
    
    # NOMBRES FINALES
    promedio_mensual.columns = [
        'HOMOLOGACIÓN', 'Mes', 
        'Promedio_Total_Primas', 'Promedio_Total_Siniestros'
    ]
    
    # ORDENAR meses 1-12
    promedio_mensual['Mes_Nombre'] = promedio_mensual['Mes'].map({
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    })
    
    return promedio_mensual.sort_values(['HOMOLOGACIÓN', 'Mes'])

# === APP PRINCIPAL ===
st.title("📊 Promedio del Total Mensual por Homologación")
st.markdown("**Promedio del total mensual de TODOS los años, por cada Homologación y Mes**")

# CARGAR DATOS
df = cargar_datos()
if df.empty:
    st.stop()

df_clean = preparar_datos(df)
st.success(f"✅ Datos preparados: {len(df_clean):,} filas | {df_clean['YEAR'].min()}-{df_clean['YEAR'].max()}")

# FILTROS
st.sidebar.header("🔍 Filtros")
homologacion_opts = sorted(df_clean['HOMOLOGACIÓN'].unique())
homologacion = st.sidebar.multiselect(
    "Homologación", 
    homologacion_opts, 
    default=homologacion_opts[:5]  # Top 5 por defecto
)

df_filt = df_clean[df_clean['HOMOLOGACIÓN'].isin(homologacion)].copy()

# === TABLA PRINCIPAL ===
st.header("🎯 Promedio Total Mensual por Homologación")
tabla_promedios = calcular_promedio_mensual(df_filt)

st.dataframe(tabla_promedios, use_container_width=True, height=600)

# === GRÁFICO PRINCIPAL ===
st.header("📈 Gráfico: Promedio Mensual por Homologación")
fig_line = px.line(
    tabla_promedios,
    x='Mes_Nombre',
    y=['Promedio_Total_Primas', 'Promedio_Total_Siniestros'],
    color='HOMOLOGACIÓN',
    title="Promedio del Total Mensual (todos los años)",
    markers=True
)
fig_line.update_layout(height=500, xaxis_tickangle=-45)
st.plotly_chart(fig_line, use_container_width=True)

# === RESUMEN POR HOMOLOGACIÓN ===
st.header("🏢 Resumen Anual Promedio por Homologación")
resumen_homo = tabla_promedios.groupby('HOMOLOGACIÓN').agg({
    'Promedio_Total_Primas': 'mean',
    'Promedio_Total_Siniestros': 'mean'
}).round(0)

resumen_homo['Promedio_Total_General'] = (
    resumen_homo['Promedio_Total_Primas'] + resumen_homo['Promedio_Total_Siniestros']
)
resumen_homo = resumen_homo.sort_values('Promedio_Total_General', ascending=False)
st.dataframe(resumen_homo, use_container_width=True)

# MÉTRICAS GLOBALES
col1, col2, col3, col4 = st.columns(4)
total_primas_prom = tabla_promedios['Promedio_Total_Primas'].sum()
total_sini_prom = tabla_promedios['Promedio_Total_Siniestros'].sum()
col1.metric("💰 Promedio Anual Primas", f"${total_primas_prom:,.0f}")
col2.metric("💰 Promedio Anual Siniestros", f"${total_sini_prom:,.0f}")
col3.metric("📈 Homologaciones", len(tabla_promedios['HOMOLOGACIÓN'].unique()))
col4.metric("📅 Meses", tabla_promedios['Mes'].nunique())

# === TOP 5 HOMOLOGACIONES ===
st.header("🔥 Top 5 Homologaciones (Promedio Total Anual)")
top_5 = resumen_homo.head(5)
fig_bar = px.bar(
    top_5.reset_index(),
    x='HOMOLOGACIÓN',
    y=['Promedio_Total_Primas', 'Promedio_Total_Siniestros'],
    title="Top 5 Homologaciones por Promedio Anual",
    barmode='group'
)
st.plotly_chart(fig_bar, use_container_width=True)

# === DETALLE CÁLCULO ===
with st.expander("🔎 Cómo se calcula"):
    st.markdown("""
    **1. Total mensual por Año-Homologación**
    ```
    df.groupby(['HOMOLOGACIÓN', 'YEAR', 'MONTH'])['Primas'].sum()
    ```
    
    **2. Promedio de esos totales (todos los años)**
    ```
    mensual.groupby(['HOMOLOGACIÓN', 'MONTH'])['Total_Primas'].mean()
    ```
    
    **Resultado**: Para cada Homologación y Mes → promedio del total mensual de todos los años
    """)

# === DESCARGAS ===
st.header("📥 Descargas")
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    csv_principal = tabla_promedios.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📊 Tabla Principal CSV",
        data=csv_principal,
        file_name=f"promedio_mensual_homologacion_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )
with col_dl2:
    excel_data = tabla_promedios.to_excel(index=False)
    st.download_button(
        label="📊 Tabla Principal Excel",
        data=excel_data,
        file_name=f"promedio_mensual_homologacion_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

st.markdown("---")
st.caption("✅ **PROMEDIO del TOTAL mensual por Homologación (todos los años)**")
