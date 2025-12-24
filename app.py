import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Predicción Primas/Siniestros", layout="wide")

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
    """Prepara datos completos con filtros"""
    df.columns = df.columns.str.strip()
    
    # FECHA
    if 'FECHA' in df.columns:
        df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
        df['YEAR'] = df['FECHA'].dt.year
        df['MONTH'] = df['FECHA'].dt.month
    
    # Valor numérico
    if 'Valor_Mensual' in df.columns:
        df['Valor_Mensual'] = pd.to_numeric(df['Valor_Mensual'], errors='coerce').fillna(0)
    
    # PRIMAS vs SINIETROS
    if 'Primas/Siniestros' in df.columns:
        df['Primas'] = np.where(df['Primas/Siniestros'] == 'Primas', df['Valor_Mensual'], 0)
        df['Siniestros'] = np.where(df['Primas/Siniestros'] == 'Siniestros', df['Valor_Mensual'], 0)
    
    # Columnas para filtros
    for col in ['HOMOLOGACIÓN', 'CIUDAD', 'COMPAÑÍA']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    return df.dropna(subset=['YEAR', 'MONTH'])

def calcular_promedio_mensual(df):
    """Promedio total mensual por Homologación"""
    mensual = df.groupby(['HOMOLOGACIÓN', 'YEAR', 'MONTH']).agg({
        'Primas': 'sum', 'Siniestros': 'sum'
    }).round(0)
    
    promedio_mensual = mensual.groupby(['HOMOLOGACIÓN', 'MONTH']).mean().round(0)
    promedio_mensual.columns = ['Promedio_Total_Primas', 'Promedio_Total_Siniestros']
    promedio_mensual = promedio_mensual.reset_index()
    
    # Nombres meses
    mes_map = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
               7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
    promedio_mensual['Mes_Nombre'] = promedio_mensual['MONTH'].map(mes_map)
    
    return promedio_mensual.sort_values(['HOMOLOGACIÓN', 'MONTH'])

def encode_categorical(df, cols):
    """Encoding SEGURO con diccionario reversible"""
    encoders = {}
    reverse_encoders = {}
    
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            unique_vals = sorted(df[col].unique())
            encoders[col] = {val: i for i, val in enumerate(unique_vals)}
            reverse_encoders[col] = {i: val for val, i in encoders[col].items()}
            df[col] = df[col].map(encoders[col]).fillna(0).astype(int)
    
    return df, encoders, reverse_encoders

def entrenar_xgboost(df_filt, target_col):
    """XGBoost con encoding reversible"""
    features = ['MONTH', 'YEAR']
    cat_cols = []
    for col in ['HOMOLOGACIÓN', 'CIUDAD', 'COMPAÑÍA']:
        if col in df_filt.columns:
            features.append(col)
            cat_cols.append(col)
    
    X, encoders, reverse_encoders = encode_categorical(df_filt[features].copy(), cat_cols)
    y = df_filt[target_col].fillna(0)
    
    if len(X) < 20:
        return None, "Datos insuficientes", None, None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, {'mae': mae, 'r2': r2, 'features': features}, encoders, reverse_encoders

def preparar_predicciones_con_nombres(model, tabla_promedios, encoders, reverse_encoders, features):
    """✅ Predicciones con NOMBRES de Homologación"""
    meses_futuros = [8,9,10,11,12]
    future_data = []
    
    # Diccionario para mapear números → nombres
    homolog_map = reverse_encoders.get('HOMOLOGACIÓN', {})
    
    for homolog_num, homolog_nombre in homolog_map.items():
        if homolog_nombre in tabla_promedios['HOMOLOGACIÓN'].values:
            for mes in meses_futuros:
                row = {'YEAR': 2025, 'MONTH': mes, 'HOMOLOGACIÓN': int(homolog_num)}
                
                # Encoding para otras categorías (usar primera categoría)
                for col in ['CIUDAD', 'COMPAÑÍA']:
                    if col in encoders and encoders[col]:
                        row[col] = 0  # Primera categoría
                
                future_data.append(row)
    
    future_df = pd.DataFrame(future_data)
    
    # Asegurar columnas
    for col in features:
        if col not in future_df.columns:
            future_df[col] = 0
    
    future_df = future_df[features]
    predicciones = model.predict(future_df)
    
    # ✅ AGREGAR NOMBRES de Homologación
    future_df['HOMOLOGACIÓN_NOMBRE'] = future_df['HOMOLOGACIÓN'].map(homolog_map)
    future_df['Mes_Nombre'] = future_df['MONTH'].map({8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'})
    future_df['Predicción'] = predicciones.round(0)
    
    # Agrupar por Homologación y Mes
    pred_tabla = future_df.groupby(['HOMOLOGACIÓN_NOMBRE', 'Mes_Nombre'])['Predicción'].sum().reset_index()
    
    return pred_tabla

# === APP ===
st.title("🔮 Predicción Primas/Siniestros 2025")
st.markdown("**✅ Nombres Homologación en Predicciones**")

# CARGAR DATOS
df = cargar_datos()
if df.empty:
    st.stop()

df_clean = preparar_datos(df)

# === FILTROS ===
st.sidebar.header("🔍 Filtros")
homologacion_opts = sorted(df_clean['HOMOLOGACIÓN'].dropna().unique())
ciudad_opts = sorted(df_clean['CIUDAD'].dropna().unique()) if 'CIUDAD' in df_clean else ['TODAS']
compania_opts = sorted(df_clean['COMPAÑÍA'].dropna().unique()) if 'COMPAÑÍA' in df_clean else ['TODAS']

homologacion = st.sidebar.multiselect("Homologación", homologacion_opts, default=homologacion_opts[:3])
ciudad = st.sidebar.multiselect("Ciudad", ciudad_opts, default=ciudad_opts[:3] if ciudad_opts != ['TODAS'] else ciudad_opts)
compania = st.sidebar.multiselect("Compañía", compania_opts, default=compania_opts[:3] if compania_opts != ['TODAS'] else compania_opts)

df_filt = df_clean.copy()
if homologacion:
    df_filt = df_filt[df_filt['HOMOLOGACIÓN'].isin(homologacion)]
if ciudad != ['TODAS'] and ciudad:
    df_filt = df_filt[df_filt['CIUDAD'].isin(ciudad)]
if compania != ['TODAS'] and compania:
    df_filt = df_filt[df_filt['COMPAÑÍA'].isin(compania)]

# === MÉTRICAS GLOBALES ===
st.header("📊 Métricas Globales")
tabla_promedios = calcular_promedio_mensual(df_filt)

promedio_primas_general = df_filt['Primas'].mean()
promedio_sini_general = df_filt['Siniestros'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Promedio Mensual Primas", f"${promedio_primas_general:,.0f}")
col2.metric("💰 Promedio Mensual Siniestros", f"${promedio_sini_general:,.0f}")
col3.metric("📈 Homologaciones", len(tabla_promedios['HOMOLOGACIÓN'].unique()))
col4.metric("📅 Años", f"{df_filt['YEAR'].min()}-{df_filt['YEAR'].max()}")

# === XGBoost ===
st.header("🔮 Predicción XGBoost Agosto-Diciembre 2025")
target = st.radio("Predecir", ["Primas", "Siniestros"])

if st.button("🚀 Entrenar y Predecir", type="primary", use_container_width=True):
    with st.spinner("Entrenando XGBoost..."):
        model, results, encoders, reverse_encoders = entrenar_xgboost(df_filt, target)
        if model:
            st.session_state.model = model
            st.session_state.results = results
            st.session_state.encoders = encoders
            st.session_state.reverse_encoders = reverse_encoders
            st.session_state.target = target
            st.session_state.tabla_promedios = tabla_promedios
            st.success("✅ Modelo entrenado!")
            st.rerun()

if 'model' in st.session_state:
    st.subheader("📈 Predicciones 2025 (Agosto-Diciembre)")
    
    # ✅ NOMBRES REALES de Homologación
    pred_tabla = preparar_predicciones_con_nombres(
        st.session_state.model,
        st.session_state.tabla_promedios,
        st.session_state.encoders,
        st.session_state.reverse_encoders,
        st.session_state.results['features']
    )
    
    # PIVOT con NOMBRES
    pivot_pred = pred_tabla.pivot(index='HOMOLOGACIÓN_NOMBRE', columns='Mes_Nombre', values='Predicción').fillna(0).round(0)
    st.dataframe(pivot_pred, use_container_width=True)
    
    # Métricas
    col1, col2 = st.columns(2)
    col1.metric("✅ MAE", f"${st.session_state.results['mae']:,.0f}")
    col2.metric("✅ R²", f"{st.session_state.results['r2']:.1%}")

# === PROMEDIOS ===
st.header("📊 Promedio Total Mensual por Homologación")
pivot_promedios = tabla_promedios.pivot(index='HOMOLOGACIÓN', columns='Mes_Nombre', values='Promedio_Total_Primas').fillna(0).round(0)
st.dataframe(pivot_promedios, use_container_width=True)

fig = px.line(tabla_promedios, x='Mes_Nombre', y='Promedio_Total_Primas', color='HOMOLOGACIÓN', markers=True)
st.plotly_chart(fig, use_container_width=True)

# DESCARGA
csv = tabla_promedios.to_csv(index=False).encode('utf-8')
st.download_button("📥 Descargar CSV", csv, f"predicciones_{pd.Timestamp.now().strftime('%Y%m%d')}.csv")
