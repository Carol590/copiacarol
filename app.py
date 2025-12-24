import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
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
    """Prepara datos con PESO reciente"""
    df.columns = df.columns.str.strip()
    
    # FECHA
    if 'FECHA' in df.columns:
        df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
        df['YEAR'] = df['FECHA'].dt.year
        df['MONTH'] = df['FECHA'].dt.month
        df = df.sort_values('FECHA')
    
    # Valor numérico
    if 'Valor_Mensual' in df.columns:
        df['Valor_Mensual'] = pd.to_numeric(df['Valor_Mensual'], errors='coerce').fillna(0)
    
    # PRIMAS vs SINIETROS
    if 'Primas/Siniestros' in df.columns:
        df['Primas'] = np.where(df['Primas/Siniestros'] == 'Primas', df['Valor_Mensual'], 0)
        df['Siniestros'] = np.where(df['Primas/Siniestros'] == 'Siniestros', df['Valor_Mensual'], 0)
    
    # PESO RECIENTE: últimos 2 años x2, últimos 3 años x1.5
    df['peso'] = 1.0
    recent_years = df['YEAR'].max()
    df.loc[df['YEAR'] >= recent_years-1, 'peso'] = 2.0  # Últimos 2 años
    df.loc[df['YEAR'] >= recent_years-2, 'peso'] = 1.5  # Últimos 3 años
    
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

def sarima_modelo_hibrido(df_filt, homologacion, target_col, steps=5):
    """🔥 SARIMA optimizado por homologación"""
    mask = df_filt['HOMOLOGACIÓN'] == homologacion
    if mask.sum() < 24:  # Min 2 años datos
        return np.full(steps, df_filt[target_col].mean())
    
    # Serie temporal mensual
    series = df_filt.loc[mask].groupby('FECHA')['Primas' if target_col=='Primas' else 'Siniestros'].sum()
    
    try:
        # SARIMA(1,1,1)(1,1,1,12) + peso reciente
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12))
        fitted = model.fit(disp=False)
        forecast = fitted.get_forecast(steps=steps)
        return forecast.predicted_mean.values.round(0)
    except:
        return np.full(steps, series.tail(6).mean())  # Promedio últimos 6 meses

def entrenar_xgboost(df_filt, target_col):
    """XGBoost MEJORADO con features temporales"""
    features = ['MONTH', 'YEAR']
    cat_cols = []
    for col in ['HOMOLOGACIÓN', 'CIUDAD', 'COMPAÑÍA']:
        if col in df_filt.columns:
            features.append(col)
            cat_cols.append(col)
    
    # Encoding
    X, encoders, reverse_encoders = encode_categorical(df_filt[features].copy(), cat_cols)
    y = df_filt[target_col].fillna(0) * df_filt['peso']  # ✅ PESO RECIENTE
    
    if len(X) < 30:
        return None, None, None, None
    
    # ✅ FEATURES TEMPORALES MEJORADAS
    X['trimestre'] = ((X['MONTH'] - 1) // 3 + 1)
    X['es_fin_año'] = (X['MONTH'] >= 10).astype(int)
    X['es_verano'] = (X['MONTH'].isin([6,7,8])).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ✅ XGBOOST OPTIMIZADO
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.07,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, X.columns.tolist(), {'mae': mae, 'r2': r2}, encoders, reverse_encoders

def ensemble_predicciones(model, features, tabla_promedios, encoders, reverse_encoders, df_filt, target):
    """🎯 ENSEMBLE: 60% XGBoost + 40% SARIMA"""
    meses_futuros = [8,9,10,11,12]
    homolog_map = reverse_encoders.get('HOMOLOGACIÓN', {})
    
    resultados = []
    
    for homolog_num, homolog_nombre in homolog_map.items():
        if homolog_nombre in tabla_promedios['HOMOLOGACIÓN'].values:
            # 1. XGBoost
            future_row = pd.DataFrame([{
                'YEAR': 2025, 'MONTH': meses_futuros[0],  # mes referencia
                'HOMOLOGACIÓN': homolog_num,
                'CIUDAD': 0, 'COMPAÑÍA': 0,  # default
                'trimestre': 3, 'es_fin_año': 1, 'es_verano': 0
            }])
            future_row = future_row[features].fillna(0)
            xgb_pred = model.predict(future_row)[0]
            
            # 2. SARIMA específico por homologación
            sarima_pred = sarima_modelo_hibrido(df_filt, homolog_nombre, target, steps=5)
            
            # 3. ENSEMBLE: 60% XGB + 40% SARIMA
            for i, mes in enumerate(meses_futuros):
                pred_ensemble = 0.6 * xgb_pred + 0.4 * sarima_pred[i]
                
                resultados.append({
                    'HOMOLOGACIÓN_NOMBRE': homolog_nombre,
                    'Mes_Nombre': ['Agosto','Septiembre','Octubre','Noviembre','Diciembre'][i],
                    'XGBoost': xgb_pred.round(0),
                    'SARIMA': sarima_pred[i].round(0),
                    'ENSEMBLE': pred_ensemble.round(0)
                })
    
    return pd.DataFrame(resultados)

# === APP ===
st.title("🔥 ENSEMBLE XGBoost + SARIMA 2025")
st.markdown("**✅ 60% XGBoost + 40% SARIMA | PESO últimos años**")

df = cargar_datos()
if df.empty:
    st.stop()

df_clean = preparar_datos(df)

# FILTROS
st.sidebar.header("🔍 Filtros")
homologacion_opts = sorted(df_clean['HOMOLOGACIÓN'].dropna().unique())
homologacion = st.sidebar.multiselect("Homologación", homologacion_opts[:5])

df_filt = df_clean[df_clean['HOMOLOGACIÓN'].isin(homologacion)]

# MÉTRICAS GLOBALES
st.header("📊 Métricas Globales")
tabla_promedios = calcular_promedio_mensual(df_filt)

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Promedio Primas", f"${df_filt['Primas'].mean():,.0f}")
col2.metric("💰 Promedio Siniestros", f"${df_filt['Siniestros'].mean():,.0f}")
col3.metric("📈 Homologaciones", len(tabla_promedios['HOMOLOGACIÓN'].unique()))
col4.metric("🔥 Peso reciente", "x2 últimos 2 años")

# ENSEMBLE
st.header("🔮 ENSEMBLE Predicción Agosto-Diciembre 2025")
target = st.radio("Predecir", ["Primas", "Siniestros"], horizontal=True)

if st.button("🚀 Entrenar ENSEMBLE", type="primary", use_container_width=True):
    with st.spinner("Entrenando XGBoost + SARIMA..."):
        model, features, results, encoders, reverse_encoders = entrenar_xgboost(df_filt, target)
        if model:
            st.session_state.model = model
            st.session_state.features = features
            st.session_state.results = results
            st.session_state.encoders = encoders
            st.session_state.reverse_encoders = reverse_encoders
            st.session_state.target = target
            st.session_state.df_filt = df_filt
            st.session_state.tabla_promedios = tabla_promedios
            st.success("✅ ENSEMBLE listo!")
            st.rerun()

if 'model' in st.session_state:
    st.subheader("📈 Predicciones ENSEMBLE 2025")
    
    # 🎯 ENSEMBLE COMPLETO
    pred_df = ensemble_predicciones(
        st.session_state.model,
        st.session_state.features,
        st.session_state.tabla_promedios,
        st.session_state.encoders,
        st.session_state.reverse_encoders,
        st.session_state.df_filt,
        st.session_state.target
    )
    
    # Tabla principal: ENSEMBLE
    pivot_ensemble = pred_df.pivot(index='HOMOLOGACIÓN_NOMBRE', columns='Mes_Nombre', values='ENSEMBLE').round(0)
    st.dataframe(pivot_ensemble, use_container_width=True)
    
    # Tabla comparativa
    st.subheader("⚖️ Comparativa Modelos")
    pivot_comp = pred_df.pivot(index='HOMOLOGACIÓN_NOMBRE', columns='Mes_Nombre', values=['XGBoost','SARIMA','ENSEMBLE']).round(0)
    with st.expander("Ver comparativa completa"):
        st.dataframe(pivot_comp, use_container_width=True)
    
    # Métricas
    col1, col2 = st.columns(2)
    col1.metric("✅ MAE XGB", f"${st.session_state.results['mae']:,.0f}")
    col2.metric("✅ R² XGB", f"{st.session_state.results['r2']:.1%}")

# PROMEDIOS HISTÓRICOS
st.header("📊 Promedios Históricos")
pivot_hist = tabla_promedios.pivot(index='HOMOLOGACIÓN', columns='Mes_Nombre', values='Promedio_Total_Primas').fillna(0).round(0)
st.dataframe(pivot_hist)

# DESCARGA
if 'model' in st.session_state:
    csv = pivot_ensemble.to_csv().encode('utf-8')
    st.download_button("📥 Descargar ENSEMBLE", csv, f"ensemble_{pd.Timestamp.now().strftime('%Y%m%d')}.csv")
