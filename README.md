```markdown
# Predicción de Primas y Siniestros — Streamlit + XGBoost

Esta aplicación (app.py) carga los datos desde la Google Sheet que indiques y genera predicciones de primas y siniestros para Agosto-Diciembre 2025 (5 meses) usando XGBoost (o un regresor de fallback si XGBoost no está disponible).

Contenido
- `app.py` — aplicación Streamlit principal (carga desde Google Sheet pública / service account / CSV).
- `requirements.txt` — dependencias necesarias.

Uso rápido
1. Crear entorno virtual (opcional)
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows

2. Instalar dependencias
   pip install -r requirements.txt

3. Ejecutar Streamlit
   streamlit run app.py

Cómo conectar la Google Sheet
- Opción A (pública): en la sidebar elige "Google Sheet pública" y pega la URL completa o el Sheet ID. La app intentará cargar automáticamente (export CSV / gviz / pub CSV).
- Opción B (service account, para hojas privadas): en la sidebar elige "Google Sheet (service account)". Guarda el JSON de la service account en `st.secrets["gcp_service_account"]` (recomendado) o súbelo en la interfaz. Además, comparte la hoja con el e-mail del service account con permiso de lectura.
  - Ejemplo en `secrets.toml` para Streamlit Cloud:
    ```
    [gcp_service_account]
    type = "service_account"
    project_id = "your-project-id"
    private_key_id = "..."
    private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
    client_email = "xxxx@xxxx.iam.gserviceaccount.com"
    client_id = "..."
    ...
    ```
  - Alternativamente puedes definir la variable de entorno `GCP_SERVICE_ACCOUNT_JSON` con el JSON serializado.

Formato esperado del dataset (si subes CSV)
- Columnas manejadas (la app intenta normalizar variantes):
  - HOMOLOGACIÓN
  - Año (opcional)
  - COMPAÑÍA
  - CIUDAD
  - RAMOS
  - Primas/Siniestros (valores: "Primas" o "Siniestros")
  - FECHA (ej. `31/08/2022 12:00:00 a. m.` — la app normaliza al primer día del mes)
  - Valor_Mensual
  - DEPARTAMENTO

Qué hace la app
- Limpia y normaliza los datos.
- Entrena un modelo por serie (HOMOLOGACIÓN × Primas/Siniestros) usando XGBoost (o fallback).
- Predice los meses Agosto a Diciembre de 2025.
- Presenta 3 páginas:
  1. Resumen por HOMOLOGACIÓN (tabla con predicciones).
  2. Ciudades objetivo (BOGOTA, MEDELLIN, CALI, BUCARAMANGA, BARRANQUILLA, CARTAGENA, TUNJA).
  3. Competidores (ESTADO, MAPFRE GENERALES, LIBERTY, AXA GENERALES, MUNDIAL, PREVISORA).
- Permite descargar las predicciones en Excel.

Notas y recomendaciones
- Si la Google Sheet no carga con la opción pública, prueba:
  - Abrir en el navegador: https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>
  - Hacer "Archivo → Publicar en la web" y usar la URL de `pub?output=csv`
  - Usar la opción service account si prefieres mantener la hoja privada.
- Para mejorar la calidad de predicción se recomienda:
  - Más datos históricos por serie (ideal >= 24 meses)
  - Ingenierías de features adicionales (festivos, variables macro)
  - Validación temporal y métricas (SMAPE, MAPE, etc.)

Si quieres que:
- añada intervalos de confianza,
- permita guardar modelos entrenados,
- o empaquete la app para despliegue en Streamlit Cloud con secrets configurados,
dímelo y lo preparo.
```

