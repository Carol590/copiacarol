# Predicción de Primas y Siniestros — app.py

Este repositorio contiene una aplicación Streamlit (app.py) diseñada para predecir las primas y los siniestros del mercado asegurador colombiano usando XGBoost (o un fallback si XGBoost no está disponible). La app crea pronósticos mensuales para los meses de agosto a diciembre de 2025 y presenta tres páginas con filtros por compañía, ciudad y ramo.

## Estructura de archivos
- `app.py` — aplicación Streamlit principal.
- `requirements.txt` — dependencias necesarias.
- `README.md` — (este archivo) instrucciones y notas.

## Formato esperado del dataset (CSV)
La app espera un CSV con las siguientes columnas (nombres exactos no son case-sensitive, pero se recomiendan tal como aparecen):

- HOMOLOGACIÓN — (ej. comuna / segmento) (string)
- Año — año (int) — opcional si FECHA existe
- COMPAÑÍA — compañía aseguradora (string)
- CIUDAD — ciudad (string)
- RAMOS — ramo / línea (string)
- Primas/Siniestros — categoría: "Primas" o "Siniestros" (string)
- FECHA — fecha con formato dd/mm/YYYY (ej. `31/08/2022 12:00:00 a. m.`) o cualquier parseable; se convertirá a comienzo de mes
- Valor_Mensual — valor numérico mensual (int/float)
- DEPARTAMENTO — departamento (string)

Ejemplo de filas:
GENERALES,2022,ALFA,BUENAVENTURA,VIDRIOS,Primas,31/08/2022 12:00:00 a. m.,0,VALLE DEL CAUCA

## Qué hace la app
- Limpia y normaliza fechas y valores.
- Permite filtrar por COMPAÑÍA, CIUDAD y RAMOS (Ramo).
- Entrena un modelo XGBoost por serie temporal (por HOMOLOGACIÓN y por variable Primas/Siniestros) cuando hay datos suficientes; si no, usa un promedio heurístico.
- Predice mensual (iterativo) de agosto a diciembre de 2025 (5 meses).
- Presenta:
  - Página 1: Gráfico histórico consolidado de Primas y Siniestros + tabla por HOMOLOGACIÓN con columnas de predicción.
  - Página 2: Mis ciudades objetivo (BOGOTA, MEDELLIN, CALI, BUCARAMANGA, BARRANQUILLA, CARTAGENA, TUNJA) — gráficos y tablas.
  - Página 3: Competidores (ESTADO, MAPFRE GENERALES, LIBERTY, AXA GENERALES, MUNDIAL, PREVISORA) — análisis similar.
- Permite descargar las predicciones en Excel.

## Cómo ejecutar
1. Crea un entorno virtual e instala dependencias:
   pip install -r requirements.txt
2. Ejecuta:
   streamlit run app.py
3. En la barra lateral sube tu CSV o utiliza la muestra (si proporcionada).

## Notas y limitaciones
- La calidad de las predicciones depende de la cantidad y calidad histórica por serie. Para series con menos de 12 meses de datos la app usa un promedio móvil como fallback.
- XGBoost se usa por defecto. Si XGBoost no está instalado, la app usa un regresor de scikit-learn (HistGradientBoostingRegressor).
- El pipeline es simple y pensado para prototipado rápido. Para producción se recomiendan pasos adicionales:
  - Feature engineering más exhaustivo (festivos, variables macro, IPC real).
  - Validación temporal robusta y calibración de intervalos de confianza.
  - Guardado y monitorización de modelos.

Si necesitas que adapte la app para calcular intervalos de confianza o exportar modelos serializados, dímelo y lo añadimos.
## Despliegue
[Insertar badge de Streamlit aquí]

## Instalación Local
```bash
pip install -r requirements.txt
streamlit run app.py
