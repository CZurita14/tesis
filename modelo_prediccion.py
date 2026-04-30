import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def cargar_y_limpiar_datos(filepath):
    print("Iniciando fase ETL (Extracción, Transformación y Carga)...")
    # Cargar datos desde la hoja correspondiente
    df = pd.read_excel(filepath, sheet_name='Hoja2')
    
    # Seleccionar las columnas relevantes (fecha y peso/valor registrado)
    df = df[['created_at', 'value']].copy()
    
    # Limpieza de valores nulos o inválidos
    df.dropna(inplace=True)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df.dropna(inplace=True)
    
    # Filtrar valores negativos o anómalos
    df = df[df['value'] > 0]
    
    # Convertir a formato fecha y tiempo
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Establecer el índice temporal
    df.set_index('created_at', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Datos limpios: {len(df)} registros disponibles.")
    return df

def realizar_eda(df):
    print("Iniciando Análisis Exploratorio de Datos (EDA)...")
    # Configuración de estilo de Seaborn
    sns.set_theme(style="whitegrid")
    
    # 1. Gráfico de Serie de Tiempo
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['value'], color='teal', alpha=0.7)
    plt.title('Serie de Tiempo - Peso Registrado por Sensores (g)', fontsize=14)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Peso Registrado (g)', fontsize=12)
    plt.tight_layout()
    plt.savefig('eda_serie_tiempo.png')
    plt.close()
    
    # 2. Distribución de los Datos
    plt.figure(figsize=(10, 6))
    sns.histplot(df['value'], bins=50, kde=True, color='indigo')
    plt.title('Distribución de los Valores de Peso', fontsize=14)
    plt.xlabel('Peso (g)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.tight_layout()
    plt.savefig('eda_distribucion.png')
    plt.close()
    
    print("Gráficos de EDA generados y guardados: 'eda_serie_tiempo.png', 'eda_distribucion.png'.")

def integrar_logica_negocio(df):
    print("Integrando variables de negocio y preparación de Features para el modelo...")
    
    # ---- VARIABLES DE NEGOCIO PROPORCIONADAS ----
    PESO_PANTALON_SECO_G = 500  # g
    TELA_ADQUIRIDA_M = 9805.66  # metros
    
    # Desperdicio histórico: Octubre 2025 - Marzo 2026 (6 meses)
    DESPERDICIO_TOTAL_KG = 16075  # kg
    PROMEDIO_DESPERDICIO_MENSUAL_KG = DESPERDICIO_TOTAL_KG / 6
    
    # Desperdicio de papel mensual y tela por pantalón
    RETAZO_PAPEL_PROM_M = 55    # Promedio entre 50 y 60 metros
    TELA_POR_PANTALON_M = 1.20  # Promedio entre 1.10 y 1.30 metros
    # ---------------------------------------------
    
    # Agrupar los datos por día (agregación diaria para predecir a futuro)
    df_diario = df.resample('D').agg({'value': 'sum'}).dropna()
    df_diario.rename(columns={'value': 'peso_total_g'}, inplace=True)
    
    # Cálculos derivados en base a la lógica de negocio
    # Cantidad estimada de pantalones procesados según el peso en sensores
    df_diario['pantalones_procesados'] = df_diario['peso_total_g'] / PESO_PANTALON_SECO_G
    
    # Estimación de la tela consumida para los pantalones procesados
    df_diario['tela_consumida_m'] = df_diario['pantalones_procesados'] * TELA_POR_PANTALON_M
    
    # Ratio de desperdicio estimado (basado en el promedio mensual histórico)
    # 1 mes promedio = 30 días aprox -> Promedio diario estimado
    promedio_diario_desperdicio_g = (PROMEDIO_DESPERDICIO_MENSUAL_KG * 1000) / 30
    df_diario['desperdicio_estimado_g'] = promedio_diario_desperdicio_g
    
    # Variables temporales para el modelo de Machine Learning
    df_diario['dia_semana'] = df_diario.index.dayofweek
    df_diario['dia_mes'] = df_diario.index.day
    df_diario['mes'] = df_diario.index.month
    
    # Creación de variables rezagadas (Lags) para predecir series de tiempo con Random Forest
    df_diario['peso_lag_1'] = df_diario['peso_total_g'].shift(1)
    df_diario['peso_lag_2'] = df_diario['peso_total_g'].shift(2)
    df_diario['peso_lag_3'] = df_diario['peso_total_g'].shift(3)
    
    # Media móvil de los últimos 3 días
    df_diario['media_movil_3d'] = df_diario['peso_total_g'].rolling(window=3).mean()
    
    # Eliminar valores NaN generados por el lag y la media móvil
    df_diario.dropna(inplace=True)
    
    return df_diario

def entrenar_modelo_random_forest(df_procesado):
    print("Entrenando el modelo Predictivo Random Forest...")
    
    # Variables predictoras (X) y variable objetivo (y)
    # Predeciremos el 'peso_total_g' del día actual basado SOLO en el pasado y tiempos
    features = ['dia_semana', 'dia_mes', 'mes', 'peso_lag_1', 'peso_lag_2', 'peso_lag_3', 'media_movil_3d']
    X = df_procesado[features]
    y = df_procesado['peso_total_g']
    
    # División de datos: 80% Entrenamiento, 20% Prueba (sin mezclar para mantener orden temporal)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Configuración y entrenamiento del Random Forest
    rf_model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predicción
    y_pred = rf_model.predict(X_test)
    
    # Evaluación del Modelo
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- RESULTADOS DEL MODELO RANDOM FOREST ---")
    print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f} g")
    print(f"MAE (Error Absoluto Medio): {mae:.2f} g")
    print(f"R² Score (Precisión): {r2:.4f}")
    print("-------------------------------------------\n")
    
    # Visualización de la Predicción vs Realidad
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Datos Reales', marker='o')
    plt.plot(y_test.index, y_pred, label='Predicción Random Forest', marker='x', linestyle='--')
    plt.title('Comparación: Predicción vs Valores Reales (Random Forest)', fontsize=14)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Peso Total (g)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediccion_random_forest.png')
    plt.close()
    
    print("Gráfico de predicción guardado: 'prediccion_random_forest.png'.")
    
    # Importancia de las variables
    importancias = rf_model.feature_importances_
    df_importancia = pd.DataFrame({'Variable': features, 'Importancia': importancias})
    df_importancia = df_importancia.sort_values(by='Importancia', ascending=False)
    print("\nImportancia de las variables:")
    print(df_importancia.to_string(index=False))
    
    return rf_model

if __name__ == "__main__":
    archivo_excel = 'Datos-sensores-entrenamiento.xlsx'
    
    print("=== PIPELINE DE PREDICCIÓN Y ANÁLISIS DE DATOS ===")
    try:
        # 1. Ejecutar ETL
        df_limpio = cargar_y_limpiar_datos(archivo_excel)
        
        # 2. Ejecutar EDA
        realizar_eda(df_limpio)
        
        # 3. Aplicar parámetros y lógica de la tesis
        df_final = integrar_logica_negocio(df_limpio)
        
        # 4. Entrenamiento del Modelo
        if len(df_final) > 10:  # Validar que hay suficientes datos después de agrupar por días
            modelo = entrenar_modelo_random_forest(df_final)
            print("\n¡Ejecución del pipeline completada con éxito!")
        else:
            print("\nAlerta: No hay suficientes datos agrupados diariamente para entrenar el modelo.")
            
    except Exception as e:
        print(f"\nOcurrió un error en la ejecución: {str(e)}")
