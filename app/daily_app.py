import os
import pickle
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st

# Título de la aplicación
st.title("Pronóstico con Prophet 📈")
st.write("""
Sube un archivo histórico en formato `.csv` con las siguientes columnas:
- **Fechas**: Fechas en formato `dd/mm/aaaa`.
- **Llamadas**: Número de llamadas (valores numéricos).

Ejemplo:
""")

# Cargar archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

if uploaded_file is not None:
    try:
        # Leer el archivo CSV
        df = pd.read_csv(uploaded_file)
        
        # Verificar formato de las columnas
        if 'Fechas' not in df.columns or 'Llamadas' not in df.columns:
            st.error("El archivo debe contener las columnas 'Fechas' y 'Llamadas'.")
        else:
            # Convertir 'Fechas' a formato datetime
            df['Fechas'] = pd.to_datetime(df['Fechas'], format='%d/%m/%Y', errors='coerce')
            if df['Fechas'].isna().any():
                st.error("La columna 'Fechas' contiene datos inválidos. Por favor, corrige los errores.")
            else:
                df.rename(columns={'Fechas': 'ds', 'Llamadas': 'y'}, inplace=True)
                
                # Crear y ajustar el modelo Prophet
                model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                model.fit(df)
                
                # Predicción para 90 días futuros
                future = model.make_future_dataframe(periods=90, freq='D')
                forecast = model.predict(future)
                
                # Mostrar pronóstico
                st.subheader("Pronóstico")
                forecast_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'Fechas', 'yhat': 'Llamadas'})
                st.write(forecast_df)
                
                # Descargar predicciones
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="Descargar Pronóstico como CSV",
                    data=csv,
                    file_name='predicciones.csv',
                    mime='text/csv'
                )
                
                # Graficar resultados
                st.subheader("Gráfico del Pronóstico")
                fig1 = model.plot(forecast)
                plt.title("Pronóstico de llamadas")
                plt.xlabel("Fecha")
                plt.ylabel("Número de llamadas")
                st.pyplot(fig1)

                # Crear un gráfico combinado de histórico vs pronóstico
                st.subheader("Gráfico Histórico vs Pronóstico")
                plt.figure(figsize=(12, 6))
                plt.plot(df['ds'], df['y'], label='Datos Históricos', color='black')
                plt.plot(forecast['ds'], forecast['yhat'], label='Pronóstico', color='blue')
                plt.axvline(x=df['ds'].max(), color='red', linestyle='--', label='Inicio del Pronóstico')
                plt.title('Histórico vs Pronóstico de Llamadas', fontsize=16)
                plt.xlabel('Fecha', fontsize=12)
                plt.ylabel('Número de Llamadas', fontsize=12)
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)
                
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
