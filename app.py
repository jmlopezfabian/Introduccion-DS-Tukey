import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Dashboard estudiantes")

st.text("Análisis exploratorio de datos de estudiantes")

df = pd.read_csv("/home/gsu/nuevoProyecto/data/csvs/examenes.csv")

st.write(df.head())

st.subheader("Estadisticas descriptivas del conjunto de datos")

st.write(df.describe())

st.subheader("Analisis monovariado de variables categoricas")

variables_categoricas_disponibles = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

opcion = st.selectbox("Elige una opción: ", variables_categoricas_disponibles)

frecuencias = df[opcion].value_counts()

st.bar_chart(frecuencias, color='#28327A')

# analisis monovariado de variables numericos

#selector para elegir la variable a visualizar 

#Selector para elegir el grafico (hist)

grafico = st.selectbox("Elige el grafico: ", ["hist", "boxplot"])

if grafico == "hist":
    # Logica para hacer un histograma
else:
    # Logica para hacer un boxplot