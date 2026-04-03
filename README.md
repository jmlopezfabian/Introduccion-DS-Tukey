# Clases DS Tukey

Notas, código y ejercicios del curso de Ciencia de Datos. Cada módulo corresponde a un tema visto en clase e incluye el notebook de trabajo correspondiente.

## Módulos

### Bases de Python

| # | Módulo | Página |
|---|--------|--------|
| B1 | Línea de Comandos | [terminal.html](terminal.html) |
| B2 | Variables y Tipos de Datos | [variables.html](variables.html) |
| B3 | Entrada y Salida de Datos | [entrada-salida.html](entrada-salida.html) |
| B4 | Operadores Aritméticos | [operadores.html](operadores.html) |
| B5 | Git y GitHub | [git-basico.html](git-basico.html) |
| B6 | Condicionales | [condicionales.html](condicionales.html) |
| B7 | Listas y Bucle for | [listas-cadenas.html](listas-cadenas.html) |

### Ciencia de Datos

| # | Módulo | Página | Notebook |
|---|--------|--------|----------|
| 01 | NumPy | [numpy.html](numpy.html) | [numpy.ipynb](notebooks/numpy.ipynb) |
| 02 | Visualización de Datos | [visualizacion.html](visualizacion.html) | [matplotlib.ipynb](notebooks/matplotlib.ipynb) |
| 03 | Análisis Exploratorio (EDA) | [eda.html](eda.html) | [notebook.ipynb](notebooks/notebook.ipynb) |
| 04 | Regresión Lineal Simple | [regresion-simple.html](regresion-simple.html) | [regresion.ipynb](notebooks/regresion.ipynb) |
| 05 | Gradiente Descendente | [gradiente.html](gradiente.html) | [gradiente.ipynb](notebooks/gradiente.ipynb) |
| 06 | Regresión con Statsmodels | [regresion-statsmodels.html](regresion-statsmodels.html) | [regresion_lib.ipynb](notebooks/regresion_lib.ipynb) |
| 07 | Regresión Avanzada | [regresion-avanzada.html](regresion-avanzada.html) | [regresion_final.ipynb](notebooks/regresion_final.ipynb) |
| 08 | Pruebas de Hipótesis | [pruebas-hipotesis.html](pruebas-hipotesis.html) | [pruebas_hipotesis.ipynb](notebooks/pruebas_hipotesis.ipynb) |
| — | Ejercicios Prácticos | [ejercicios.html](ejercicios.html) | [ejercicios_practicos.ipynb](notebooks/ejercicios/ejercicios_practicos.ipynb) |

## Estructura

```
Clasespavos2026/
├── index.html                        # Página principal
├── terminal.html
├── variables.html
├── entrada-salida.html
├── operadores.html
├── git-basico.html
├── condicionales.html
├── listas-cadenas.html
├── numpy.html
├── visualizacion.html
├── eda.html
├── regresion-simple.html
├── gradiente.html
├── regresion-statsmodels.html
├── regresion-avanzada.html
├── pruebas-hipotesis.html
├── ejercicios.html
│
├── notebooks/
│   ├── numpy.ipynb
│   ├── matplotlib.ipynb
│   ├── notebook.ipynb
│   ├── regresion.ipynb
│   ├── gradiente.ipynb
│   ├── regresion_lib.ipynb
│   ├── regresion_final.ipynb
│   ├── pruebas_hipotesis.ipynb
│   └── ejercicios/
│       └── ejercicios_practicos.ipynb
│
├── data/
│   └── csvs/
│       └── examenes.csv
│
├── app.py                            # Dashboard Streamlit
├── utils.py                          # Funciones auxiliares
└── requirements.txt
```

## Instalación

```bash
git clone https://github.com/jmlopezfabian/Clasespavos2026.git
cd Clasespavos2026

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Uso

### Notebooks

```bash
jupyter notebook
```

Abre el notebook del módulo que quieras desde la carpeta `notebooks/`.

### Dashboard Streamlit

```bash
streamlit run app.py
```

## Dependencias

- Python 3.9+
- pandas, numpy, matplotlib, seaborn
- statsmodels, scikit-learn, scipy
- streamlit
- jupyter
