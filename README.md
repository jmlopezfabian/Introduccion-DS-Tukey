# Clases DS Tukey

Notas, código y ejercicios del curso de Ciencia de Datos. Cada módulo corresponde a un tema visto en clase e incluye el notebook de trabajo correspondiente.

## Módulos

### Bases de Python

| # | Módulo | Página |
|---|--------|--------|
| B1 | Línea de Comandos | [terminal.html](docs/terminal.html) |
| B2 | Variables y Tipos de Datos | [variables.html](docs/variables.html) |
| B3 | Entrada y Salida de Datos | [entrada-salida.html](docs/entrada-salida.html) |
| B4 | Operadores Aritméticos | [operadores.html](docs/operadores.html) |
| B5 | Git y GitHub | [git-basico.html](docs/git-basico.html) |
| B6 | Condicionales | [condicionales.html](docs/condicionales.html) |
| B7 | Listas y Bucle for | [listas-cadenas.html](docs/listas-cadenas.html) |

### Ciencia de Datos

| # | Módulo | Página | Notebook |
|---|--------|--------|----------|
| 01 | NumPy | [numpy.html](docs/numpy.html) | [numpy.ipynb](notebooks/numpy.ipynb) |
| 02 | Visualización de Datos | [visualizacion.html](docs/visualizacion.html) | [matplotlib.ipynb](notebooks/matplotlib.ipynb) |
| 03 | Análisis Exploratorio (EDA) | [eda.html](docs/eda.html) | [notebook.ipynb](notebooks/notebook.ipynb) |
| 04 | Regresión Lineal Simple | [regresion-simple.html](docs/regresion-simple.html) | [regresion.ipynb](notebooks/regresion.ipynb) |
| 05 | Gradiente Descendente | [gradiente.html](docs/gradiente.html) | [gradiente.ipynb](notebooks/gradiente.ipynb) |
| 06 | Regresión con Statsmodels | [regresion-statsmodels.html](docs/regresion-statsmodels.html) | [regresion_lib.ipynb](notebooks/regresion_lib.ipynb) |
| 07 | Regresión Avanzada | [regresion-avanzada.html](docs/regresion-avanzada.html) | [regresion_final.ipynb](notebooks/regresion_final.ipynb) |
| 08 | Pruebas de Hipótesis | [pruebas-hipotesis.html](docs/pruebas-hipotesis.html) | [pruebas_hipotesis.ipynb](notebooks/pruebas_hipotesis.ipynb) |
| — | Ejercicios Prácticos | [ejercicios.html](docs/ejercicios.html) | [ejercicios_practicos.ipynb](notebooks/ejercicios/ejercicios_practicos.ipynb) |

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
