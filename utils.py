"""Funciones de utilidad para calculo de regresion lineal simple y gradiente"""

import numpy as np

def calcular_y(x:np.ndarray, m:float, b:float) -> np.ndarray:
    """Esta funcion estima las y utilizando una regresion lineal simple."""
    y = x * m + b
    return y

def ecm(y_real: np.ndarray, y_est: np.ndarray) -> float:
    return ((y_real - y_est) ** 2).mean()

def gradiente_descendente_m(m_act: float, alpha: float, b: float , x: np.ndarray, y_real: np.ndarray):
    return m_act - alpha * (2 * ( x * (m_act * x + b - y_real)).mean())