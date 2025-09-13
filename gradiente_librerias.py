import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


def gradiente_librerias(x, y, alpha, n_iterations):
    y = y.ravel()   # <- aplana a un vector de 1D (shape = (100,))

    # Se estandariza X para mejorar la convergencia
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    # Modelo SGD
    sgd_reg = SGDRegressor(
        max_iter=n_iterations,       # número de iteraciones
        tol=1e-3,            # criterio de convergencia
        eta0=alpha,            # tasa de aprendizaje inicial
        learning_rate="constant",  # tipo de schedule para α
        penalty=None,        # sin regularización (para imitar regresión lineal simple)
        random_state=42
    )

    # Entrenamiento
    sgd_reg.fit(X_scaled, y)

    # Resultados
    print("Parámetros finales (θ0, θ1):", sgd_reg.intercept_, sgd_reg.coef_)

    # 5. Predicciones
    X_new = np.array([[0], [2]])
    X_new_scaled = scaler.transform(X_new)
    y_pred = sgd_reg.predict(X_new_scaled)

    # 6. Visualización
    plt.scatter(x, y, alpha=0.7, label="Datos")
    plt.plot(X_new, y_pred, "r-", linewidth=2, label="Modelo con SGD")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    