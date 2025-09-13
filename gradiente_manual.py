import numpy as np
import matplotlib.pyplot as plt


def gradiente_manual(x, y, alpha, n_iterations):
    n = len(x)        # número de ejemplos
    X_b = np.c_[np.ones((100, 1)), x] # Agregamos una columna de 1s a X para representar el sesgo (θ0). Así cada fila queda como: [1, x]


    theta = np.random.randn(2, 1) # valores iniciales aleatorios cercanos a 0
    for iteration in range(n_iterations):
        # Calculamos el gradiente de la función de costo (ECM)
        gradients = (2/n) * X_b.T.dot(X_b.dot(theta) - y)
        # Actualizamos los parámetros en la dirección opuesta al gradiente
        theta = theta - alpha * gradients


    # Resultados
    print("Parámetros finales (θ0, θ1):", theta.ravel())

    # ------------------------------
    # 7. Predicciones
    # ------------------------------
    X_new = np.array([[0], [2]])      # dos valores de X para probar
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    y_predict = X_new_b.dot(theta)    # y = θ0 + θ1 * x

    # ------------------------------
    # 8. Visualizamos el ajuste
    # ------------------------------
    plt.scatter(x, y, alpha=0.7, label="Datos")  # puntos reales
    plt.plot(X_new, y_predict, "r-", linewidth=2, label="Modelo") # recta ajustada
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()