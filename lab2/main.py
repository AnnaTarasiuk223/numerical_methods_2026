import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- ПУНКТ 1: Вхідні дані (Варіант 5) ---
x_nodes = np.array([100, 200, 400, 800, 1600], dtype=float)
y_nodes = np.array([120, 110, 90, 65, 40], dtype=float)

# --- ПУНКТ 2: Специфічні функції за методичкою ---

def omega_k(x, nodes, k):
    """Знаходження значення omega_k(x) = П(x - x_i)"""
    res = 1.0
    for i in range(k):
        res *= (x - nodes[i])
    return res

def divided_differences_table(x, y):
    """Обчислення розділених різниць f(x0,...,xk)"""
    n = len(y)
    table = np.zeros([n, n])
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])
    return table

def Newton_Nn(x, nodes, diff_table):
    """Знаходження значення багаточлена Ньютона N_n(x)"""
    n = diff_table.shape[1]
    res = diff_table[0, 0]
    for k in range(1, n):
        res += diff_table[0, k] * omega_k(x, nodes, k)
    return res

def Lagrange_Ln(x, nodes, y_values):
    """Метод Лагранжа для порівняння"""
    total = 0
    n = len(nodes)
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if i != j:
                term *= (x - nodes[j]) / (nodes[i] - nodes[j])
        total += term
    return total

def true_f(x):
    """Гіпотетична модель рушія для розрахунку похибок"""
    return 130 * np.exp(-0.00075 * x)

# --- ПУНКТ 3-4: Основні розрахунки ---

diff_table_base = divided_differences_table(x_nodes, y_nodes)

# Прогноз для 1000 об'єктів
fps_1000 = Newton_Nn(1000, x_nodes, diff_table_base)

# Знаходження x для FPS = 60
x_range = np.linspace(100, 1600, 5000)
y_interp_base = [Newton_Nn(val, x_nodes, diff_table_base) for val in x_range]
limit_60fps = x_range[np.argmin(np.abs(np.array(y_interp_base) - 60))]

print("ОСНОВНІ РЕЗУЛЬТАТИ:")
print(f"Прогноз FPS для 1000 об'єктів: {fps_1000:.2f}")
print(f"Межа для 60 FPS: ~{int(limit_60fps)} об'єктів")

# --- ПУНКТ 5: Графік похибок для n=5, 10, 15, 20 ---

plt.figure(figsize=(10, 6))
for n_count in [5, 10, 15, 20]:
    x_test_nodes = np.linspace(100, 1600, n_count)
    y_test_nodes = true_f(x_test_nodes)
    table_test = divided_differences_table(x_test_nodes, y_test_nodes)

    x_plot = np.linspace(100, 1600, 500)
    errors = [abs(true_f(val) - Newton_Nn(val, x_test_nodes, table_test)) for val in x_plot]
    plt.plot(x_plot, errors, label=f'n={n_count}')

plt.yscale('log')
plt.ylim(1e-15, 1e2)
plt.title("Графік похибок інтерполяції (n=5, 10, 15, 20)")
plt.xlabel("Кількість об'єктів")
plt.ylabel("Похибка (log scale)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()

# ==============================================================
# КОНСОЛЬНЕ ДОСЛІДЖЕННЯ ВПЛИВУ КРОКУ ТА КІЛЬКОСТІ ВУЗЛІВ
# ==============================================================

print("\n" + "="*75)
print("1. ДОСЛІДЖЕННЯ ВПЛИВУ КІЛЬКОСТІ ВУЗЛІВ (Фіксований інтервал [100, 1600])")
print("-" * 75)
print(f"{'n (вузлів)':<12} | {'Макс. похибка (Newton)':<25} | {'Статус'}")
print("-" * 75)

x_dense_test = np.linspace(100, 1600, 1000)
for n in [5, 10, 15, 20]:
    x_test = np.linspace(100, 1600, n)
    y_test = true_f(x_test)
    table = divided_differences_table(x_test, y_test)
    errors = [abs(true_f(x) - Newton_Nn(x, x_test, table)) for x in x_dense_test]
    max_err = max(errors)
    status = "Відмінно" if max_err < 1e-10 else "Збігається"
    print(f"{n:<12} | {max_err:<25.2e} | {status}")

print("\n" + "="*75)
print("2. ДОСЛІДЖЕННЯ ВПЛИВУ КРОКУ (Зі збільшенням кількості точок)")
print("-" * 75)
print(f"{'Крок (h)':<12} | {'n (точок)':<12} | {'Макс. похибка':<20}")
print("-" * 75)

for step in [375, 150, 100, 75]:
    x_step = np.arange(100, 1600 + step, step)
    n_pts = len(x_step)
    y_step = true_f(x_step)
    table_step = divided_differences_table(x_step, y_step)
    errors_step = [abs(true_f(x) - Newton_Nn(x, x_step, table_step)) for x in x_dense_test]
    print(f"{step:<12} | {n_pts:<12} | {max(errors_step):.2e}")

# ==============================================================
# ДОДАТКОВІ ГРАФІКИ: ЕФЕКТ РУНГЕ ТА ПОРІВНЯННЯ МЕТОДІВ
# ==============================================================

# Аналіз ефекту Рунге
def runge_function(x): return 1 / (1 + 0.0005 * x**2)

plt.figure(figsize=(10, 6))
x_runge_dense = np.linspace(-200, 200, 1000)
for n_nodes in [5, 10, 15, 20]:
    x_r = np.linspace(-200, 200, n_nodes)
    y_r = runge_function(x_r)
    t_r = divided_differences_table(x_r, y_r)
    y_i = [Newton_Nn(val, x_r, t_r) for val in x_runge_dense]
    plt.plot(x_runge_dense, y_i, label=f"n={n_nodes}")

plt.plot(x_runge_dense, runge_function(x_runge_dense), 'k--', label="Функція Рунге")
plt.ylim(-0.5, 1.5)
plt.title("Ефект Рунге: коливання на краях при n=20")
plt.legend()
plt.grid(True)
plt.show()

