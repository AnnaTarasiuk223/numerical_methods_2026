import requests
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. ЗАПИТ ДО API ТА ТАБУЛЯЦІЯ (Крок 1-3) [cite: 280, 292]
# ---------------------------------------------------------
locations = (
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
    "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
    "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
    "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
    "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
    "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
)

url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"
data = requests.get(url).json()
results = data["results"]
n = len(results)

print(f"Кількість вузлів: {n}")  # [cite: 295]
print("\nТабуляція вузлів (Latitude | Longitude | Elevation):")
for i, p in enumerate(results):
    print(f"{i:2d} | {p['latitude']:.6f} | {p['longitude']:.6f} | {p['elevation']:.2f}")


# ---------------------------------------------------------
# 2. ВІДСТАНЬ ТА ВИСОТА (Крок 4-5) [cite: 303, 325]
# ---------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # [cite: 309]
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl, dp = np.radians(lon2 - lon1), np.radians(lat2 - lat1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))  # [cite: 317]


distances = [0.0]
elevations = [p['elevation'] for p in results]

for i in range(1, n):
    d = haversine(results[i - 1]['latitude'], results[i - 1]['longitude'],
                  results[i]['latitude'], results[i]['longitude'])
    distances.append(distances[-1] + d)  # [cite: 321]

print("\nТабуляція (Відстань | Висота):")  # [cite: 322]
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")


# ---------------------------------------------------------
# 3. ФУНКЦІЯ СПЛАЙНА ТА МЕТОД ПРОГОНКИ (Крок 6-9) [cite: 327, 329]
# ---------------------------------------------------------
def solve_spline(x_pts, y_pts):
    n_pts = len(x_pts)
    h_i = np.diff(x_pts)  # [cite: 204]

    # Побудова трьохдіагональної матриці (Пункт 6) [cite: 238]
    A = np.zeros((n_pts, n_pts))
    B = np.zeros(n_pts)
    A[0, 0] = 1
    A[n_pts - 1, n_pts - 1] = 1

    for i in range(1, n_pts - 1):
        A[i, i - 1] = h_i[i - 1]
        A[i, i] = 2 * (h_i[i - 1] + h_i[i])  # [cite: 237]
        A[i, i + 1] = h_i[i]
        B[i] = 3 * ((y_pts[i + 1] - y_pts[i]) / h_i[i] - (y_pts[i] - y_pts[i - 1]) / h_i[i - 1])

    # Розв'язок методом прогонки (Пункт 7) [cite: 239, 329]
    c_i = np.linalg.solve(A, B)

    # Коефіцієнти (Пункт 8-9) [cite: 331]
    a_i = y_pts[:-1]  # [cite: 224]
    b_i = (y_pts[1:] - y_pts[:-1]) / h_i - h_i * (c_i[1:] + 2 * c_i[:-1]) / 3  # [cite: 226]
    d_i = (c_i[1:] - c_i[:-1]) / (3 * h_i)  # [cite: 225]

    return a_i, b_i, c_i, d_i


# Обчислюємо коефіцієнти для всіх точок
a, b, c, d = solve_spline(np.array(distances), np.array(elevations))

print("\nКоефіцієнти сплайнів (a | b | c | d):")  # [cite: 330, 332]
for i in range(len(a)):
    print(f"{i:2d} | {a[i]:8.2f} | {b[i]:8.4f} | {c[i]:8.6f} | {d[i]:8.8f}")


# ---------------------------------------------------------
# 4. ВІЗУАЛІЗАЦІЯ ТА ПОХИБКА (Крок 10-12) [cite: 333, 335]
# ---------------------------------------------------------
def get_spline_y(x_target, x_orig, a_i, b_i, c_i, d_i):
    y_res = []
    for val in x_target:
        idx = np.searchsorted(x_orig, val) - 1
        idx = max(0, min(idx, len(a_i) - 1))
        dx = val - x_orig[idx]
        y_res.append(a_i[idx] + b_i[idx] * dx + c_i[idx] * dx ** 2 + d_i[idx] * dx ** 3)  # [cite: 199]
    return np.array(y_res)


x_plot = np.linspace(distances[0], distances[-1], 500)
y_plot = get_spline_y(x_plot, distances, a, b, c, d)

# Похибка (Пункт 12)
y_at_nodes = get_spline_y(distances, distances, a, b, c, d)
epsilon = np.abs(np.array(elevations) - y_at_nodes)
print(f"\nМаксимальна похибка на вузлах: {np.max(epsilon):.2e}")

# Графік порівняння вузлів (Пункт 10)
plt.figure(figsize=(10, 6))
for nodes_count in [10, 15, 20]:
    idx = np.linspace(0, n - 1, nodes_count, dtype=int)
    x_n, y_n = np.array(distances)[idx], np.array(elevations)[idx]
    a_n, b_n, c_n, d_n = solve_spline(x_n, y_n)
    y_n_plot = get_spline_y(x_plot, x_n, a_n, b_n, c_n, d_n)
    plt.plot(x_plot, y_n_plot, label=f'{nodes_count} вузлів')

plt.scatter(distances, elevations, color='black', label='Оригінальні GPS точки')
plt.title("Порівняння точності при різній кількості вузлів ")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------------
# 5. ДОДАТКОВО: ХАРАКТЕРИСТИКИ ТА ГРАДІЄНТ [cite: 338, 346]
# ---------------------------------------------------------
total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n))  # [cite: 343]
total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n))  # [cite: 345]
energy_j = 80 * 9.81 * total_ascent  # [cite: 362]

# Аналіз градієнта (Пункт 2 Додатково)
grad_full = np.gradient(y_plot, x_plot) * 100  # [cite: 353]

print(f"\n--- Характеристики маршруту ---")
print(f"Загальна довжина: {distances[-1]:.2f} м")  # [cite: 341]
print(f"Сумарний набір висоти: {total_ascent:.2f} м")  # [cite: 343]
print(f"Механічна робота: {energy_j / 1000:.2f} кДж")  # [cite: 366]
print(f"Енергія: {energy_j / 4184:.2f} ккал")  # [cite: 368]
print(f"Максимальний підйом: {np.max(grad_full):.1f}%")  # [cite: 355]
print(f"Середній градієнт: {np.mean(np.abs(grad_full)):.1f}%")  # [cite: 357]
