import requests
import numpy as np
import matplotlib.pyplot as plt


locations = ( #координати широта довгота
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
    "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
    "48.167011,24.530061|48.166053,24.528039|48.166555,24.526064|"
    "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
    "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
    "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
) #url для api
url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"

try:
    response = requests.get(url)  # Відправка запиту на сервер
    data = response.json()        # Перетворення текстової відповіді JSON у словник Python
    results = data["results"]     # Витягування списку словників з результатами (lat, lon, elevation)
except Exception as e:            # Обробка помилок (якщо немає інтернету або API лежить)
    print(f"Помилка запиту: {e}. Перевірте інтернет або формат URL.")
    exit()

n = len(results)                  # Підрахунок кількості отриманих точок
print(f"Кількість вузлів: {n}")


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Радіус Землі в метрах
    # Перетворення градусів у радіани
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    # Формула гаверсину для обчислення відстані на кулі
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# Витягуємо координати та висоти в окремі списки
coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]
distances = [0.0]  # Початкова точка — 0 метрів

# Наповнюємо масив distances накопичувальною сумою відстаней між точками
for i in range(1, n):
    d = haversine(*coords[i-1], *coords[i])
    distances.append(distances[-1] + d)


def solve_spline(x, y):
    m = len(x)
    h = np.diff(x)  # Крок між сусідніми вузлами (x_i+1 - x_i)

    A = np.zeros((m, m))  # Створення порожньої матриці СЛАУ
    B = np.zeros(m)  # Створення вектора правої частини

    # Крайові умови (природний сплайн: друга похідна на кінцях дорівнює 0)
    A[0, 0] = 1
    A[m - 1, m - 1] = 1

    # Заповнення тридіагональної матриці згідно з математичними умовами гладкості
    for i in range(1, m - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        B[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    c = np.linalg.solve(A, B)  # Знаходження коефіцієнтів "c" через розв'язання системи

    # Розрахунок інших коефіцієнтів сплайна a, b, d на основі c та y
    a = y[:-1]
    d = np.diff(c) / (3 * h)
    b = (np.diff(y) / h) - (h / 3) * (c[1:] + 2 * c[:-1])

    return a, b, c, d, x


def get_spline_y(x_target, coeffs):
    a, b, c, d, x_orig = coeffs
    y_res = []
    for val in x_target:
        # Пошук сегмента, в який потрапляє точка
        idx = np.searchsorted(x_orig, val) - 1
        idx = max(0, min(idx, len(a) - 1))
        dx = val - x_orig[idx]
        # Обчислення значення кубічного полінома: y = a + b*dx + c*dx^2 + d*dx^3
        y_res.append(a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3)
    return np.array(y_res)


# Створення 500 точок для плавного малювання графіка
x_plot = np.linspace(distances[0], distances[-1], 500)
plt.figure(figsize=(10, 6))

# Цикл для порівняння: що буде, якщо взяти лише 10, 15 або всі 20 точок
for count in [10, 15, 20]:
    indices = np.linspace(0, n-1, count, dtype=int)
    x_sub = np.array(distances)[indices]
    y_sub = np.array(elevations)[indices]
    coeffs = solve_spline(x_sub, y_sub)
    y_vals = get_spline_y(x_plot, coeffs)
    plt.plot(x_plot, y_vals, label=f'{count} вузлів')

# Додавання "сирих" даних у вигляді точок
plt.scatter(distances, elevations, color='black', label='Original GPS')
plt.title("Профіль висоти: Заросляк - Говерла")
plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.legend()
plt.grid(True)
plt.show()


# Рахуємо суму всіх підйомів (тільки коли різниця висот > 0)
total_ascent = sum(max(elevations[i] - elevations[i-1], 0) for i in range(1, n))
# Рахуємо суму всіх спусків
total_descent = sum(max(elevations[i-1] - elevations[i], 0) for i in range(1, n))

# Розрахунок градієнта (крутизни) через похідну сплайна
full_coeffs = solve_spline(np.array(distances), np.array(elevations))
y_full = get_spline_y(x_plot, full_coeffs)
grad_full = np.gradient(y_full, x_plot) * 100  # Перетворення у відсотки

# 1. Обчислюємо наближені значення саме у тих точках, де ми маємо оригінальні дані
y_approx = get_spline_y(distances, full_coeffs)

# 2. Обчислюємо похибку за формулою зі скріншота: epsilon = |y_orig - y_approx|
error = np.abs(np.array(elevations) - y_approx)

# 3. Будуємо окремий графік похибки
plt.figure(figsize=(10, 4))
plt.plot(distances, error, 'r-o', label='Похибка ε = |y - y_набл|')
plt.title("Графік похибки інтерполяції")
plt.xlabel("Відстань (м)")
plt.ylabel("Абсолютна похибка (м)")
plt.grid(True)
plt.legend()
plt.show()

# Фізичні константи
mass = 80    # Вага туриста
g = 9.81     # Прискорення вільного падіння
# Формула потенціальної енергії E = m*g*h
energy_j = mass * g * total_ascent

# Вивід результатів
print(f"Загальна довжина: {distances[-1]:.2f} м")
print(f"Сумарний набір висоти: {total_ascent:.2f} м")
print(f"Максимальний підйом: {np.max(grad_full):.2f} %")
print(f"Енерговитрати: {energy_j / 4184:.2f} ккал")
