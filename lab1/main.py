import requests
import numpy as np
import matplotlib.pyplot as plt


locations = (
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
    "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
    "48.167011,24.530061|48.166053,24.528039|48.166555,24.526064|"
    "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
    "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
    "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
)
url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"

try:
    response = requests.get(url)
    data = response.json()
    results = data["results"]
except Exception as e:
    print(f"Помилка запиту: {e}. Перевірте інтернет або формат URL.")
    exit()

n = len(results)
print(f"Кількість вузлів: {n}")


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]
distances = [0.0]

for i in range(1, n):
    d = haversine(*coords[i-1], *coords[i])
    distances.append(distances[-1] + d)


def solve_spline(x, y):
    m = len(x)
    h = np.diff(x)


    A = np.zeros((m, m))
    B = np.zeros(m)
    A[0, 0] = 1
    A[m-1, m-1] = 1

    for i in range(1, m-1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        B[i] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])


    c = np.linalg.solve(A, B)


    a = y[:-1]
    d = np.diff(c) / (3 * h)
    b = (np.diff(y) / h) - (h / 3) * (c[1:] + 2 * c[:-1])

    return a, b, c, d, x

def get_spline_y(x_target, coeffs):
    a, b, c, d, x_orig = coeffs
    y_res = []
    for val in x_target:
        idx = np.searchsorted(x_orig, val) - 1
        idx = max(0, min(idx, len(a) - 1))
        dx = val - x_orig[idx]
        y_res.append(a[idx] + b[idx]*dx + c[idx]*dx**2 + d[idx]*dx**3)
    return np.array(y_res)


x_plot = np.linspace(distances[0], distances[-1], 500)
plt.figure(figsize=(10, 6))

for count in [10, 15, 20]:
    indices = np.linspace(0, n-1, count, dtype=int)
    x_sub = np.array(distances)[indices]
    y_sub = np.array(elevations)[indices]
    coeffs = solve_spline(x_sub, y_sub)
    y_vals = get_spline_y(x_plot, coeffs)
    plt.plot(x_plot, y_vals, label=f'{count} вузлів')

plt.scatter(distances, elevations, color='black', label='Original GPS')
plt.title("Профіль висоти: Заросляк - Говерла")
plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.legend()
plt.grid(True)
plt.show()


total_ascent = sum(max(elevations[i] - elevations[i-1], 0) for i in range(1, n))
total_descent = sum(max(elevations[i-1] - elevations[i], 0) for i in range(1, n))


full_coeffs = solve_spline(np.array(distances), np.array(elevations))
y_full = get_spline_y(x_plot, full_coeffs)
grad_full = np.gradient(y_full, x_plot) * 100

mass = 80
g = 9.81
energy_j = mass * g * total_ascent

print("\n--- ХАРАКТЕРИСТИКИ МАРШРУТУ ---")
print(f"Загальна довжина: {distances[-1]:.2f} м")
print(f"Сумарний набір висоти: {total_ascent:.2f} м")
print(f"Сумарний спуск: {total_descent:.2f} м")
print(f"Максимальний підйом: {np.max(grad_full):.2f} %")
print(f"Механічна робота: {energy_j / 1000:.2f} кДж")
print(f"Енерговитрати: {energy_j / 4184:.2f} ккал") 
