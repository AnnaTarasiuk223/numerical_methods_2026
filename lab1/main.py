import requests
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. ОТРИМАННЯ ДАНИХ (21 вузол)
# ---------------------------------------------------------
coords = [
    [48.164214, 24.536044], [48.164983, 24.534836], [48.165605, 24.534068],
    [48.166228, 24.532915], [48.166777, 24.531927], [48.167326, 24.530884],
    [48.167011, 24.530061], [48.166053, 24.528039], [48.166655, 24.526064],
    [48.166497, 24.523574], [48.166128, 24.520214], [48.165416, 24.517170],
    [48.164546, 24.514640], [48.163412, 24.512980], [48.162331, 24.511715],
    [48.162015, 24.509462], [48.162147, 24.506932], [48.161751, 24.504244],
    [48.161197, 24.501793], [48.160580, 24.500537], [48.160250, 24.500106]
]


def get_elevations(locations):
    url = "https://api.open-elevation.com/api/v1/lookup"
    payload = {"locations": [{"latitude": l[0], "longitude": l[1]} for l in locations]}
    try:
        response = requests.post(url, json=payload, timeout=15)
        return [p['elevation'] for p in response.json()['results']]
    except:
        # Резервний розрахунок, якщо API лежить
        return [1263, 1285, 1284, 1334, 1310, 1320, 1318, 1338, 1375, 1418, 1487, 1524, 1553, 1630, 1757, 1792, 1828,
                1887, 1974, 1975, 2031]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl, dp = np.radians(lon2 - lon1), np.radians(lat2 - lat1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


full_elevations = get_elevations(coords)
full_distances = [0.0]
for i in range(1, len(coords)):
    d = haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
    full_distances.append(full_distances[-1] + d)

X_full = np.array(full_distances)
Y_full = np.array(full_elevations)


# ---------------------------------------------------------
# 2. ФУНКЦІЯ СПЛАЙН-ІНТЕРПОЛЯЦІЇ
# ---------------------------------------------------------
def solve_spline(x, y):
    n = len(x)
    h = np.diff(x)
    A = np.zeros((n, n))
    B = np.zeros(n)
    A[0, 0] = 1
    A[n - 1, n - 1] = 1
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        B[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    c = np.linalg.solve(A, B)
    a = y[:-1]
    b = (y[1:] - y[:-1]) / h - h * (c[1:] + 2 * c[:-1]) / 3
    d = (c[1:] - c[:-1]) / (3 * h)

    def interpolate(val):
        if val <= x[0]: return y[0]
        if val >= x[-1]: return y[-1]
        idx = np.searchsorted(x, val) - 1
        dx = val - x[idx]
        return a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3

    return np.vectorize(interpolate)


# ---------------------------------------------------------
# 3. ПОБУДОВА ГРАФІКІВ
# ---------------------------------------------------------
nodes_counts = [10, 15, 21]
plt.figure(figsize=(12, 6))
plt.scatter(X_full, Y_full, color='black', label='Original GPS', zorder=5)

x_smooth = np.linspace(X_full[0], X_full[-1], 300)
errors_list = []

for count in nodes_counts:
    # Вибираємо вузли рівномірно
    indices = np.linspace(0, len(X_full) - 1, count, dtype=int)
    x_nodes = X_full[indices]
    y_nodes = Y_full[indices]

    spline_func = solve_spline(x_nodes, y_nodes)
    y_smooth = spline_func(x_smooth)

    plt.plot(x_smooth, y_smooth, label=f'{count} вузлів')

    # Рахуємо похибку у всіх 21 оригінальних точках
    y_approx = spline_func(X_full)
    errors_list.append(np.abs(Y_full - y_approx))

plt.title("Профіль висоти: Заросляк - Говерла")
plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.legend()
plt.grid(True)
plt.show()

# Графіки похибок
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
for i, count in enumerate(nodes_counts):
    axs[i].plot(X_full, errors_list[i], 'r-o', markersize=4)
    axs[i].set_title(f"Похибка для {count} вузлів")
    axs[i].set_ylabel("Абс. похибка (м)")
    axs[i].grid(True)
    if i == 2: axs[i].set_xlabel("Відстань (м)")

plt.tight_layout()
plt.show()
