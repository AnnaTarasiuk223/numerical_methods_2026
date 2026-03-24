import numpy as np
import matplotlib.pyplot as plt

# Визначення функції та її аналітичної похідної
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM_dt_exact(t):
    # Явний вираз першої похідної: M'(t) = -5*e^(-0.1t) + 5*cos(t)
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

def central_diff(f, t, h):
    return (f(t + h) - f(t - h)) / (2 * h)

# Параметри
t0 = 1.0
t_axis = np.linspace(0, 20, 500)

# Дослідження залежності похибки від кроку h
h_values = np.logspace(-20, 3, 100)
errors = []
exact_val = dM_dt_exact(t0)

for hi in h_values:
    y_h_i = central_diff(M, t0, hi)
    errors.append(abs(y_h_i - exact_val))

h_opt = h_values[np.argmin(errors)] # Пошук оптимального h0
R0 = min(errors)
print(f"Оптимальний крок h0 = {h_opt:.1e}, мінімальна похибка R0 = {R0:.2e}")

# Розрахунок при фіксованому h = 10^-3
h = 10**-3
y_h = central_diff(M, t0, h)
y_2h = central_diff(M, t0, 2 * h)
y_4h = central_diff(M, t0, 4 * h)
R1 = abs(y_h - exact_val)

# Метод Рунге-Ромберга
y_R = y_h + (y_h - y_2h) / 3
R2 = abs(y_R - exact_val)

# Метод Ейткена та порядок точності p
y_E = (y_2h**2 - y_4h * y_h) / (2 * y_2h - (y_4h + y_h))
# Розрахунок порядку точності p
p = (1 / np.log(2)) * np.log(abs((y_4h - y_2h) / (y_2h - y_h)))
R3 = abs(y_E - exact_val)

# Візуалізація результатів
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Графік вологості
ax1.plot(t_axis, M(t_axis), label=r'$M(t)$', color='tab:blue', lw=2)
ax1.set_title('Soil Moisture Model M(t)')
ax1.grid(True)
ax1.legend()

# Графік похибки від кроку h 
ax2.loglog(h_values, errors, label='Error R(h)')
ax2.scatter([h_opt], [R0], color='red', label=f'Optimal h0={h_opt:.1e}')
ax2.set_title("Dependence of Error on Step h")
ax2.set_xlabel('Step h')
ax2.set_ylabel('Error R')
ax2.grid(True, which="both", ls="-")
ax2.legend()

plt.tight_layout()
plt.show()

# Вивід результатів у консоль
print(f"\nАналіз у точці t0 = {t0}:")
print(f"Точна похідна: {exact_val:.10f}")
print(f"R1 (h=10^-3): {R1:.2e}")
print(f"Метод Рунге-Ромберга (y_R): {y_R:.10f} (Похибка R2: {R2:.2e})")
print(f"Метод Ейткена (y_E): {y_E:.10f} (Похибка R3: {R3:.2e})")
print(f"Розрахований порядок точності p: {p:.4f}")
