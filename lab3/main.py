import csv
import math
import matplotlib.pyplot as plt

# 1. Зчитування даних з CSV 
def read_data(filename):
    x, f = [], []
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['Month']))
            f.append(float(row['Temp']))
    return x, f

# 2. Формування матриці B та вектора C для МНК 
def form_matrix_b(x, m):
    size = m + 1
    matrix = [[0.0] * size for _ in range(size)]
    for k in range(size):
        for l in range(size):
            matrix[k][l] = sum(xi ** (k + l) for xi in x)
    return matrix

def form_vector_c(x, f, m):
    size = m + 1
    vector = [0.0] * size
    for k in range(size):
        vector[k] = sum(fi * (xi ** k) for xi, fi in zip(x, f))
    return vector

# 3. Розв'язування СЛАР методом Гаусса з вибором головного елемента 
def gauss_solve(A, b):
    n = len(b)
    # Прямий хід 
    for k in range(n):
        # Вибір головного елемента 
        max_row = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[max_row][k]):
                max_row = i
        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]

        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            b[i] -= factor * b[k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]

    # Зворотній хід 
    x_sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_ax = sum(A[i][j] * x_sol[j] for j in range(i + 1, n))
        x_sol[i] = (b[i] - sum_ax) / A[i][i]
    return x_sol

# 4. Обчислення значення многочлена 
def polynomial(x_val, coef):
    return sum(a * (x_val ** i) for i, a in enumerate(coef))

# 5. Обчислення дисперсії (середньоквадратичної похибки) 
def calculate_variance(y_true, y_approx):
    n = len(y_true)
    mse = sum((yt - ya)**2 for yt, ya in zip(y_true, y_approx)) / n
    return math.sqrt(mse)

# Завантаження даних
x_data, y_data = read_data('data.csv')

variances = []
degrees = list(range(1, 11)) # m від 1 до 10 

# Знаходимо дисперсії для кожного ступеня
for m in degrees:
    A = form_matrix_b(x_data, m)
    C = form_vector_c(x_data, y_data, m)
    coef = gauss_solve(A, C)
    y_approx = [polynomial(xi, coef) for xi in x_data]
    var = calculate_variance(y_data, y_approx)
    variances.append(var)
    print(f"Ступінь m={m}: Дисперсія = {var:.4f}")

# Вибір оптимального ступеня (мінімум дисперсії) 
optimal_m = degrees[variances.index(min(variances))]
print(f"\nОптимальний ступінь полінома: m={optimal_m}")

# Розрахунок для оптимального ступеня
A_opt = form_matrix_b(x_data, optimal_m)
C_opt = form_vector_c(x_data, y_data, optimal_m)
final_coef = gauss_solve(A_opt, C_opt)

# Прогноз на наступні 3 місяці (25, 26, 27)
x_future = [25, 26, 27]
y_future = [polynomial(xf, final_coef) for xf in x_future]
print(f"Прогноз на наступні 3 місяці: {['%.2f' % v for v in y_future]}")

# Побудова графіків
plt.figure(figsize=(12, 8))

# Графік 1: Дані та Апроксимація
plt.subplot(2, 1, 1)
plt.scatter(x_data, y_data, color='red', label='Фактичні дані')
x_range = [x/10.0 for x in range(int(x_data[0]*10), int(x_future[-1]*10))]
y_range = [polynomial(xr, final_coef) for xr in x_range]
plt.plot(x_range, y_range, label=f'Апроксимація (m={optimal_m})')
plt.scatter(x_future, y_future, color='green', label='Прогноз')
plt.title('Апроксимація температури та прогноз')
plt.legend()
plt.grid(True)

# Графік 2: Похибка апроксимації 
plt.subplot(2, 1, 2)
errors = [abs(y - polynomial(x, final_coef)) for x, y in zip(x_data, y_data)]
plt.bar(x_data, errors, color='orange', label='Похибка |f(x) - φ(x)|')
plt.title('Графік похибки')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Графік залежності дисперсії від ступеня 
plt.figure()
plt.plot(degrees, variances, marker='o')
plt.xlabel('Ступінь m')
plt.ylabel('Дисперсія')
plt.title('Залежність дисперсії від ступеня многочлена')
plt.grid(True)
plt.show()
