import numpy as np

def generate_iterative(n=100, x_val=2.5):
    A = np.random.uniform(1, 10, (n, n))
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + 1.0
    x_true = np.full(n, x_val)
    B = np.dot(A, x_true)
    np.savetxt('matrix_A_iter.txt', A)
    np.savetxt('vector_B_iter.txt', B)
    return A, B

def simple_iteration(A, B, eps, max_iter=5000):
    n = len(B)
    tau = 1.0 / np.max(np.sum(np.abs(A), axis=1))
    x = np.ones(n)
    for k in range(max_iter):
        x_next = x - tau * (np.dot(A, x) - B)
        if np.max(np.abs(x_next - x)) < eps:
            return x_next, k + 1
        x = x_next
    return x, max_iter

def jacobi_method(A, B, eps, max_iter=5000):
    n = len(B)
    x = np.ones(n)
    D = np.diag(A)
    R = A - np.diagflat(D)
    for k in range(max_iter):
        x_next = (B - np.dot(R, x)) / D
        if np.max(np.abs(x_next - x)) < eps:
            return x_next, k + 1
        x = x_next
    return x, max_iter

def seidel_method(A, B, eps, max_iter=5000):
    n = len(B)
    x = np.ones(n)
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (B[i] - s1 - s2) / A[i, i]
        if np.max(np.abs(x - x_old)) < eps:
            return x, k + 1
    return x, max_iter

A, B = generate_iterative()
eps_0 = 1e-9

print(f"{'Метод':<18} | {'Ітер.':<6} | {'Похибка':<9} | {'Перші 5 значень'}")

for name, method in [("Проста ітерація", simple_iteration),
                     ("Якобі", jacobi_method),
                     ("Зейдель", seidel_method)]:
    sol, iters = method(A, B, eps_0)
    res = np.max(np.abs(np.dot(A, sol) - B))
    sol_preview = ", ".join([f"{val:.4f}" for val in sol[:5]])
    print(f"{name:<18} | {iters:<6} | {res:.2e} | [{sol_preview}, ...]")
