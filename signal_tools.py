import numpy as np
import scipy.special


def binomial_cumulative(N):
    # Faster and more reliable implementation than the one based on binomial coefficients
    nom = 0
    den = np.zeros(N)
    F = np.zeros(N+1)
    for n in range(1, N):
        d = np.log(n)
        nom += d
        den[n] = nom
    norm = (N - 1) * np.log(2)
    P = np.exp(nom - norm - den - den[::-1])
    for n in range(1, N+1):
        F[n] = F[n-1] + P[n-1]
    return F[1:]


def sign_test(e, lower_bound, upper_bound=None):
    res = e - e.mean()
    sgn = np.sign(res)
    sgn[sgn == 0] = -1
    changes = sgn[1:] * sgn[:-1]
    changes[changes == 1] = 0
    num_changes = -np.sum(changes)
    return (num_changes >= lower_bound) and (num_changes <= upper_bound if upper_bound is not None else True)


def vec_kron(a, b):
    # Vector-to-vector Kronecker product
    na = len(a)
    if na <= 10:
        w = a.flatten()
        v = b.flatten()
        nb = len(v)
        c = np.zeros(na*nb)
        for i in range(na):
            c[nb*i:nb*(i+1)] = w[i]*v
        return c
    else:
        return np.kron(a, b)


def group_vec_kron(a, b):
    # Prepares a matrix of vector-to-vector Kronecker products, faster than using numpy.kron
    na, K = a.shape
    nb = len(b)
    c = np.zeros((na*nb, K))
    for i in range(na):
        c[nb*i:nb*(i+1), :] = a[i, :]*b
    return c


def diag_kron(A, b):
    # For diagonal and vector kronecker product, it can be much faster than general purpose numpy.kron
    na, _ = A.shape
    v = b.flatten()
    nb = len(v)
    C = np.zeros((na, na*nb))
    for i in range(na):
        C[i, nb*i:nb*(i+1)] = A[i, i]*v
    return C


def polynomials(m, t, normalize=True):
    # Function prepares the matrix of m polynomial basis functions, either normalized or not
    K = len(t)
    f = np.ones((m, K))
    Gamma = np.identity(m)
    f[0, :] = f[0, :] / K
    tau = t / K
    k = K // 2 if normalize else 1
    for i in range(1, m):
        f[i, :] = f[i-1, :] * tau
        kp = k
        sgn = -1
        for j in range(i-1, -1, -1):
            Gamma[i, j] = sgn*scipy.special.binom(i, i-j) / kp
            sgn *= -1
            kp *= k

    return f, Gamma


def fourier_basis(m, t):
    K = len(t)
    k = K // 2
    f = np.ones((m, K))
    m0 = m // 2
    arg = (np.pi*t) / (2*k)
    const = np.pi / (2*k)
    A = np.identity(m)
    for i in range(1, m0+1):
        f[2*i-1, :] = np.sin(i*arg)
        f[2*i, :] = np.cos(i*arg)
        a = np.sin(const*i)
        b = np.cos(const*i)
        A[2*i-1, 2*i-1] = b
        A[2*i-1, 2*i] = -a
        A[2*i, 2*i-1] = a
        A[2*i, 2*i] = b
    return f, A


def w_orthonormalize(f, w):
    # Algorithm that performs orthonormalization of a given basis functions with a given weighting sequence
    fw = w * f
    Q = np.dot(fw, f.T)
    Qp = np.linalg.cholesky(Q)
    return np.linalg.solve(Qp, f), Qp


def irw_transition_matrix(n, m):
    # Preparing transition matrix for the Integrated Random Walk model
    nm = n*m
    F = np.zeros((nm, nm))
    sg = -1
    # Setting the transition matrix according to the IRW model
    for i in range(m):
        f = sg * scipy.special.binom(m, i + 1)
        sg *= -1
        F[:n, i * n:(i + 1) * n] = -f * np.eye(n)
    F[n:, :nm - n] = np.eye(nm - n)
    return F


def autocorrelation(x, p=0):
    r = np.zeros(p+1)
    T = len(x)
    for i in range(p+1):
        r[i] = np.dot(x[:T-i], x[i:]) / (T - i)
    return r / r[0]