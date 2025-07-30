import unittest
import numpy as np
import signal_tools as sig
import tv_models as models


class TestTools(unittest.TestCase):
    N = 501

    def test_binomial_func(self):
        F = sig.binomial_cumulative(self.N)
        zero_st = abs(F[0]) ** 2 < 1e-06
        one_end = abs(1 - F[-1]) ** 2 < 1e-06
        diff = F[1:] - F[:-1]
        assert (zero_st and one_end and np.all(diff >= 0))

    def test_sign_verification(self):
        F = sig.binomial_cumulative(self.N-1)
        risk = 0.01
        ids = np.argwhere(F <= risk)
        thr = ids.flatten()[-1]
        results = np.array(500*[0], dtype=bool)
        sgn = np.sqrt(self.N) / 2
        for i in range(500):
            e = np.random.randn(self.N)
            results[i] = sig.sign_test(e, thr)
        num_passes = np.count_nonzero(results)
        assert num_passes >= (1-risk)*500 - sgn

    def test_kronecker_products(self):
        a = np.random.randn(8)
        b = np.random.randn(5)
        c = np.kron(a, b)
        d = sig.vec_kron(a, b)
        e = c - d
        I = np.identity(50)
        C = np.kron(I, b)
        D = sig.diag_kron(I, b)
        E = C - D
        assert np.dot(e, e) < 1e-16
        assert np.trace(np.abs(E)) < 1e-16

    def test_polynomial_basis(self):
        t = np.arange(-100, 101, 1)
        f, A = sig.polynomials(5, t)
        e = f[:, 33] - np.dot(A, f[:, 34])
        fo, _ = sig.w_orthonormalize(f, np.ones(201))
        F = np.dot(fo, fo.T)
        E = np.identity(5) - F
        assert np.dot(e, e) < 1e-08
        assert np.sum(np.abs(E)) / 25 < 1e-12

    def test_fourier_basis(self):
        t = np.arange(-100, 101, 1)
        f, A = sig.fourier_basis(5, t)
        e = f[:, 33] - np.dot(A, f[:, 34])
        fo, _ = sig.w_orthonormalize(f, np.ones(201))
        F = np.dot(fo, fo.T)
        E = np.identity(5) - F
        assert np.dot(e, e) < 1e-16
        assert np.sum(np.abs(E)) / 25 < 1e-12

    def test_autocorrelation(self):
        x = np.random.randn(5000)
        rx = sig.autocorrelation(x, 500)
        z = np.zeros(5000)
        for t in range(1, 5000):
            z[t] = 0.95*z[t-1] + x[t]
        rz = sig.autocorrelation(z, 500)
        tr = 0.95 ** np.arange(31)

        assert abs(rx[0] - 1) < 1e-16
        assert np.max(np.abs(rx[1:])) < 1e-01
        assert abs(rz[0] - 1) < 1e-16
        assert np.max(np.abs(rz[:30] - tr[:30])) < 5e-02


class TestModels(unittest.TestCase):

    def test_ARX_stability(self):
        a_st = np.array([[0.2, 0.1],
                         [-0.5, 0.5],
                         [0.3, -0.4]])
        a_unst = np.array([[0.2, 0.1],
                         [-0.5, 1.3],
                         [0.3, -0.4]])

        ind_unst = models.ARX.check_stability(a_unst)
        assert models.ARX.check_stability(a_st)
        assert ind_unst[0] == 1

    def test_ARX_regressors(self):
        model_arx = models.ARX(2, (1, 2))
        model_ar = models.ARX(2, 0)
        model_fir = models.ARX(0, (1, 2))
        y = np.array([1, 2, 3, 4, 5])
        u = np.array([[-1, 0.1], [-2, 0.2], [-3, 0.3], [-4, 0.4], [-5, 0.5], [-6, 0.6]])
        Phi_arx = model_arx.regression_vectors(y, u)
        Phi_ar = model_ar.regression_vectors(y, u)
        Phi_fir = model_fir.regression_vectors(y, u)

        oracle_arx = np.array([[0, 0, -1, 0.1, 0], [1, 0, -2, 0.2, 0.1],
                               [2, 1, -3, 0.3, 0.2], [3, 2, -4, 0.4, 0.3], [4, 3, -5, 0.5, 0.4]])
        oracle_ar = np.array([[0, 0], [1, 0], [2, 1], [3, 2], [4, 3]])
        oracle_fir = np.array([[-1, 0.1, 0], [-2, 0.2, 0.1], [-3, 0.3, 0.2], [-4, 0.4, 0.3], [-5, 0.5, 0.4], [-6, 0.6, 0.5]])

        assert np.all(Phi_arx == oracle_arx)
        assert np.all(Phi_ar == oracle_ar)
        assert np.all(Phi_fir == oracle_fir)

    def test_output_generation(self):
        model_fir = models.ARX(0, 6)
        model_ar = models.ARX(1, 0)

        b = np.array([1, 0, 0, -1, 2, 3])
        b = np.tile(b, (10, 1))
        u = np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]])

        a = np.tile([0.9], (10, 1))

        model_fir.set_params(None, b)
        y_fir = model_fir.generate_output(u, var_e=0)

        model_ar.set_params(a, None)
        e = np.array([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]])
        y_ar = model_ar.generate_output(None, e=e)

        y_fir_oracle = np.concatenate([b[0], np.zeros(4)])
        y_ar_oracle = np.concatenate([np.zeros(1), 0.9**np.arange(9)])

        assert np.all(y_fir == y_fir_oracle)
        assert np.all(np.abs(y_ar - y_ar_oracle) < 0.0001)
