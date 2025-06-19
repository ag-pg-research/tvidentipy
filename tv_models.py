import numpy as np


class ARX:
    def __init__(self, ny, nu):
        self.ny = ny
        if isinstance(nu, tuple):
            self.nu = nu
            self.num_in = len(nu)
        elif nu == 0:
            self.nu = (0,)
            self.num_in = 0
        else:
            self.nu = (nu,)
            self.num_in = 1
        self.T = 0
        self.a = []
        self.b = []

    def set_params(self, a, b):
        """
        :param a: time-varying a parameters (T x ny)
        :param b: time-varying b parameters for each input (num_in x T x nu)
        :return:
        """
        dim_a = a.shape
        Ta = dim_a[0]
        dim_b = b.shape
        if len(dim_b) < 2:
            b = np.reshape(b, (1, dim_b[0], dim_b[1]))
        Tb = dim_b[1]
        if dim_b[2] != self.nu:
            raise Exception("Wrong number of coefficients in b!")
        elif dim_b[0] != self.num_in:
            raise Exception("Wrong number of polynomials b!")

        self.T = min(Ta, Tb)
        self.a = a[1:self.T]
        self.b = b[:, 1:self.T, :]

    @staticmethod
    def check_stability(a):
        """
        Functions checks if every characteristic polynomial from time-varying coefficients is stable
        :param a: time-varying a parameters (T x ny)
        :return: True if the corresponding object is stable, list of indices of unstable polynomials
        """
        inst_id = []
        for t, at in enumerate(a):
            poly = np.concatenate(([1], -at))
            roots = np.roots(poly)
            rad = np.abs(roots)
            if np.any(rad >= 1):
                inst_id.append(t)

        if len(inst_id) == 0:
            return True
        else:
            return inst_id

    def regression_vectors(self, y, u):
        """
        Function generates a matrix of regression vectors, assuming zero initial conditions
        :param y: output data of length T
        :param u: input data (T x num inputs)
        :return: Phi - matrix of regression vectors (T x ny+sum(nu)), first columns contain the outputs, next columns
                        contain inputs
        """
        Tu, num_in = u.shape
        if num_in != self.num_in and self.num_in > 0:
            raise Exception("Number of inputs doesn't match the declared one!")
        Ty = y.size
        T = min(Tu, Ty)
        Phi = np.zeros((T, self.ny + np.sum(self.nu)))
        # Adding output signals to regressors
        for i in range(self.ny):
            Phi[i+1:, i] = y[:T-i-1]
        # Adding input signals
        nid = self.ny
        for iu, nu in enumerate(self.nu):
            for i in range(nu):
                Phi[i:, nid] = u[:T-i, iu]
                nid += 1

        return Phi
