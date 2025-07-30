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
        self.T = None
        self.a = None
        self.b = None

    def set_params(self, a=None, b=None):
        """
        :param a: time-varying a parameters (T x ny)
        :param b: time-varying b parameters for each input (T x sum(nu))
        :return:
        """
        if a is not None:
            dim_a = a.shape
            Ta = dim_a[0]
        else:
            Ta = None
        if b is not None:
            dim_b = b.shape
            Tb = dim_b[0]
            if sum(self.nu) != dim_b[1]:
                raise Exception("Wrong number of parameters!")
        else:
            Tb = None

        if Ta is not None and Tb is not None:
            self.T = min(Ta, Tb)
        elif Ta is None:
            self.T = Tb
        elif Tb is None:
            self.T = Ta
        else:
            self.T = None
        if a is not None:
            self.a = a[:self.T]
        if b is not None:
            self.b = b[:self.T]

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
        if self.num_in > 0:
            Tu, num_in = u.shape
            if num_in != self.num_in and self.num_in > 0:
                raise Exception("Number of inputs doesn't match the declared one!")
        if self.ny > 0:
            Ty = y.size
        if self.num_in > 0 and self.ny > 0:
            T = min(Tu, Ty)
        elif self.num_in > 0:
            T = Tu
        else:
            T = Ty
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

    def generate_output(self, u, e=None, var_e=1):
        if self.T is None:
            raise Exception('Cannot generate output signals without parameters!')

        if e is not None and len(e) != self.T:
            raise Exception('Wrong length of the noise sequence!')

        y = np.zeros(self.T)
        if e is None:
            e = np.random.randn(self.T)*np.sqrt(var_e)

        if self.num_in > 0:
            if u.shape[0] != self.T or u.shape[1] != self.num_in:
                raise Exception('Wrong dimensions of inputs!')
            input_part = ARX(0, self.nu)
            Phiu = input_part.regression_vectors(0, u)
        for t in range(self.ny, self.T):
            y[t] = e[t]
            if self.num_in > 0:
                y[t] += np.dot(Phiu[t], self.b[t])

            if self.ny > 0:
                y[t] += np.dot(np.fliplr(y[t-self.ny:t]), self.a[t]) if self.ny > 1 else self.a[t]*y[t-self.ny]

        return y
