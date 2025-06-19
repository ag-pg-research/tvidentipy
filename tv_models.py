import numpy as np


class ARX:
    def __init__(self, ny, nu):
        self.ny = ny
        self.nu = max(nu)
        self.num_in = len(nu) if isinstance(nu, list) else 1
        self.T = 0
        self.a = []
        self.b = []

    def set_params(self, a, b):
        """
        :param a: time-varying a parameters (ny x T)
        :param b: time-varying b parameters for each input (num_in x nu x T)
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
        self.b = b[:, :, 1:self.T]

    @staticmethod
    def check_stability(a):
        """
        Functions checks if every characteristic polynomial from time-varying coefficients is stable
        :param a: time-varying a parameters (ny x T)
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
