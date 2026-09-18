import pickle
import numpy as np
import scipy as sp

class Polieletrolito:
    def __init__(self):
        self.params = {
            # Tipo de condição de contorno: 'potential' (y(0)=y_s) ou 'charge' (y'(0)=yprime0)
            # 'bc_type': 'potential',

            # Se 'potential', fixe o potencial superficial (adimensional) y(0)=y_s
            'a': 5,
            'c_salt': 6.02e-8,  # A^{-3} 0,1 mM #fig 1a
            'f': 1,
            # 'y_s': -1,  # típico: -0.5 a -2 (em unidades e*psi/kBT)1
            'phib2': 1e-6,
            'v': 50,
            'e': 80,
            'T': 300,
            'lb': 7.2,  # angstrom
            'w2': 0.0,

            # Se 'charge', fixe a derivada do potencial na parede
            # 'yprime0': 1.0,

            # Grade numérica
            'x0': 0.,
            'xn': 60.,
            'nx': 11,
            # 'h' : 0.1,
            'relaxation_tax': 0.007,

            'y0': -1,
            'h0': 0,
            'yn': 0,
            'hn': 1,

            'dirichlet_boundary': [True, True, True, True],

            'boundary': 'dirichlet',

            'error': 1e-6,

            'normalisation':0
        }

        self.filename = 'file.pkl'

    def run(self, p=True):

        N = self.params['nx']
        R = np.zeros(2 * N)
        Y = np.zeros(2 * N)
        J = sp.sparse.lil_matrix((2*N, 2*N))
        X = np.linspace(0, self.params['xn'], N)
        dx = X[1] - X[0]
        dy = np.ones(N)
        iter = 0

        if self.params['dirichlet_boundary'][0]:
            J[0, 0] = 1
        else:
            J[0, 0] = -1 / dx
            J[0, 2] = 1 / dx

        if self.params['dirichlet_boundary'][1]:
            J[1, 1] = 1
        else:
            J[1, 1] = -1 / dx
            J[1, 3] = 1 / dx

        if self.params['dirichlet_boundary'][2]:
            J[2 * N - 2, 2 * N - 2] = 1
        else:
            J[2 * N - 2, 2 * N - 4] = -1 / dx
            J[2 * N - 2, 2 * N - 2] = 1 / dx

        if self.params['dirichlet_boundary'][3]:
            J[2 * N - 1, 2 * N - 1] = 1
        else:
            J[2 * N - 1, 2 * N - 3] = -1 / dx
            J[2 * N - 1, 2 * N - 1] = 1 / dx

        while np.linalg.norm(dy) > self.params['error']:
            if p:
                print('iter', iter, 'error', np.linalg.norm(dy))

            if self.params['dirichlet_boundary'][0]:
                R[0] = Y[0] -self.params['y0']
            else:
                R[0] = (Y[2] - Y[0])/dx -self.params['y0']

            if self.params['dirichlet_boundary'][1]:
                R[1] = Y[1] -self.params['h0']
            else:
                R[1] = (Y[3] - Y[1]) / dx -self.params['h0']

            if self.params['dirichlet_boundary'][2]:
                R[-2] = Y[-2] - self.params['yn']
            else:
                R[-2] = (Y[-2] - Y[-4]) / dx - self.params['yn']

            if self.params['dirichlet_boundary'][3]:
                R[-1] = Y[-1] - self.params['hn']
            else:
                R[-1] = (Y[-1] - Y[-3]) / dx - self.params['hn']

            for i in range(2, 2*N-2, 2):

                J[i, i - 2] = 1
                J[i, i] = -2 - dx**2 * self.params['gamma'] * np.cosh(Y[i])
                J[i, i] -= dx**2 * self.params['delta'] * np.exp(Y[i])
                J[i, i + 1] = - dx**2*self.params['delta']*(-2*Y[i+1])
                J[i, i + 2] = 1

                J[i + 1, i - 1] = 1
                J[i + 1, i] = -dx ** 2 * self.params['zeta'] * Y[i + 1]

                J[i + 1, i + 1] = -2
                J[i + 1, i + 1] += -3 * dx ** 2 * self.params['omega'] * Y[i + 1] ** 2
                J[i + 1, i + 1] += dx ** 2 * self.params['omega']
                J[i + 1, i + 1] += -5 * dx ** 2 * self.params['alpha'] * Y[i + 1] ** 4
                J[i + 1, i + 1] += dx ** 2 * self.params['alpha']
                J[i + 1, i + 1] += -dx ** 2 * self.params['zeta'] * Y[i]

                J[i + 1, i + 3] = 1

                R[i] = Y[i-2] -2*Y[i] + Y[i+2] - dx**2 * self.params['gamma'] * np.sinh(Y[i])
                R[i] -= dx**2 * self.params['delta'] * (np.exp(Y[i])-Y[i+1]**2)
                R[i + 1] = Y[i-1] -2*Y[i+1] + Y[i+3]
                R[i + 1] += -dx ** 2 * self.params['omega'] * (Y[i+1] ** 3-Y[i+1])
                R[i + 1] += -dx ** 2 * self.params['alpha'] * (Y[i + 1] ** 5 - Y[i + 1])
                R[i + 1] -= dx**2*self.params['zeta']*Y[i]*Y[i+1]

            dy = sp.sparse.linalg.spsolve(J.tocsr(), -R)

            Y = Y + self.params['relaxation_tax']*dy
            iter+=1

        f = open(self.filename, "wb")
        pickle.dump([X, Y, self.params], f)

        Y2 = np.ones(len(X))

        return X, Y, self.params

    def set_filename(self, name):
        self.filename = name

    def calculate_constants(self):

        if self.params['normalisation'] == 0:
            self.params['D'] = 30
        elif self.params['normalisation'] == 1:
            self.params['D'] = self.params['lb']
        elif self.params['normalisation'] == 2:
            self.params['D'] = 1

        self.params['k2'] = 8 * np.pi * self.params['lb'] * self.params['c_salt']
        self.params['km2'] = 4 * np.pi * self.params['lb'] * self.params['phib2'] * self.params['f']
        self.params['gamma'] = self.params['k2'] * self.params['D'] ** 2
        self.params['delta'] = self.params['km2'] * self.params['D'] ** 2
        self.params['omega'] = 6 * (self.params['D'] / self.params['a']) ** 2 * self.params['v'] * self.params['phib2']
        self.params['zeta'] = 6 * (self.params['D'] / self.params['a']) ** 2 * self.params['f']
        self.params['alpha'] = 6 * (self.params['D'] / self.params['a']) ** 2 * self.params['w2'] / 2 * self.params[
            'phib2'] ** 2
        self.params['xn'] /= self.params['D']

    def set_filename(self, name):
        self.filename = name
