import matplotlib.pyplot as plt
import pickle
import os
import time

class Polieletrolito:
    def __init__(self, gpu = False, normalisation = 2, exemplo = 13):
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

            'normalisation':0,

            'convergence_error':100,
            'lambda0': 0,
            'lambdaf':-1,
            'n_lambda':-1
        }

        self.gpu = gpu

        self.filename = 'file.pkl'
        # self.calculate_constants()

    def run4(self):

        if self.gpu:
            import cupy as cp
            import cupyx.scipy.sparse as cps
            from cupyx.scipy.sparse.linalg import spsolve
        else:
            import numpy as np
            import scipy as sp

        N = self.params['nx']
        R = np.zeros(2 * N)
        Y = np.zeros(2 * N)
        # J = np.zeros((2 * N, 2 * N))
        J = sp.sparse.lil_matrix((2 * N, 2 * N))
        X = np.linspace(0, self.params['xn'], N)
        dx = X[1] - X[0]
        dy = np.ones(N)
        iter = 0

        data = [1,0,0,1,0,0,2,2,2,2,2,3,3,3,3,3,0,0,1,0,0,1]
        columns = [0,1,2,1,2,3,0,1,2,3,4,1,2,3,4,5,2,3,4,3,4,5]
        rows = [0,3,6,11,16,19, 22]

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

            inicio1 = time.perf_counter()
            # dy = np.linalg.solve(J, -R)
            dy = sp.sparse.linalg.spsolve(J.tocsr(), -R)
            fim = time.perf_counter()
            tempo_total = fim - inicio1
            print(f"tempo total: {tempo_total:.6f} s")

            Y = Y + self.params['relaxation_tax']*dy
            iter+=1

        f = open(self.filename, "wb")
        pickle.dump([X, Y, self.params], f)

        if self.gpu:
            X = X.get()
            Y = Y.get()
            Y2 = np.ones(len(X)).get()
        else:
            Y2 = np.ones(len(X))

    def run3(self):

        if self.gpu:
            import cupy as np
        else:
            import numpy as np
            import scipy as sp

        N = self.params['nx']
        R = np.zeros(2 * N)
        Y = np.zeros(2 * N)
        # J = np.zeros((2 * N, 2 * N))
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

        for i in range(2, 2 * N - 2, 2):
            J[i, i - 2] = 1
            J[i, i] = 1
            J[i, i + 1] = 1
            J[i, i + 2] = 1

            J[i + 1, i - 1] = 1
            J[i + 1, i] = 1
            J[i + 1, i + 1] = 1
            J[i + 1, i + 3] = 1

        J_csr = J.tocsr()

        while np.linalg.norm(dy) > self.params['error']:
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

            inicio1 = time.perf_counter()
            # dy = np.linalg.solve(J, -R)
            dy = sp.sparse.linalg.spsolve(J.tocsr(), -R)
            fim = time.perf_counter()
            tempo_total = fim - inicio1
            print(f"tempo total: {tempo_total:.6f} s")

            Y = Y + self.params['relaxation_tax']*dy
            iter+=1

        f = open(self.filename, "wb")
        pickle.dump([X, Y, self.params], f)

        if self.gpu:
            X = X.get()
            Y = Y.get()
            Y2 = np.ones(len(X)).get()
        else:
            Y2 = np.ones(len(X))

    def run(self, p = True):

        if self.gpu:
            import cupy as cp
            import cupyx.scipy.sparse as cps
            from cupyx.scipy.sparse.linalg import spsolve
        else:
            import numpy as np
            import scipy as sp

        N = self.params['nx']
        R = np.zeros(2 * N)
        Y = np.zeros(2 * N)
        # J = np.zeros((2 * N, 2 * N))
        if self.gpu:
            J = sp.sparse.lil_matrix((2 * N, 2 * N))
        else:
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

            # inicio1 = time.perf_counter()
            # dy = np.linalg.solve(J, -R)
            dy = sp.sparse.linalg.spsolve(J.tocsr(), -R)
            # fim = time.perf_counter()
            # tempo_total = fim - inicio1
            # print(f"tempo total: {tempo_total:.6f} s")

            Y = Y + self.params['relaxation_tax']*dy
            iter+=1

        f = open(self.filename, "wb")
        pickle.dump([X, Y, self.params], f)

        if self.gpu:
            X = X.get()
            Y = Y.get()
            Y2 = np.ones(len(X)).get()
        else:
            Y2 = np.ones(len(X))

        return X, Y, self.params

    def set_filename(self, name):
        self.filename = name

    def calculate_constants(self):

        if self.gpu:
            import cupy as np
        else:
            import numpy as np

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

    def run2(self):

        if self.gpu:
            import cupy as np
        else:
            import numpy as np

        if self.params['lambdaf']!= -1:
            lambda_l = np.linspace(self.params['lambda0'],
                                   self.params['lambdaf'],
                                    self.params['n_lambda'])
        else:
            lambda_l = [self.params['lambda0']]

        f = open(self.filename, "wb")
        data = []
        for l in lambda_l:
            self.params['relaxation_tax'] = l
            print('lambda', self.params['relaxation_tax'])
            N = self.params['nx']
            R = np.zeros(2 * N)
            Y = np.zeros(2 * N)
            J = np.zeros((2 * N, 2 * N))
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
                J[2 * N - 2, 2 * N - 4] = -1 / dx
                J[2 * N - 2, 2 * N - 2] = 1 / dx
                J[2 * N - 1, 2 * N - 3] = -1 / dx
                J[2 * N - 1, 2 * N - 1] = 1 / dx

            while np.linalg.norm(dy) > self.params['error']:
                # print('iter', iter, 'error', np.linalg.norm(dy))

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
                    J[i + 1, i] = -dx**2 * self.params['zeta'] * Y[i+1]
                    J[i + 1, i + 1] = -2
                    J[i + 1, i + 1] += -3 * dx**2*self.params['omega']*Y[i+1]**2
                    J[i + 1, i + 1] +=  dx**2*self.params['omega']
                    J[i + 1, i + 1] += -5 * dx ** 2 * self.params['alpha'] * Y[i + 1] ** 4
                    J[i + 1, i + 1] += dx ** 2 * self.params['alpha']
                    J[i + 1, i + 1] -= -dx**2*self.params['zeta']*Y[i]

                    J[i + 1, i + 3] = 1

                    R[i] = Y[i-2] -2*Y[i] + Y[i+2] - dx**2 * self.params['gamma'] * np.sinh(Y[i])
                    R[i] -= dx**2 * self.params['delta'] * (np.exp(Y[i])-Y[i+1]**2)
                    R[i+1] = Y[i-1] -2*Y[i+1] + Y[i+3] - dx**2*self.params['omega']*(Y[i+1]**3-Y[i+1])
                    R[i+1] -= dx**2*self.params['zeta']*Y[i]*Y[i+1]

                dy = np.linalg.solve(J, -R)
                Y = Y + self.params['relaxation_tax']*dy
                iter+=1

            X1 = X.copy()
            Y1 = Y.copy()

            error = 1
            error_l = []
            N_l = []
            print('N = ', N, 'antes convergencia', 'iter = ', iter)
            while error> self.params['convergence_error']:
                N -= 1
                N *= 2
                N += 1

                # N = self.params['nx']
                R = np.zeros(2 * N)
                Y = np.zeros(2 * N)
                J = np.zeros((2 * N, 2 * N))
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
                    J[2 * N - 2, 2 * N - 4] = -1 / dx
                    J[2 * N - 2, 2 * N - 2] = 1 / dx
                    J[2 * N - 1, 2 * N - 3] = -1 / dx
                    J[2 * N - 1, 2 * N - 1] = 1 / dx

                while np.linalg.norm(dy) > self.params['error']:
                    # print('iter', iter, 'error', np.linalg.norm(dy))

                    if self.params['dirichlet_boundary'][0]:
                        R[0] = Y[0] - self.params['y0']
                    else:
                        R[0] = (Y[2] - Y[0]) / dx - self.params['y0']

                    if self.params['dirichlet_boundary'][1]:
                        R[1] = Y[1] - self.params['h0']
                    else:
                        R[1] = (Y[3] - Y[1]) / dx - self.params['h0']

                    if self.params['dirichlet_boundary'][2]:
                        R[-2] = Y[-2] - self.params['yn']
                    else:
                        R[-2] = (Y[-2] - Y[-4]) / dx - self.params['yn']

                    if self.params['dirichlet_boundary'][3]:
                        R[-1] = Y[-1] - self.params['hn']
                    else:
                        R[-1] = (Y[-1] - Y[-3]) / dx - self.params['hn']

                    for i in range(2, 2 * N - 2, 2):
                        J[i, i - 2] = 1
                        J[i, i] = -2 - dx ** 2 * self.params['gamma'] * np.cosh(Y[i])
                        J[i, i] -= dx ** 2 * self.params['delta'] * np.exp(Y[i])
                        J[i, i + 1] = - dx ** 2 * self.params['delta'] * (-2 * Y[i + 1])
                        J[i, i + 2] = 1

                        J[i + 1, i - 1] = 1
                        J[i + 1, i] = -dx ** 2 * self.params['zeta'] * Y[i + 1]
                        J[i + 1, i + 1] = -2 - 3 * dx ** 2 * self.params['omega'] * Y[i + 1] ** 2
                        J[i + 1, i + 1] += dx ** 2 * self.params['omega'] - dx ** 2 * self.params['zeta'] * Y[i]

                        J[i + 1, i + 3] = 1

                        R[i] = Y[i - 2] - 2 * Y[i] + Y[i + 2] - dx ** 2 * self.params['gamma'] * np.sinh(Y[i])
                        R[i] -= dx ** 2 * self.params['delta'] * (np.exp(Y[i]) - Y[i + 1] ** 2)
                        R[i + 1] = Y[i - 1] - 2 * Y[i + 1] + Y[i + 3] - dx ** 2 * self.params['omega'] * (
                                    Y[i + 1] ** 3 - Y[i + 1])
                        R[i + 1] -= dx ** 2 * self.params['zeta'] * Y[i] * Y[i + 1]

                    dy = np.linalg.solve(J, -R)
                    Y = Y + self.params['relaxation_tax'] * dy
                    iter += 1

                X2 = X.copy()
                Y2 = Y.copy()

                y1 = Y1[0::2]
                eta1 = Y1[1::2]
                y2 = Y2[0::2]
                eta2 = Y2[1::2]
                y3 = y2[0::2]
                eta3 = eta2[0::2]
                deltay = y1 - y3
                deltae = eta1 - eta3
                v = np.concatenate((deltay, deltae), axis=0)
                error = np.linalg.norm(v)
                # print('erro',erro)
                X1 = X2.copy()
                Y1 = Y2.copy()
                print('N = ', N, 'error = ', error, 'iter = ', iter, ' dx =  ' , dx)
                error_l.append(error)
                # if len(error_l) > 2 and error_l[-1]> error_l[-2]> error_l[-3]:
                #     break
                N_l.append([((N-1)/2)+1, N])
            data.append([N_l[-1][0],  l, X2, Y2, self.params, N_l, error_l])
        pickle.dump(data, f)
        f.close()

    def set_filename(self, name):
        self.filename = name

    def examples(self, exemplo = 2):
        # exemplos do artigo

        if exemplo == 2:
            self.params['a'] = 5
            self.params['f'] = 1
            self.params['y0'] = -0.5
        elif exemplo == 3:
            self.params['a'] = 10
            self.params['f'] = 1
            self.params['y0'] = -0.5
        elif exemplo == 4:
            self.params['a'] = 5
            self.params['f'] = 0.1
            self.params['y0'] = -0.5
        elif exemplo == 5:
            self.params['Xmax'] = 150
            self.params['y0'] = -0.5
            self.params['c_salt'] *= 700
            self.params['f'] = 0.12
        elif exemplo == 6:
            self.params['Xmax'] = 150
            self.params['y0'] = -0.5
            self.params['c_salt'] *= 700
            self.params['f'] = 0.1
        elif exemplo == 7:
            self.params['Xmax'] = 150
            self.params['y0'] = -0.5
            self.params['c_salt'] *= 700
            self.params['f'] = 0.09
        elif exemplo == 8:
            self.params['Xmax'] = 150
            self.params['y0'] = -0.5
            self.params['c_salt'] *= 700
            self.params['f'] = 0.08
        elif exemplo == 9:
            self.params['c_salt'] *= 10
        elif exemplo == 10:
            self.params['c_salt'] *= 100
        elif exemplo == 11:
            self.params['c_salt'] *= 1e-1
        elif exemplo == 12:
            self.params['c_salt'] *= 1e-2
        elif exemplo == 13:
            self.params['y0'] = -1