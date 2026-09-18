"""
Numerical solution of a polyelectrolyte system.

This module implements a numerical solver for a coupled system of
nonlinear differential equations describing the electrostatic potential
and polymer concentration profiles in a polyelectrolyte system.

The equations are discretised using finite differences and solved
iteratively using a damped Newton–Raphson method. The resulting linear
systems are handled using sparse matrix routines from SciPy.

The solver supports Dirichlet and Neumann boundary conditions for both
the electrostatic potential and the square root of the normalised
polymer concentration.

The simulation results are stored in pickle files containing:
    - Spatial coordinates.
    - Interleaved electrostatic potential and square-root
      concentration values.
    - A dictionary containing the simulation parameters.

Dependencies:
    - Python 3
    - NumPy
    - SciPy

Warning:
    Only load pickle files from trusted sources, as unpickling
    untrusted data can execute arbitrary code.
"""

import pickle
import numpy as np
import scipy as sp

class Polyelectrolyte:
    """
    Represent a polyelectrolyte system and its numerical parameters.

    This class defines the physical parameters, boundary conditions,
    numerical settings, and output filename required to solve the
    coupled equations for the electrostatic potential and normalised
    polymer concentration profiles.
    """

    def __init__(self):
        """
        Initialise the polyelectrolyte system with default parameters.

        The parameter dictionary contains the physical properties of
        the system, numerical discretisation settings, boundary
        conditions, convergence tolerance, and normalisation scheme.
        """

        self.params = {

            # ========================================================
            # Physical parameters
            # ========================================================

            'a': 5,  # Monomer size.
            'c_salt': 6.02e-8,  # Bulk salt concentration (Å^-3; 0.1 mM).
            'f': 1,  # Fraction of charged monomers.
            'phib2': 1e-6,  # Bulk polymer concentration parameter.
            'v': 50,  # Excluded-volume interaction parameter.
            'e': 80,  # Relative dielectric permittivity.
            'T': 300,  # Temperature (K).
            'lb': 7.2,  # Bjerrum length (Å).
            'w2': 0.0,  # Higher-order polymer interaction parameter.

            # ========================================================
            # Spatial discretisation
            # ========================================================

            'x0': 0.,  # Initial spatial coordinate.
            'xn': 60.,  # Final spatial coordinate.
            'nx': 11,  # Number of spatial grid points.

            # ========================================================
            # Numerical solver settings
            # ========================================================

            'relaxation_factor': 0.007,  # Newton–Raphson relaxation factor.
            'error': 1e-6,  # Convergence tolerance.

            # ========================================================
            # Boundary conditions
            # ========================================================

            # Boundary values for the electrostatic potential (y)
            # and the square root of the normalised concentration (h).

            'y0': -1,  # Potential boundary value at x = x0.
            'h0': 0,  # Concentration-related boundary value at x = x0.
            'yn': 0,  # Potential boundary value at x = xn.
            'hn': 1,  # Concentration-related boundary value at x = xn.

            # Select Dirichlet (True) or Neumann (False) conditions
            # for y(x0), h(x0), y(xn), and h(xn), respectively.
            'dirichlet_boundary': [True, True, True, True],

            # Select the characteristic length used for normalisation:
            # 0: D = 30
            # 1: D = Bjerrum length
            # 2: D = 1
            'normalisation': 0
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
