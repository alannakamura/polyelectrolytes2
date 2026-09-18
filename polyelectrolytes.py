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
        """
        Solve the coupled nonlinear equations using a damped
        Newton–Raphson method.

        The equations are discretised on a uniform spatial grid using
        finite differences. The electrostatic potential and the square
        root of the normalised polymer concentration are stored in an
        interleaved solution vector.

        Parameters
        ----------
        p : bool, optional
            If True, print the iteration number and convergence error
            during the numerical solution. The default is True.

        Returns
        -------
        X : numpy.ndarray
            Spatial grid coordinates.
        Y : numpy.ndarray
            Interleaved electrostatic potential and square-root
            concentration profiles.
        params : dict
            Dictionary containing the simulation parameters.
        """

        # Retrieve the number of spatial grid points.
        N = self.params['nx']

        # Initialise the residual vector for the coupled equations.
        R = np.zeros(2 * N)

        # Initialise the solution vector with zero values.
        # Even indices represent the electrostatic potential,
        # while odd indices represent the square root of the
        # normalised polymer concentration.
        Y = np.zeros(2 * N)

        # Initialise the sparse Jacobian matrix in LIL format,
        # allowing efficient modification of individual elements.
        J = sp.sparse.lil_matrix((2 * N, 2 * N))

        # Generate a uniformly spaced spatial grid.
        X = np.linspace(0, self.params['xn'], N)

        # Calculate the spatial grid spacing.
        dx = X[1] - X[0]

        # Initialise the correction vector to ensure that the
        # Newton–Raphson iteration begins.
        dy = np.ones(N)

        # Initialise the iteration counter.
        iter = 0

        # ============================================================
        # Initialise the Jacobian matrix for the boundary conditions
        # ============================================================

        # Apply the boundary condition for the electrostatic potential
        # at the left boundary (x = x0).
        if self.params['dirichlet_boundary'][0]:

            # Dirichlet condition: prescribe the potential value.
            J[0, 0] = 1

        else:

            # Neumann condition: prescribe the potential derivative
            # using a first-order forward finite difference.
            J[0, 0] = -1 / dx
            J[0, 2] = 1 / dx

        # Apply the boundary condition for the square root of the
        # normalised polymer concentration at the left boundary (x = x0).
        if self.params['dirichlet_boundary'][1]:

            # Dirichlet condition: prescribe the square-root concentration value.
            J[1, 1] = 1

        else:

            # Neumann condition: prescribe its derivative using
            # a first-order forward finite difference.
            J[1, 1] = -1 / dx
            J[1, 3] = 1 / dx

        # Apply the boundary condition for the electrostatic potential
        # at the right boundary (x = xn).
        if self.params['dirichlet_boundary'][2]:

            # Dirichlet condition: prescribe the potential value.
            J[2 * N - 2, 2 * N - 2] = 1

        else:

            # Neumann condition: prescribe the potential derivative
            # using a first-order backward finite difference.
            J[2 * N - 2, 2 * N - 4] = -1 / dx
            J[2 * N - 2, 2 * N - 2] = 1 / dx

        # Apply the boundary condition for the square root of the
        # normalised polymer concentration at the right boundary (x = xn).
        if self.params['dirichlet_boundary'][3]:

            # Dirichlet condition: prescribe the square-root concentration value.
            J[2 * N - 1, 2 * N - 1] = 1

        else:

            # Neumann condition: prescribe its derivative using
            # a first-order backward finite difference.
            J[2 * N - 1, 2 * N - 3] = -1 / dx
            J[2 * N - 1, 2 * N - 1] = 1 / dx

        # Continue the Newton–Raphson iterations until the Euclidean norm
        # of the correction vector falls below the convergence tolerance.
        while (np.linalg.norm(dy) * self.params['relaxation_factor']
               > self.params['error']):
            # Print the current iteration number and the Euclidean norm of the
            # correction vector if iteration monitoring is enabled.
            if p:
                print('iter', iter,
                      'error', np.linalg.norm(dy) * self.params['relaxation_factor'])

            # ============================================================
            # Evaluate the residuals associated with the boundary conditions
            # ============================================================

            # Evaluate the boundary residual for the electrostatic potential
            # at the left boundary (x = x0).
            if self.params['dirichlet_boundary'][0]:

                # Dirichlet condition: difference between the calculated
                # potential and the prescribed boundary value.
                R[0] = Y[0] - self.params['y0']

            else:

                # Neumann condition: difference between the potential derivative,
                # approximated using a first-order forward finite difference,
                # and the prescribed derivative.
                R[0] = (Y[2] - Y[0]) / dx - self.params['y0']

            # Evaluate the boundary residual for the square root of the
            # normalised polymer concentration at the left boundary (x = x0).
            if self.params['dirichlet_boundary'][1]:

                # Dirichlet condition: difference between the calculated
                # square-root concentration and the prescribed boundary value.
                R[1] = Y[1] - self.params['h0']

            else:

                # Neumann condition: difference between the derivative of the
                # square-root concentration, approximated using a first-order
                # forward finite difference, and the prescribed derivative.
                R[1] = (Y[3] - Y[1]) / dx - self.params['h0']

            # Evaluate the boundary residual for the electrostatic potential
            # at the right boundary (x = xn).
            if self.params['dirichlet_boundary'][2]:

                # Dirichlet condition: difference between the calculated
                # potential and the prescribed boundary value.
                R[-2] = Y[-2] - self.params['yn']

            else:

                # Neumann condition: difference between the potential derivative,
                # approximated using a first-order backward finite difference,
                # and the prescribed derivative.
                R[-2] = (Y[-2] - Y[-4]) / dx - self.params['yn']

            # Evaluate the boundary residual for the square root of the
            # normalised polymer concentration at the right boundary (x = xn).
            if self.params['dirichlet_boundary'][3]:

                # Dirichlet condition: difference between the calculated
                # square-root concentration and the prescribed boundary value.
                R[-1] = Y[-1] - self.params['hn']

            else:

                # Neumann condition: difference between the derivative of the
                # square-root concentration, approximated using a first-order
                # backward finite difference, and the prescribed derivative.
                R[-1] = (Y[-1] - Y[-3]) / dx - self.params['hn']

            # ============================================================
            # Assemble the Jacobian matrix at the interior grid points
            # ============================================================

            # Iterate over the interior grid points, excluding the boundaries.
            # Even indices correspond to the electrostatic potential,
            # while odd indices correspond to the square root of the
            # normalised polymer concentration.
            for i in range(2, 2 * N - 2, 2):
                # --------------------------------------------------------
                # Jacobian contributions from the electrostatic equation
                # --------------------------------------------------------

                # Contribution from the potential at the previous grid point.
                J[i, i - 2] = 1

                # Diagonal contribution from the finite-difference operator
                # and the nonlinear electrostatic terms.
                J[i, i] = -2 - dx ** 2 * self.params['gamma'] * np.cosh(Y[i])
                J[i, i] -= dx ** 2 * self.params['delta'] * np.exp(Y[i])

                # Coupling contribution from the square root of the
                # normalised polymer concentration at the current grid point.
                J[i, i + 1] = - dx ** 2 * self.params['delta'] * (-2 * Y[i + 1])

                # Contribution from the potential at the next grid point.
                J[i, i + 2] = 1

                # --------------------------------------------------------
                # Jacobian contributions from the polymer concentration equation
                # --------------------------------------------------------

                # Contribution from the square-root concentration
                # at the previous grid point.
                J[i + 1, i - 1] = 1

                # Coupling contribution from the electrostatic potential.
                J[i + 1, i] = -dx ** 2 * self.params['zeta'] * Y[i + 1]

                # Initialise the diagonal element with the contribution
                # from the second-order finite-difference operator.
                J[i + 1, i + 1] = -2

                # Add the contributions from the cubic polymer interaction term.
                J[i + 1, i + 1] += -3 * dx ** 2 * self.params['omega'] * Y[i + 1] ** 2
                J[i + 1, i + 1] += dx ** 2 * self.params['omega']

                # Add the contributions from the fifth-order polymer interaction term.
                J[i + 1, i + 1] += -5 * dx ** 2 * self.params['alpha'] * Y[i + 1] ** 4
                J[i + 1, i + 1] += dx ** 2 * self.params['alpha']

                # Add the coupling contribution from the electrostatic potential.
                J[i + 1, i + 1] += -dx ** 2 * self.params['zeta'] * Y[i]

                # Contribution from the square-root concentration
                # at the next grid point.
                J[i + 1, i + 3] = 1

                # ============================================================
                # Evaluate the residuals at the interior grid points
                # ============================================================

                # ------------------------------------------------------------
                # Residual of the electrostatic potential equation
                # ------------------------------------------------------------

                # Compute the second-order central finite-difference contribution
                # and subtract the nonlinear electrostatic term involving gamma.
                R[i] = Y[i - 2] - 2 * Y[i] + Y[i + 2] - dx ** 2 * self.params['gamma'] * np.sinh(Y[i])

                # Include the coupling between the electrostatic potential
                # and the normalised polymer concentration.
                R[i] -= dx ** 2 * self.params['delta'] * (np.exp(Y[i]) - Y[i + 1] ** 2)

                # ------------------------------------------------------------
                # Residual of the polymer concentration equation
                # ------------------------------------------------------------

                # Compute the second-order central finite-difference contribution
                # for the square root of the normalised polymer concentration.
                R[i + 1] = Y[i - 1] - 2 * Y[i + 1] + Y[i + 3]

                # Include the cubic nonlinear contribution associated with omega.
                R[i + 1] += -dx ** 2 * self.params['omega'] * (Y[i + 1] ** 3 - Y[i + 1])

                # Include the fifth-order nonlinear contribution associated with alpha.
                R[i + 1] += -dx ** 2 * self.params['alpha'] * (Y[i + 1] ** 5 - Y[i + 1])

                # Include the coupling between the electrostatic potential
                # and the square root of the normalised polymer concentration.
                R[i + 1] -= dx ** 2 * self.params['zeta'] * Y[i] * Y[i + 1]

            # Solve the sparse linear system J * dy = -R to obtain the
            # Newton–Raphson correction vector.
            # Convert the Jacobian to CSR format for efficient sparse solving.
            dy = sp.sparse.linalg.spsolve(J.tocsr(), -R)

            # Update the solution vector using the damped Newton–Raphson correction.
            # The relaxation factor controls the step size of each iteration.
            Y = Y + self.params['relaxation_factor'] * dy

            # Increment the iteration counter.
            iter += 1

        # ============================================================
        # Save and return the simulation results
        # ============================================================

        # Open the output file in binary write mode.
        f = open(self.filename, "wb")

        # Save the spatial coordinates, solution vector, and simulation
        # parameter dictionary to a pickle file.
        pickle.dump([X, Y, self.params], f)

        # Return the spatial coordinates, interleaved solution vector,
        # and simulation parameter dictionary.
        return X, Y, self.params

    def set_filename(self, name):
        """
        Set the filename used to save the simulation results.

        Parameters
        ----------
        name : str
            Name or path of the output pickle file.
        """

        # Update the output filename.
        self.filename = name

    def calculate_constants(self):
        """
        Calculate the constants used in the coupled differential equations.

        Select the characteristic length according to the normalisation
        scheme and calculate the electrostatic, polymer interaction,
        and coupling coefficients.

        The calculated constants are stored in the simulation parameter
        dictionary. The upper limit of the spatial domain is also
        normalised by the characteristic length.

        Notes
        -----
        The normalisation schemes are:
            0: D = 30
            1: D = Bjerrum length
            2: D = 1

        This method modifies self.params in place and does not return
        any values.
        """

        # ============================================================
        # Select the characteristic length for normalisation
        # ============================================================

        # Use a fixed characteristic length.
        if self.params['normalisation'] == 0:
            self.params['D'] = 30

        # Use the Bjerrum length as the characteristic length.
        elif self.params['normalisation'] == 1:
            self.params['D'] = self.params['lb']

        # Use a unit characteristic length.
        elif self.params['normalisation'] == 2:
            self.params['D'] = 1

        # ============================================================
        # Calculate the electrostatic coefficients
        # ============================================================

        # Calculate the squared inverse Debye screening length
        # associated with the bulk salt concentration.
        self.params['k2'] = 8 * np.pi * self.params['lb'] * self.params['c_salt']

        # Calculate the electrostatic coefficient associated with
        # the bulk polymer concentration and charged monomer fraction.
        self.params['km2'] = 4 * np.pi * self.params['lb'] * self.params['phib2'] * self.params['f']

        # Normalise the electrostatic coefficients using the
        # square of the characteristic length.
        self.params['gamma'] = self.params['k2'] * self.params['D'] ** 2
        self.params['delta'] = self.params['km2'] * self.params['D'] ** 2

        # ============================================================
        # Calculate the polymer interaction and coupling coefficients
        # ============================================================

        # Calculate the coefficient associated with the cubic
        # nonlinear polymer interaction term.
        self.params['omega'] = 6 * (self.params['D'] / self.params['a']) ** 2 * self.params['v'] * self.params['phib2']

        # Calculate the coupling coefficient between the electrostatic
        # potential and the square root of the normalised concentration.
        self.params['zeta'] = 6 * (self.params['D'] / self.params['a']) ** 2 * self.params['f']

        # Calculate the coefficient associated with the fifth-order
        # nonlinear polymer interaction term.
        self.params['alpha'] = 6 * (self.params['D'] / self.params['a']) ** 2 * self.params['w2'] / 2 * self.params[
            'phib2'] ** 2

        # ============================================================
        # Normalise the spatial domain
        # ============================================================

        # Divide the upper spatial boundary by the characteristic length.
        self.params['xn'] /= self.params['D']
