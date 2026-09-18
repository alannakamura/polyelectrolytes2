"""
Configure and run a polyelectrolyte simulation.

This script initialises a polyelectrolyte system, sets the physical
and numerical parameters, and specifies the boundary conditions.

The simulation results are saved to a pickle file whose name
contains the main simulation parameters and boundary condition types.

The script also calculates the model constants, solves the coupled
nonlinear differential equations, and measures the total execution
time.

Dependencies:
    - polyelectrolytes
    - NumPy (imported through polyelectrolytes)
    - SciPy (used by the numerical solver)
"""

from polyelectrolytes import *
import time


# ============================================================
# Initialise the polyelectrolyte system
# ============================================================

# Create an instance of the polyelectrolyte solver.
p = Polyelectrolyte()

# Define the output directory for the simulation results.
filename = '260902/fig3/'

# Record the start time using a high-resolution performance counter.
inicio1 = time.perf_counter()


# ============================================================
# Configure the numerical and physical parameters
# ============================================================

# Set the number of spatial grid points to 2^12 + 1.
n = 13
p.params['nx'] = int(2**n) + 1

# Set the upper limit of the spatial domain before normalisation.
p.params['xn'] = 30*10

# Set the fraction of charged monomers.
p.params['f'] = 0.14

# Set the excluded-volume interaction parameter.
p.params['v'] = 0.1

# Set the bulk salt concentration (Å^-3).
p.params['c_salt'] = 6.02e-8

# Set the higher-order polymer interaction parameter.
p.params['w2'] = 1500

# Set the bulk polymer concentration parameter.
p.params['phib2'] = 1e-5

# Set the Newton–Raphson relaxation factor.
p.params['relaxation_factor'] = 7e-3

# Set the convergence tolerance for the correction vector.
p.params['error'] = 1e-4


# ============================================================
# Configure the boundary conditions
# ============================================================

# Specify the boundary condition types for:
# [potential at x0, square-root concentration at x0,
#  potential at xn, square-root concentration at xn].
#
# True  = Dirichlet boundary condition.
# False = Neumann boundary condition.
p.params['dirichlet_boundary'] = [False, True, False, True]

# Alternative surface potential derivative calculated from
# the prescribed surface charge density.
# p.params['y0'] = -4*np.pi*p.params['lb']*1e-4*30

# Set the prescribed potential derivative at the left boundary.
p.params['y0'] = -1.0

# Set the prescribed potential derivative at the right boundary.
p.params['yn'] = 0.0


# ============================================================
# Encode the boundary condition types in the filename
# ============================================================

# Initialise the boundary condition identifier.
t = ''

# Convert each Boolean value into its first lowercase character:
# True -> 't', False -> 'f'.
#
# For the current configuration, the identifier is 'ftft'.
for i in p.params['dirichlet_boundary']:
    t = t+str(i).lower()[0]


# ============================================================
# Construct the output filename
# ============================================================

# Append the simulation parameter names and their corresponding
# values to the output directory.
#
# The resulting filename identifies the simulation configuration
# and allows results to be sorted by individual parameters.

# Bulk polymer concentration parameter.
filename += 'phib2_'
filename += str(p.params['phib2'])

# Higher-order polymer interaction parameter.
filename += '_w2_'
filename += str(p.params['w2'])

# Upper limit of the spatial domain before normalisation.
filename += '_xn_'
filename += str(p.params['xn'])

# Number of spatial grid points.
filename += '_nx_'
filename += str(p.params['nx'])

# Newton–Raphson relaxation factor.
filename += '_lambda_'
filename += str(p.params['relaxation_factor'])

# Fraction of charged monomers.
filename += '_f_'
filename += str(p.params['f'])

# Excluded-volume interaction parameter.
filename += '_v_'
filename += str(p.params['v'])

# Bulk salt concentration.
filename += '_s_'
filename += str(p.params['c_salt'])

# Prescribed boundary value for the electrostatic potential
# or its derivative at the left boundary.
filename += '_y0_'
filename += str(p.params['y0'])

# Boundary condition identifier.
filename += '_t_'
filename += t

# Append the pickle file extension.
filename += '.pkl'


# ============================================================
# Set the output filename
# ============================================================

# Assign the constructed filename to the solver.
p.set_filename(filename)


# ============================================================
# Calculate the model constants and run the simulation
# ============================================================

# Calculate the normalisation length and the coefficients
# required by the coupled differential equations.
p.calculate_constants()

# Print the squared inverse Debye screening length.
print(p.params['k2'])

# Solve the coupled nonlinear differential equations
# and save the simulation results to the specified file.
p.run()


# ============================================================
# Measure and display the total execution time
# ============================================================

# Record the end time.
fim = time.perf_counter()

# Calculate the elapsed time in seconds.
tempo_total = fim - inicio1

# Display the total execution time to three decimal places.
print(f"tempo total: {tempo_total:.3f} s")