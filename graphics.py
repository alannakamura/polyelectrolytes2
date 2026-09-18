"""
Plot normalised electrostatic potential and concentration profiles
from simulation results.

This script reads multiple pickle files from a specified directory,
sorts them according to a selected simulation parameter, and generates
plots comparing the normalised electrostatic potential and normalised
concentration profiles for different parameter values.

Each pickle file is expected to contain:
    data[0]: Spatial coordinates.
    data[1]: Interleaved normalised electrostatic potential and
             square-root concentration values.
    data[2]: Dictionary containing the simulation parameters,

The script produces four figures:
    1. Normalised electrostatic potential profiles without a legend.
    2. Normalised concentration profiles without a legend.
    3. Normalised electrostatic potential profiles with a title and legend.
    4. Normalised concentration profiles with a title and legend.

The spatial coordinates are normalised by the characteristic
length 'xn'. The concentration profiles are obtained by squaring
the corresponding values stored in the simulation results.

The curves are sorted in ascending order of the selected parameter
and coloured using a gradient from red to blue. Line styles are
repeated cyclically to help distinguish individual curves.

Dependencies:
    - Python 3
    - NumPy
    - Matplotlib

Warning:
    Only load pickle files from trusted sources, as unpickling
    untrusted data can execute arbitrary code.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

# Define the directory containing the simulation results.
directory = "260902/fig6_1/"

# Retrieve the names of all files in the directory.
list_dir = os.listdir(directory)

# Define the available simulation parameters.
parameter = ['phib2', 'w2', 'xn', 'nx', 'lambda', 'f', 'v', 'y0', 's']

# Select the parameter to be varied (the fourth from the end: 'f').
parameter = parameter[-4]

# Find the position of the selected parameter in the filename.
# Assumes that filenames contain parameter names and their corresponding
# values separated by underscores.
parameter = list_dir[0].split('_').index(parameter)

# Extract the selected parameter value from each filename.
# Store the numerical value alongside its corresponding filename.
for i in range(len(list_dir)):

    # Remove the '.pkl' extension, split the filename into components,
    # and extract the value associated with the selected parameter.
    # Convert this value to a float for numerical sorting.
    list_dir[i] = list_dir[i]
    list_dir[i] = list((
        float(list_dir[i].strip('.pkl').split('_')[parameter + 1]),
        list_dir[i]
    ))

# Sort the files in ascending order of the selected parameter value.
list_dir.sort(reverse=False)

# Initialise empty lists to store the simulation data.

y_list = []  # Normalised electrostatic potential profiles.
e_list = []  # Square roots of the normalised concentration profiles.
x_list = []  # Normalised spatial coordinates.
t_list = []  # Legend labels identifying the selected parameter values.

# Iterate over the simulation files sorted by the selected parameter.
for i in list_dir:

    # Open the current pickle file in binary read mode.
    f = open(directory + i[1], 'rb')

    # Load the simulation data.
    l = pickle.load(f)

    # Normalise the spatial coordinates by the characteristic length 'xn'.
    x_list.append(l[0] / l[2]['xn'])

    # Extract the normalised electrostatic potential values
    # from the even-indexed elements.
    y_list.append(l[1][0::2])

    # Extract the square roots of the normalised concentration values
    # from the odd-indexed elements.
    e_list.append(l[1][1::2])

    # Remove the file extension from the filename.
    i[1] = i[1].split('.pk')[0]

    # Retrieve the name of the selected parameter from the filename.
    temp = str(i[1].split('_')[parameter])

    # Generate a LaTeX-formatted legend label for 'phib2'.
    if temp == 'phib2':
        t_list.append('$\\phi_b^2$'
                      + ' = ' +
                      str(i[1].split('_')[parameter + 1]))

    # Generate a standard legend label for all other parameters.
    else:
        t_list.append(str(i[1].split('_')[parameter])
                      + ' = ' +
                      str(i[1].split('_')[parameter + 1]))

    # Close the pickle file.
    f.close()

# Retrieve the filename of the first simulation file.
l = list_dir[0][1]

# Remove the file extension.
l = l.split('.pk')[0]

# Split the filename into its individual components.
l = l.split('_')

# Remove the selected parameter and its corresponding value
# to retain only the parameters common to all simulations.
l2 = l[0:parameter] + l[parameter+2:]

# Reconstruct the filename without the varying parameter
# to use it as the common title for the plots.
l = str.join('_', l2)

# Generate RGB colour components for each simulation curve.
# The colours transition linearly from red to blue.
red = np.linspace(1., 0., len(list_dir))
green = np.linspace(0., 0., len(list_dir))
blue = np.linspace(0., 1., len(list_dir))

# Define line styles to distinguish the simulation curves.
# The styles are repeated cyclically when plotting more than four curves.
linestyle2 = ['solid', 'dashed', 'dashdot', (0, (6, 2, 6, 4))]

# Create a new figure and its associated axes.
fig, ax = plt.subplots()

# Set the font size used in the figure.
size = 30

# Plot the normalised electrostatic potential profiles
# for all simulations.
for i in range(len(x_list)):
    plt.plot(
        x_list[i],
        y_list[i],
        color=(red[i], green[i], blue[i]),
        linestyle=linestyle2[i % 4]
    )

# Label the horizontal axis with the normalised position.
plt.xlabel('$x/x_n$', fontsize=size)

# Label the vertical axis with the normalised electrostatic potential.
plt.ylabel('y', fontsize=size, rotation=0, labelpad=35)

# Set the font size of the axis tick labels.
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)

# Adjust the plot margins.
plt.subplots_adjust(left=0.18, right=0.85, bottom=0.18, top=0.85)

# Display the figure.
plt.show()

# Plot the normalised concentration profiles for all simulations.
# The values stored in e_list correspond to the square roots of the
# normalised concentrations, so they are squared here before plotting.
for i in range(len(x_list)):
    plt.plot(
        x_list[i],
        e_list[i]**2,
        color=(red[i], green[i], blue[i]),
        linestyle=linestyle2[i % 4]
    )

# Label the horizontal axis with the normalised position.
plt.xlabel('$x/x_n$', fontsize=size)

# Label the vertical axis with the normalised concentration.
plt.ylabel('$\\eta^2$', fontsize=size, rotation=0, labelpad=35)

# Adjust the plot margins.
plt.subplots_adjust(left=0.18, right=0.85, bottom=0.18, top=0.85)

# Set the font size of the axis tick labels.
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)

# Display the figure.
plt.show()

# Set the font size used in the title, legend, and axis labels.
size = 20

# Add a title identifying the parameters common to all simulations.
plt.title(l, fontsize=size)

# Plot the normalised electrostatic potential profiles
# for all simulations.
for i in range(len(x_list)):
    plt.plot(
        x_list[i],
        y_list[i],
        color=(red[i], green[i], blue[i]),
        linestyle=linestyle2[i % 4]
    )

# Add a legend indicating the value of the selected parameter
# for each simulation curve.
plt.legend(t_list, fontsize=size)

# Label the horizontal axis with the normalised position.
plt.xlabel('$x/x_n$', fontsize=size)

# Label the vertical axis with the normalised electrostatic potential.
plt.ylabel('y', fontsize=size, rotation=0, labelpad=35)

# Set the font size of the axis tick labels.
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)

# Adjust the plot margins.
plt.subplots_adjust(left=0.18, right=0.85, bottom=0.18, top=0.85)

# Display the figure.
plt.show()

# Add a title identifying the parameters common to all simulations.
plt.title(l, fontsize=size)

# Plot the normalised concentration profiles for all simulations.
# The values stored in e_list correspond to the square roots of the
# normalised concentrations, so they are squared before plotting.
for i in range(len(x_list)):
    plt.plot(
        x_list[i],
        e_list[i]**2,
        color=(red[i], green[i], blue[i]),
        linestyle=linestyle2[i % 4]
    )

# Label the horizontal axis with the normalised position.
plt.xlabel('$x/x_n$', fontsize=size)

# Label the vertical axis with the normalised concentration.
plt.ylabel('$\\eta^2$', fontsize=size, rotation=0, labelpad=35)

# Add a legend indicating the value of the selected parameter
# for each simulation curve.
plt.legend(t_list, fontsize=size)

# Set the font size of the axis tick labels.
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)

# Display the figure.
plt.show()