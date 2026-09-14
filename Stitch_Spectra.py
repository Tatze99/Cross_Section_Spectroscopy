# File created by: Martin Beyer, 2026, September 14th

import numpy as np
import os
from tkinter import Tk
from tkinter.filedialog import askdirectory
import matplotlib.pyplot as plt

def stitch_spectra(arrays, lambda_min = None, lambda_max = None):
    """
    Stitch an arbitrary number of overlapping spectra using
    cosine-weighted interpolation in the overlap regions.

    Each array must have:
        array[:, 0] = wavelength
        array[:, 1] = intensity
    """

    # ---------------------------------------------------------
    # 1. Sort spectra by starting wavelength
    # ---------------------------------------------------------
    arrays = sorted(arrays, key=lambda a: a[0, 0])

    # ---------------------------------------------------------
    # 2. Create common wavelength grid
    # ---------------------------------------------------------
    lambda_start = min(a[0, 0] for a in arrays) if lambda_min is None else lambda_min
    lambda_end   = max(a[-1, 0] for a in arrays) if lambda_max is None else lambda_max

    dl = arrays[0][1, 0] - arrays[0][0, 0]

    lambdas = np.arange(lambda_start, lambda_end + dl, dl)

    # ---------------------------------------------------------
    # 3. Interpolate all spectra onto common grid
    # ---------------------------------------------------------
    spectra = []
    coverage = []

    for array in arrays:

        x = array[:, 0]
        y = array[:, 1]

        # Where this spectrum actually exists
        mask = (lambdas >= x.min()) & (lambdas <= x.max())

        # Interpolate
        y_interp = np.interp(lambdas, x, y)

        spectra.append(y_interp)
        coverage.append(mask)

    spectra = np.asarray(spectra)
    coverage = np.asarray(coverage)

    # ---------------------------------------------------------
    # 4. Stitch spectra sequentially
    # ---------------------------------------------------------
    result = np.zeros_like(lambdas)

    # Start with the first spectrum
    result[coverage[0]] = spectra[0, coverage[0]]

    current_mask = coverage[0].copy()

    for i in range(1, len(spectra)):

        new_mask = coverage[i]

        # -----------------------------------------------------
        # Find overlap between current result and new spectrum
        # -----------------------------------------------------
        overlap = current_mask & new_mask

        # -----------------------------------------------------
        # Region where only the new spectrum exists
        # -----------------------------------------------------
        new_only = new_mask & ~current_mask

        # -----------------------------------------------------
        # Blend overlap
        # -----------------------------------------------------
        if np.any(overlap):

            indices = np.where(overlap)[0]

            length = len(indices)

            weight = 0.5 * (
                1 + np.cos(np.linspace(0, np.pi, length))
            )

            result[indices] = (
                weight * result[indices]
                + (1 - weight) * spectra[i, indices]
            )

        # -----------------------------------------------------
        # Add new-only region
        # -----------------------------------------------------
        result[new_only] = spectra[i, new_only]

        # Update mask
        current_mask |= new_mask

    return np.column_stack((lambdas, result))


if __name__ == "__main__":

    ASK_FOR_FOLDER = True

    if ASK_FOR_FOLDER:
        path = askdirectory(title='Select Folder') # shows dialog box and return the path
    else:
        path = os.path.join(os.path.dirname(__file__), "measurements", "260911_NdCaF2", "20260911_NdCaF2_Absorption")
        print(path) 

    file_name = path.split(os.sep)[-1]
    data = []

    for file in os.listdir(path):
        if file.endswith(".txt") and "Startzeiten" not in file and file_name not in file:
            print(file)
            data.append(np.loadtxt(os.path.join(path, file), skiprows=1, delimiter=","))

    final_data = stitch_spectra(data, lambda_min = 450, lambda_max = 1200)

    np.savetxt(os.path.join(path, f"{file_name}.txt"), final_data, header="IstTemp[K]=...K", fmt="%.3f,%.3e")


    # figure plotting
    plt.figure()
    plt.plot(final_data[:,0], final_data[:,1], label="Stitched")

    for i, array in enumerate(data):
        plt.plot(array[:,0], array[:,1], label=f"File {i+1}", lw=0.5)

        