"""
This set of functions will compute several observable properties from the output of a simulation.
This includes EW_maps (saved as a numpy dataset), EW fracctional differences (as function of distance) and 
impact parameter profiles.
"""
import mock_spaxels as ms
import numpy as np
import pandas as pd
import os 
from scipy.constants import c

def empty_folder(path):
    dir = os.listdir(path) 
    # Checking if the list is empty or not 
    if len(dir) == 0: 
        return True
    else: 
        False   

def list_folders(path):
    dir = os.listdir(path)
    list = []
    for subfolder in dir:
        if os.path.isdir(path + '/' + subfolder):
            list.append(subfolder)
    return list

def spaxel_calc(path_sp, n_side, species):

    path_species= path_sp + "/" + species + "/"
    spectrum_sample = np.loadtxt(path_species + os.listdir(path_species)[0])
    wave = spectrum_sample[:,0]

    total_sp = np.zeros_like(wave)
    empty_spectrum = np.ones_like(wave)
    for i in range(n_side):
        for j in range(n_side):
            path_spectrum = path_species + "spectrum_{}_{}.txt".format(i,j)

            if os.path.exists(path_spectrum):
                spectrum = np.loadtxt(path_spectrum)
                total_sp += spectrum[:,2]
            else:
                total_sp += empty_spectrum
    
    return wave, total_sp/(n_side**2)

def qso_spectrum(path_sp, n_side, species):
    
    path_species= path_sp + "/" + species + "/"
    spectrum_sample = np.loadtxt(path_species + os.listdir(path_species)[0])
    wave = spectrum_sample[:,0]

    total_sp = np.zeros_like(wave)
    empty_spectrum = np.ones_like(wave)

    i = j = int(n_side/2)
    path_spectrum = path_species + "spectrum_{}_{}.txt".format(i,j)

    if os.path.exists(path_spectrum):
        spectrum = np.loadtxt(path_spectrum)
        total_sp += spectrum[:,2]
    else:
        total_sp += empty_spectrum
    
    return wave, total_sp

def data_cube(path_sim, species, dlambda=0.1, spaxel=True):

    strings = path_sim.split('_')
    deg, sampling, size, id = strings[-1], strings[-2], strings[-3], strings[-5]
    list_dir = list_folders(path_sim)
    N_sp = len(list_dir)
    N_side = int(np.sqrt(N_sp))

    N_wave = int(200/dlambda)   # THE WAVELENGTH RANGE HAS TO BE LOWER!! 
    spectra_matrix = np.zeros((N_wave, N_side, N_side))
    k = 0
    
    for folder in list_dir:
        if k%10 ==0:
            print('Folders done: {}/{}'.format(k, N_sp))

        if not empty_folder(path_sim + folder):
            name, i, j = folder.split('_')
            if spaxel:
                result = spaxel_calc(path_sim + folder, int(sampling[0]), species)
            else:
                result = qso_spectrum(path_sim + folder, int(sampling[0]), species)

            spectra_matrix[:,int(i),int(j)] = result[1]
        k += 1
    
    if spaxel:
        name_map = 'spaxels'
    else:
        name_map = 'qso'

    # In the near future, this should produce a fits file, the header will have all the information needed. 
    np.save(path_sim + '/spectra_matrix_{}_{}_{}_{}.npy'.format(id, deg, species, name_map) , spectra_matrix)
    np.save(path_sim + '/wavelength.npy', result[0])


def measure_EW(rel_flux, dx=0.1):
    new_flux = 1 - rel_flux
    return np.trapz(new_flux, dx=dx)

def compute_EW_map(spectra_matrix, dlambda=0.1, z=1):
    return np.trapz(spectra_matrix, axis=0, dx=dlambda)/(1+z)

def compute_velocity_statistics(wave, spectra_matrix, reference_wavelenght, dlambda=0.1):

    normalization = np.trapz(wave, 1-spectra_matrix, dx=dlambda, axis=0)
    c_kms = c/1000
    mean_wave = np.trapz(wave, (1-spectra_matrix)*wave[:,np.newaxis, np.newaxis], dx=dlambda, axis=0)/normalization
    
    std_wave = np.sqrt(np.trapz(wave, (1-spectra_matrix)*(wave**2)[:,np.newaxis, np.newaxis], dx=dlambda, axis=0)/normalization - mean_wave**2)

    mean_vel = c_kms * (mean_wave - reference_wavelenght)/reference_wavelenght  
    
    std_vel = c_kms * std_wave/reference_wavelenght

    return mean_vel, std_vel

def distance_matrix(matrix, center, scale):
    """
    Computes the distance from a given center for each point within a matrix. 
    Scale = distance between contiguous pixels (horizontal and vertical). 
    """
    N = matrix.shape[0]
    j_matrix = np.zeros_like(matrix) + np.arange(N)
    i_matrix = j_matrix.transpose()
    
    distance_matrix = np.sqrt((j_matrix - center[1])**2 + (i_matrix - center[0])**2) * scale

    return distance_matrix

def fracctional_diff(matrix, scale):
    N = matrix.shape[0]
    N_tot  = N**2
    
    flat_matrix = matrix.flatten()
    tot_results = int((N_tot/2)*(N_tot-1))

    results = np.zeros((tot_results, 2))

    for i in range(N_tot - 1):
        distance  = 1
        i_center = int(i/N)
        j_center = i%N

        
        distance_array = distance_matrix(matrix, (i_center, j_center), scale).flatten()
        partial_distance = distance_array[i+1:]

        partial_array = flat_matrix[i+1:]
        initial_index = int(i * (N_tot - 0.5 * (i + 1)))
        final_index = initial_index + (N_tot - 1 - i)

        results[initial_index:final_index, 0] = np.abs(partial_array - flat_matrix[i])/np.maximum(partial_array, flat_matrix[i]) 
        results[initial_index:final_index, 1] = partial_distance
    return results

def impact_parameter_profile(EW_map, center, scale):
    distances = distance_matrix(EW_map, center, scale)
    return distances.flatten(), EW_map.flatten()