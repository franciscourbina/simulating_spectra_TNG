"""
This set of functions will create a grid and then sample the simulated halo in spaxel-like
fashion.
The output of each simulation is a folder with all the ray objects and a numpy array indicating 
where a ray was actually traced. 
A log file is also created to check wether the simulation was done correctly. 
"""

import coordinates as coords
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import requests
import pandas as pd
import yt
from yt.units import kpc
import os
import trident
import multiprocessing as mp

# Some simple cosmological calculations to estimate the size of the box to simulate. 
def empty_folder(path):
    dir = os.listdir(path) 
    # Checking if the list is empty or not 
    if len(dir) == 0: 
        return True
    else: 
        False   

def fit_delta_crit(x):
    # Extracted from Bryan 1998 (fit with Omega_R = 0)
    return 18*np.pi**2 + 82*x - 39*(x**2)

def virial_radius(M):
    # Mass has to be given in units of solar masses
    cosmo = FlatLambdaCDM(H0 = 67.74, Om0=0.3089, Ob0=0.0486)
    density = fit_delta_crit(cosmo.Om(1) - 1) * cosmo.critical_density(1)
    density = density.to(u.M_sun/u.kpc**3)
    r_cube = (3/(4*np.pi*density)) * (M*u.M_sun)

    return r_cube**(1/3)

def iterable_func(params, halo):
    # params = x0, y0, i, j, size_sp, N_side

    halo.create_spaxel(params[0], params[1], params[2], params[3], params[4], params[5])
    halo.spaxels_done[params[2], params[3]] = 1

def iterable_func_spectra(params, halo):
    # params = i, j, N_side, z
    halo.create_spectra_spaxel(params[0], params[1], params[2], params[3])

class Mock_halo_spectra:
    def __init__(self, id, root, lines_list):
        
        self.id = id
        self.root = root
        self.path_catalog = self.root + 'catalog_subhalos.csv'

        self.lines_list = lines_list

        catalog = pd.read_csv(self.path_catalog, index_col=0)
        
        # Loading data
        self.ds = yt.load(self.root + 'data/cutout_{}.hdf5'.format(id))
        
        # We are going to calculate the virial radius by extracting the mass of the halo
        data = catalog.loc[id]
        h = 0.6774
        mass_subhalo = data['mass']*(10**(10))/h
        self.virial_radius = virial_radius(mass_subhalo)
        
        # Extract the position of the halo
        pos_x = data['pos_x'] * self.ds.length_unit.in_units('kpc')
        pos_y = data['pos_y'] * self.ds.length_unit.in_units('kpc')
        pos_z = data['pos_z'] * self.ds.length_unit.in_units('kpc')

        self.position = [pos_x , pos_y, pos_z] 
        

    def pnw_coordinates(self, angle):
        """
        Create the new coordinate basis
        """
        angle_rad = angle * np.pi/180
        P, N, W = coords.create_PNW_base(self.ds, angle_rad, self.position)

        self.P = P
        self.N = N
        self.W = W

        self.center_PNW = coords.sim_to_pnw_coords(np.array([pos.value for pos in self.position]), self.P, self.N, self.W)

    def create_grid(self, size_sp):
        """
        Create the centers of over which a spaxel will be created.
        """

        n_points = int(self.virial_radius.value/size_sp)
        x = np.linspace(-n_points, n_points, 2*n_points+1)*size_sp + self.center_PNW[1]
        y = np.linspace(-n_points, n_points, 2*n_points+1)*size_sp + self.center_PNW[2]

        return x, y

    def create_ray(self, sp_path, x0, y0, i, j):
        name = sp_path + '/ray' + str(i) + '_' + str(j) + '.h5'
        dz = self.virial_radius.value

        start = np.array([self.center_PNW[0] - dz,  x0, y0])
        end = np.array([self.center_PNW[0] + dz, x0, y0])

        start_sim =coords.pnw_to_sim_coords(start, self.P, self.N, self.W)
        end_sim = coords.pnw_to_sim_coords(end, self.P, self.N, self.W)
        

        start_vec = [start_sim[0] * kpc, start_sim[1] * kpc , start_sim[2] * kpc]
        end_vec = [end_sim[0] * kpc, end_sim[1] * kpc , end_sim[2] * kpc]

        try:
            ray = trident.make_simple_ray(self.ds,
                        start_position=start_vec,
                        end_position=end_vec,
                        data_filename=name,
                        lines=self.lines_list, line_database='lines2.txt')
        except:
            print('No gas detected!')


    def create_spaxel(self, x0, y0, i, j, size_sp, N_side):
        # notice x0 and y0 are in the PNW system (N,W) 
        # Create folder

        os.chdir(self.path_spaxels)
        path_sp = self.path_spaxels + 'sp_{}_{}'.format(i,j)
        
        os.mkdir(path_sp)
        os.chdir(path_sp)

        x_centers = np.linspace(-size_sp/2, size_sp/2, N_side)  + x0 
        y_centers = np.linspace(-size_sp/2, size_sp/2, N_side)  + y0

        for k in range(N_side):
            for m in range(N_side):
                self.create_ray(path_sp, x_centers[k], y_centers[m], k, m)

    def run_simulation(self, angle, size_sp, N_side):
        # Angle is provided in degrees
        # Sampling is going to be N_side^2
        # First, we create the new basiss

        self.path_spaxels = self.root + '{}_spaxels_{}kpc_{}x{}_{}deg/'.format(self.id, size_sp, N_side, N_side, angle)

        if not os.path.exists(self.path_spaxels):
            os.mkdir(self.path_spaxels)

        self.pnw_coordinates(angle)
        x, y = self.create_grid(size_sp)

        self.spaxels_done = np.zeros((len(x), len(y)))

        array_params = []

        for i in range(len(x)):
            for j in range(len(y)):
                array_params.append([x[i], y[j], i, j, size_sp, N_side])

        #if __name__ == '__main__':  ## this won't run if you import the module!
        pool = mp.Pool(mp.cpu_count())
        pool.starmap(iterable_func, [(array_params[i], self) for i in range(len(array_params))])
        pool.close()
        
        print('Spaxels done: {} out of {}.'.format(np.sum(self.spaxels_done), len(x)*len(y)))

    def create_spectra_spaxel(self, i,j, N_side, z):
        sp_path = self.path_spaxels + 'sp_{}_{}'.format(i,j)
        if not empty_folder(sp_path):
            os.chdir(sp_path)
            for species in self.lines_list:
                wavelenght = int(species.split(' ')[2])    # String are "Element Ionization Lambda"
                path_species = sp_path + '/' +species.replace(' ', '_')
                os.mkdir(path_species)                
                # Now we create the spectra for each ray
                for i in range(N_side):
                    for j in range(N_side):
                        ray_name = sp_path + '/ray{}_{}.h5'.format(i,j)
                        name_spectrum = path_species + '/spectrum_{}_{}.txt'.format(i,j)
                        min_lambda = (1+z) * wavelenght - 100
                        max_lambda = (1+z) * wavelenght + 100
                        try:    # it could be the case that some rays were not generated 
                            sg = trident.SpectrumGenerator(lambda_min=min_lambda, lambda_max=max_lambda,
                                                        dlambda=0.1, line_database='lines2.txt')
                        
                            sg.make_spectrum(ray_name, lines=self.lines_list)
                            sg.save_spectrum(name_spectrum)
                        except:
                            pass
        else:
            pass

    def create_all_spectra(self, N_side, size_sp=3,  z=1):

        x, y = self.create_grid(size_sp)
        Nx, Ny = (len(x), len(y))
        
        array_params = []
        for i in range(Nx):
            for j in range(Ny):
                array_params.append([i,j, N_side, z])

        pool = mp.Pool(mp.cpu_count())
        pool.starmap(iterable_func_spectra, [(array_params[i], self) for i in range(len(array_params))])
        pool.close()

    def create_logs(self):
        """
        To be done :) 
        """
        pass