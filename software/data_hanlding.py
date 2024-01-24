import numpy as np
import requests
import pandas as pd

# globals for TNG data access
baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"6335ea41b5d1119dc38508e161c34f41"} 

def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string


def download_data(path_catalog, ids, params):
    """
    Given a subhalo ID, download its cutout.
    """
    subhalos_url = 'http://www.tng-project.org/api/TNG50-1/snapshots/50/subhalos/'
    catalog = pd.read_csv(path_catalog, index_col=0)
    
    metadata = catalog.loc[ids]

    # Download
    n = 1
    for id in ids:
        if  metadata.loc[id]['Downloaded'] == 0: 
            print('Downloading halo {} out of {}. (ID = {})'.format(n, len(ids), id))
            get(subhalos_url + str(id) + '/cutout.hdf5', params)
            n += 1

            # updating value in the catalog
            catalog['Downloaded'].loc[id] = 1
        else:
            print('Halo already downloaded. (ID = {})'.format(id))


    # Update the catalog file
    catalog.to_csv(path_catalog)

    print('Catalog updated.')