import numpy as np

def pnw_to_sim_coords(vector_pnw, P, N, W):
    # Provided a vector in the Projection-North-West coordinate system
    # return to simulation coordinates
    # P,N and W are in the simulation coordinates

    return vector_pnw[0] * P + vector_pnw[1] * N + vector_pnw[2] * W

def dot_prod(a,b):
    return np.sum(a*b)

def sim_to_pnw_coords(vector_sim, P, N, W):
    # Provided a vector in the simulation coordinate system
    # return to projection-north-west coordinates
    # P,N and W are in the simulation coordinates

    P_comp = dot_prod(vector_sim, P)
    N_comp = dot_prod(vector_sim, N)
    W_comp = dot_prod(vector_sim, W)
    
    return np.array([P_comp, N_comp, W_comp])

def create_PNW_base(ds, angle, cm):
    # For a given angle of projection, cutout and center of mass, return
    # the Projection-North-West ortogonal base.

    cm_x, cm_y, cm_z = cm
    sp = ds.sphere((cm_x, cm_y, cm_z), (20, 'kpc'))
    L_star = sp.quantities.angular_momentum_vector(use_particles=True, particle_type='PartType4')

    L_norm = L_star.value/np.sqrt(np.sum(L_star.value*L_star.value))

    normal_vector = np.cross(L_norm, np.array([0,0,1]))
    normal_vector = normal_vector / np.sqrt(np.dot(normal_vector, normal_vector))
    

    projection = normal_vector*np.cos(angle) + np.sin(angle)*L_norm
    north = - normal_vector*np.sin(angle) + np.cos(angle)*L_norm
    west = np.cross(projection, north)

    return projection, north, west
