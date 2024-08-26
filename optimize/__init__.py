#MAIN FILE CONTAINING OPTIMIZATION FUNCTIONS

import numpy as np
import pygimli as pg
import pygimli.physics.ert as ert
import matplotlib.pyplot as plt
import pandas as pd
from numba import njit
import scipy.linalg as lin
import sys
import csv
import time

def init_model(scheme, mesh, irm=False, complex=False):
    '''This function initializes the ERTModelling object

    Parameters:
    ----------
    scheme: 
        DataContainerERT object containing the electrode positions and configurations
    mesh: 
        pygimli mesh object
    irm:
        if True, the region manager is ignored. TODO: WHEN shoud this be done?
    complex: 
        if True, assume complex model. NOT IMPLEMENTED YET

    Returns:
    -------
    fop: 
        ERTModelling object
    '''
    
    fop = ert.ERTModelling(verbose=True, sr=True)
    if complex == True:
        fop.setComplex(True)
    fop.setData(scheme)
    if irm == True:
        fop.setMesh(mesh, ignoreRegionManager=True)
    else:
        fop.setMesh(mesh, ignoreRegionManager=False)
    fop.mesh()
    return fop

def calc_jacobian(fop, rho):
    '''This function calculates the Jacobian matrix for a given model

    Parameters:
    ----------
    fop:
        ERTModelling object
    rho:
        resistivity model

    Returns:
    -------
    J:
        Jacobian matrix
    '''

    fop.createJacobian(rho)
    J = np.array(fop.jacobian())
    return J

def calc_jacobian_weighted(fop, rho, mesh, rhoa_noise, cellweight=False):
    '''This function calculates the Jacobian matrix for a given model, weighted by cell size and data errors

    Parameters:
    ----------
    fop:
        ERTModelling object
    rho:
        resistivity model
    mesh:
        pygimli mesh object
    rhoa_noise:
        array of data errors
    cellweight:
        if True, the Jacobian is weighted by cell size

    Returns:
    -------
    J:
        Jacobian matrix
    '''

    fop.createJacobian(rho)
    J = np.array(fop.jacobian())

    if cellweight == True:
        # add weighting by cell size
        meshsize = np.array(pg.Vector(mesh.cellSizes()))
        sizefactor = 1/meshsize
        sizefactor = np.tile(sizefactor, (len(J),1))

        J = sizefactor*J

    if isinstance(rhoa_noise, np.ndarray) == False:
        print(J.shape)
        return J

    #if errors are all zero, return J
    elif not rhoa_noise.any() == True:
        print(J.shape)
        return J
    else:
        #add data weighting
        #convert to only positive values
        rhoa_noise = np.abs(rhoa_noise)
        Wd = 1/rhoa_noise
        J_weighted = Wd[:, np.newaxis]*J
        print(J_weighted.shape)
        return J_weighted


def interpolate_J(J_fine, mesh_fine, mesh_coarse):
    '''This function interpolates the sensitivities of a fine mesh to a coarse mesh

    Parameters:
    ----------
    J_fine:
        Jacobian matrix of fine mesh
    mesh_fine:
        pygimli mesh object of fine mesh
    mesh_coarse:
        pygimli mesh object of coarse mesh
    
    Returns:
    -------
    J_interpolated:
        Jacobian matrix of fine mesh interpolated to coarse mesh
    '''
    cell_ids_fine = []

    #find cell ids of fine mesh
    for cell in mesh_fine.cells():
        cell_ids_fine.append(cell.id())    
    
    corresponding_id = []

    #find ids of coarse mesh that correspond to the fine mesh ids
    for cell_pos in mesh_fine.cellCenters():
        cell = mesh_coarse.findCell(cell_pos)
        corresponding_id.append(cell.id())

    fine_ids = np.array(cell_ids_fine)
    coarse_ids = np.array(corresponding_id)
    id_map = np.column_stack((fine_ids, coarse_ids))

    #add Sensitivities of fine mesh to id map
    data = np.concatenate((id_map, J_fine.T), axis=1)
    # create dataframe to use groupby function
    df = pd.DataFrame(data)

    data_interpolated = df.groupby(1).sum()
    data_interpolated = data_interpolated.iloc[:, 1:]

    #back to array
    J_interpolated_T = data_interpolated.to_numpy()
    #divide by the number of fine grid cells each coarse grid cell contains
    J_interpolated = J_interpolated_T.T#/(mesh_fine.cellCount()/mesh_coarse.cellCount())

    return J_interpolated


def create_constraints(fop, order):
    '''This function creates constraint matrix for R-matrix calculation

    Parameters:
    ----------
    fop:
        ERTModelling object
    order:
        order of the constraints

    Returns:
    -------
    WTW:
        constraint matrix

    '''
    rm = fop.regionManager()
    rm.setConstraintType(order)
    fop.createConstraints()
    WTW = fop.constraints()
    WTW = pg.utils.sparseMatrix2coo(WTW)
    WTW = WTW.toarray()
    #IF SOMETHING IS WRONG WITH THE CONSTRAINTS, CHANGE THE RETURN OUTPUT TO WM
    #WTW = Wm
    #WTW = np.dot(Wm.T,Wm)
    return WTW

def calc_R(J,WTW, lam=0.5):
    '''Calculates R-matrix for a given Jacobian matrix and constraint matrix

    Parameters:
    ----------
    J:
        Jacobian matrix
    WTW:
        constraint matrix
    lam:
        regularization parameter

    Returns:
    -------
    R:
        R-matrix
    '''
    JTJ = J.T.dot(J)
    RT = JTJ + lam * WTW
    R = lin.solve(RT, JTJ, assume_a='sym')
    return R

def get_measurement(J_c, index):
    '''This function returns the sensitivity distribution of a single measurement in the jacobian matrix

    Parameters:
    ----------
    J_c:
        Jacobian matrix
    index:
        index of the measurement in J_c

    Returns:
    -------
    sensitivity:
        sensitivity distribution of the measurement
    index:
        index of the measurement
    '''
    # get single sensitivity distribution from index
    sensitivity = J_c[index, :]
    sensitivity = np.resize(sensitivity, (len(sensitivity), 1))
    return sensitivity, index


def get_measurements(indices, J_c):
    '''This function returns the sensitivity distributions of a set of measurements from the jacobian matrix

    Parameters:
    ----------
    J_c:
        Jacobian matrix of comprehensive dataset
    indices:
        indices of the measurements in J_c
    
    Returns:
    -------
    sensitivities:
        sensitivity distributions of the measurements
    
    '''
    if type(indices) == np.ndarray:
        sensitivities = J_c[indices]
    else:
        sensitivities = J_c[np.array(indices)]
        sensitivities = np.resize(sensitivities, (1, len(sensitivities)))
    return sensitivities


@njit(parallel=True, fastmath=True)   
def calc_Z_Y(JTJ_base, lam, WTW,G):
    '''This function calculates matrices needed for the Sherman-Morrison rank 1 update

    Parameters:
    ----------
    JTJ_base:
        J.T x J matrix
    lam:
        regularization parameter
    WTW:
        constraint matrix
    G:
        sensitivity distributions of added measurements
    
    Returns:
    -------
    Z:
        Z-matrix, to be used in calc_dr
    Y:
        Y-matrix, to be used in calc_dr
    value_below:
        value below the diagonal of the GZ matrix, to be used in calc_dr
    '''

    Z = np.linalg.solve((JTJ_base + lam * WTW),G.T)
    #Z = np.dot(np.linalg.inv((JTJ_base + lam * WTW)),G.T)
    Y = np.dot(JTJ_base, Z)

    #gz = np.dot(G, Z)
    #diag = np.diag(gz)

    #in one line
    diag = (np.multiply(G, Z.T)).sum(-1)
    value_below = diag+1
    return Z,Y,value_below

@njit(parallel=True, fastmath=True)
def calc_dr(value_below, Z, G, Y):
    '''This function uses previously determined matrices to calculate the rank 1 update of the diagonal of the R-matrix,
    given a set of added measurements.

    Parameters:
    ----------
    value_below:
        value below the diagonal of the GZ matrix
    Z:
        Z-matrix
    G:
        sensitivity distributions of added measurements
    Y:
        Y-matrix
    
    Returns:
    -------
    dR:
        rank 1 update of the R-matrix. 
        The rows of the matrix contain the rank 1 updates of the resolution matrix
        when the corresponding measurements from G are added, respectively.
    '''
    left = Z/value_below
    #matrix with diags of Rb1, Rb2, Rb3... as rows
    dR = np.multiply(left.T, G-Y.T)
    return dR

def calculation_morrison(JTJ_base, lam, WTW,G):
    '''This function calculates the rank 1 updates of the diagonal of the R-matrix, using functions calc_Z_Y and calc_dr.

    Parameters:
    ----------
    JTJ_base:
        J.T x J matrix
    lam:
        regularization parameter
    WTW:
        constraint matrix
    G:
        sensitivity distributions of added measurements
    
    Returns:
    -------
    dR:
        rank 1 updates of the R-matrix, stored as rows in dR.
    '''

    Z,Y,value_below = calc_Z_Y(JTJ_base, lam, WTW,G)
    value_below = np.tile(value_below, (len(Z), 1))

    dR = calc_dr(value_below, Z, G, Y)
    return dR   

def calc_morrison_diag(JTJ_base, J_c, lam, indices, WTW):
    '''This function calculates the rank 1 updates of the diagonal of the R-matrix, using the Morrison rank 1 update.

    Parameters:
    ----------
    JTJ_base:
        J.T x J matrix
    J_c:
        Jacobian matrix of comprehensive dataset
    lam:
        regularization parameter
    indices:
        indices of the measurements in J_c
    WTW:
        constraint matrix
    
    Returns:
    -------
    dR:
        rank 1 updates of the R-matrix, stored as rows in dR.
    '''
    G = get_measurements(indices, J_c)

    dR = calculation_morrison(JTJ_base, lam, WTW,G)
    return dR

def check_rank(mesh, dR, R_c, weights):
    '''This function calculates the ranking function Fcr for a set of rank 1 updates of the R-matrix.

    Parameters:
    ----------
    mesh:
        pygimli mesh object
    dR:
        rank 1 updates of the R-matrix
    R_c:
        R-matrix of comprehensive dataset
    weights:
        cell weights
    
    Returns:
    -------
    Fcr:
        ranking results for each R-update in dR
    '''
    diag_Rc = np.einsum('...ii -> ...i', R_c)
    weights_stacked = np.tile(weights, (len(dR), 1))
    
    top = np.multiply(weights_stacked, dR)

    Fcr = 1/(len(mesh.cells()))*np.sum(np.divide(top, diag_Rc), axis=1)
    return Fcr

def set_cellweight_area(mesh, xlim, ylim):
    '''This function sets cell weights for a given area in the mesh.

    Parameters:
    ----------
    mesh:
        pygimli mesh object
    xlim:
        x-limits of the area
    ylim:
        y-limits of the area
    
    Returns:
    -------
    weights:
        cell weights
    '''
    # set cell weights outside of a certain area to 10^-12
    weights = []
    for cell in mesh.cells():
        x = cell.center()[0]
        y = cell.center()[1]
        if (y > ylim[0] and y < ylim[1] and x > xlim[0] and x < xlim[1]):
            weights.append(1)
        else:
            weights.append(10**(-12))
    return np.array(weights)

#@jit(parallel=True, fastmath=True)
def average_rel_resolution(R, R_c, weights):
    '''This function calculates the average relative resolution, compared to the comprehensive set, of a set of measurements.

    Parameters:
    ----------
    R:
        R-matrix of the current set
    R_c:
        R-matrix of the comprehensive dataset
    weights:
        cell weights

    Returns:
    -------
    S:
        average relative resolution of current set
    '''
    diag_entries = np.diag(R)
    diag_entries_c = np.diag(R_c)

    weights[weights < 0.01] = 0

    diag_weighted = diag_entries*weights
    diag_weighted = diag_weighted[diag_weighted != 0]
    diag_weighted_c = diag_entries_c*weights
    diag_weighted_c = diag_weighted_c[diag_weighted_c != 0]

    S = 1/(len(diag_weighted))*(diag_weighted/diag_weighted_c).sum()
    return S


def check_ortho(fcr_selected, threshhold, glist):
    '''This function checks the orthogonality of a set of measurements,
    and adds them to the list of added measurements if they are sufficiently orthogonal.

    Parameters:
    ----------
    fcr_selected:
        measurement to be added
    threshhold:
        orthogonality threshhold

    Returns:
    -------
    glist:
        list of sensitivities with highest rank, according to fcr_selected.
    '''

    #if there are no added configurations already, add the index of this one
    if not glist:
        glist.append(fcr_selected)
    # if there are already added configurations, add the next one if it fulfills the orthogonality check
    else:
        g = fcr_selected[2:]
        g_added = np.array(glist)[:,2:]
        orthlist = []
        for sens in g_added:
            a = (np.linalg.norm(np.dot(sens.T,g)))
            b = (np.linalg.norm(sens)*np.linalg.norm(g))
            orth = a/b
            orthlist.append(orth)
        if all(i < threshhold for i in orthlist) == True:
            glist.append(fcr_selected)
    return glist

def search_J_from_comp(J_comp_coarse, scheme_comprehensive, scheme):
    '''This function searches the sensitivities of a set of measurements in the comprehensive dataset to avoid recalculation.
    
    Parameters:
    ----------
    J_comp_coarse:
        sensitivities of the comprehensive dataset
    scheme_comprehensive:
        comprehensive dataset
    scheme:
        set of measurements with unknown sensitivities

    Returns:
    -------
    J_new:
        sensitivities of the set of measurements
    '''
    abmn_comp = np.asarray([scheme_comprehensive['a'], scheme_comprehensive['b'], scheme_comprehensive['m'], scheme_comprehensive['n'], scheme_comprehensive['k'], scheme_comprehensive['err']])
    abmn_comp = abmn_comp.T

    abmn = np.asarray([scheme['a'], scheme['b'], scheme['m'], scheme['n'], scheme['k'], scheme['err']])
    abmn = abmn.T        
    abmnJ_comp = np.concatenate((abmn_comp, J_comp_coarse), axis=1)

    matching_indices = []
    for row in abmn[:,:4]:
        index = (abmnJ_comp[:,0:4] == row).all(axis=1).nonzero()
        matching_indices.append(index[0][0])

    abmnJ_new = abmnJ_comp[matching_indices]

    if len(abmnJ_new) != len(abmn):
        sys.exit('ERROR: new sensitivity array does not match length of scheme')
    if abmnJ_new.ndim == 1:
        J_new = abmnJ_new[6:]
    else:
        J_new = abmnJ_new[:,6:]

    return J_new

def search_errors_from_comp(scheme_comprehensive, scheme, pos):
    '''This function searches the measurement errors of a set of configurations in the comprehensive dataset to avoid recalculation.

    Parameters:
    ----------
    scheme_comprehensive:
        comprehensive dataset
    scheme:
        set of measurements
    pos:
        electrode positions

    Returns:
    -------
    scheme:
        set of measurements, now with corrseponding errors
    '''
    abmn_comp = np.asarray([scheme_comprehensive['a'], scheme_comprehensive['b'], scheme_comprehensive['m'], scheme_comprehensive['n'], scheme_comprehensive['k'], scheme_comprehensive['err']])
    abmn_comp = abmn_comp.T

    abmn = np.asarray([scheme['a'], scheme['b'], scheme['m'], scheme['n'], scheme['k'], scheme['err']])
    abmn = abmn.T    

    matching_indices = []
    for row in abmn[:,:4]:
        index = (abmn_comp[:,0:4] == row).all(axis=1).nonzero()
        matching_indices.append(index[0][0])

    abmn_new = abmn_comp[matching_indices]

    if len(abmn_new) != len(abmn):
        sys.exit('repair search error from comp function')   

    abmn = abmn_new[:, 0:4]
 
    k = abmn_new[:, 4]
    err = abmn_new[:, 5]
    scheme = pg.DataContainerERT()
    scheme.setSensorPositions(pos)
    scheme.resize(len(abmn))
    for i, j in enumerate("abmn"):
        scheme.set(j, abmn[:, i])
    scheme.set("valid", np.ones(len(abmn)))
    scheme["k"] = k
    scheme['err'] = err

    return scheme

#@jit(parallel=True, fastmath=True)
def update_R(abmnk, scheme, scheme_comprehensive, pos, J_comp_coarse, lam, WTW):
    '''This function updates the R-matrix after adding a set of measurements to the scheme.

    Parameters:
    ----------
    abmnk:
        set of measurements to be added
    scheme:
        current scheme
    scheme_comprehensive:
        comprehensive dataset
    pos:
        electrode positions
    J_comp_coarse:
        sensitivities of the comprehensive dataset
    lam:
        regularization parameter
    WTW:
        constraint matrix

    Returns:
    -------
    scheme:
        updated scheme
    J:
        updated sensitivities
    R:
        updated R-matrix
    '''
    #add new configs to scheme
    abmnlist = np.asarray([scheme['a'], scheme['b'], scheme['m'], scheme['n'], scheme['k'], scheme['err']])
    abmnlist = abmnlist.T
    combined = np.concatenate([abmnlist, abmnk])
    abmn = combined[:, 0:4]
    k = combined[:, 4]
    err = combined[:, 5]
    scheme = pg.DataContainerERT()
    scheme.setSensorPositions(pos)
    scheme.resize(len(abmn))
    for i, j in enumerate("abmn"):
        scheme.set(j, abmn[:, i])
    scheme.set("valid", np.ones(len(abmn)))
    scheme["k"] = k
    scheme['err'] = err

    J = search_J_from_comp(J_comp_coarse, scheme_comprehensive, scheme)
    R = calc_R(J, WTW, lam)

    return scheme, J, R

def run_optimization(settings,mesh, pos, J, J_c, R, R_c, scheme, scheme_comprehensive, WTW):
    '''This function runs the optimization algorithm
    
    Parameters:
    ----------
    settings:
        dictionary containing settings
    mesh:
        pygimli mesh object
    pos:
        electrode positions
    J:
        sensitivities of the current scheme
    J_c:
        sensitivities of the comprehensive dataset
    R:
        R-matrix of the current scheme
    R_c:
        R-matrix of the comprehensive dataset
    scheme:
        current scheme
    scheme_comprehensive:
        comprehensive dataset
    WTW:
        constraint matrix
        
    Returns:
    -------
    scheme:
        updated scheme
    R:
        updated R-matrix
    res_set:
        list of average relative resolutions
    nconfigs:
        list of number of configurations
    '''

    weights = set_cellweight_area(mesh, settings['weighting_x'], settings['weighting_y'])
    res_set = []
    nconfigs = []
    while len(scheme['k']) < settings['nr_configs']:
        abmnk = one_step(settings, mesh, scheme, scheme_comprehensive, J_c, J, R, R_c, weights, res_set, WTW)
        scheme, J, R = update_R(abmnk, scheme,scheme_comprehensive, pos,J_c, settings['lambda'], WTW)
        nconfigs.append(len(scheme['k']))
    if len(scheme['k']) > settings['nr_configs']:
        abmnk = []
        for i in range(0,settings['nr_configs']):
            a = scheme['a'][int(i)]
            b = scheme['b'][int(i)]
            m = scheme['m'][int(i)]
            n = scheme['n'][int(i)]
            k = scheme['k'][int(i)]
            err = scheme['err'][int(i)]
            abmnk.append([a,b,m,n,k,err])
        abmnk = np.array(abmnk)
        abmn = abmnk[:, 0:4]
        k = abmnk[:, 4]
        err = abmnk[:, 5]
        scheme = pg.DataContainerERT()
        scheme.setSensorPositions(pos)
        scheme.resize(len(abmn))
        for i, j in enumerate("abmn"):
            scheme.set(j, abmn[:, i])
        scheme.set("valid", np.ones(len(abmn)))
        scheme["k"] = k
        scheme['err'] = err
        nconfigs.append(len(scheme['k']))

        J = search_J_from_comp(J_c, scheme_comprehensive, scheme)
        R = calc_R(J, WTW, settings['lambda'])
        res_set.append(average_rel_resolution(R, R_c, weights))

        return scheme, R, res_set, nconfigs

    else:
        return scheme, R, res_set, nconfigs


def init_scheme(pos, config, name, K_limit, mesh, numerical_K = False, plot=True, R_log = False):
    '''This function initializes a DataContainerERT object with a set of measurements

    Parameters:
    ----------
    pos:
        electrode positions
    config:
        set of measurements
    name:
        name of the scheme
    K_limit:
        K-factor limit
    mesh:
        pygimli mesh object
    numerical_K:
        if True, calculate K-factors numerically
    plot:
        if True, plot the K-factor distribution
    R_log:
        if True, use R instead of rhoa in Jacobian

    Returns:
    -------
    scheme:
        DataContainerERT object
    '''
    scheme = pg.DataContainerERT()
    scheme.setSensorPositions(pos)
    scheme.resize(len(config))
    for i, j in enumerate("abmn"):
        scheme.set(j, config[:, i])

    scheme.set("valid", np.ones(len(config)))
    #switch potential electrodes to remove possible negative K-Factors
    scheme['k'] = ert.createGeometricFactors(scheme, numerical=numerical_K, mesh=mesh)

    indices = np.where(scheme['k']<0)[0]
    indices_remaining = np.where(scheme['k']>=0)[0]
    m = np.array(scheme['m'])[indices]
    n = np.array(scheme['n'])[indices]
    m_new = n
    n_new = m
    m_remaining = np.array(scheme['m'])[indices_remaining]
    n_remaining = np.array(scheme['n'])[indices_remaining]

    new = np.column_stack((indices, m_new, n_new))
    remaining = np.column_stack((indices_remaining, m_remaining, n_remaining))

    combined = np.concatenate((new, remaining), axis=0)
    sorted_array = combined[combined[:,0].argsort()]

    scheme['m'] = sorted_array[:,1]
    scheme['n'] = sorted_array[:,2]

    scheme['k'] = ert.createGeometricFactors(scheme, numerical=numerical_K, mesh=mesh)
    #remove K-factors above set limit
    scheme.remove(scheme['k'] > K_limit)

    #set K=1 if jacobi should be resistance instead of rhoa
    if R_log == True:
        scheme['k'] = np.full_like(scheme['k'],1)

    print("Final length of {} ".format(name) + "is {}".format(len(scheme['k'])))
    if plot == True:
        fig, ax = plt.subplots()
        ax.hist(np.array(scheme['k']))
        ax.set_title("Geometric factors of {}".format(name))
        ax.set_xlabel('K')

    return scheme

def create_data(mesh, scheme, rho, err_R_rel, err_R_abs, seed):
    '''This function creates synthetic data for a given set of measurements

    Parameters:
    ----------
    mesh:
        pygimli mesh object
    scheme:
        set of measurements
    rho:
        resistivity model
    err_R_rel:
        relative resistance error
    err_R_abs:
        absolute resistance error
    seed:
        random seed

    Returns:
    -------
    rhoa_noise:
        error of apparent resistivities
    '''
    np.random.seed(seed)
    data = ert.simulate(mesh, scheme, rho, verbose=True)
    rhoa = data['rhoa'].array()
    K = np.abs(scheme['k'].array())
    R = rhoa/K
    R_noise = np.random.normal(
        loc=0,
        scale=err_R_rel/ 100 * R + err_R_abs
    )
    rhoa_noise = R_noise*K

    return rhoa_noise

def create_error_weights(mesh, scheme, rho, err_R_rel, err_R_abs):
    '''This function creates error weights for a given set of measurements

    Parameters:
    ----------
    mesh:
        pygimli mesh object
    scheme:
        set of measurements
    rho:
        resistivity model
    err_R_rel:
        relative resistance error
    err_R_abs:
        absolute resistance error

    Returns:
    -------
    R:
        resistance
    rhoa:
        apparent resistivity
    R_noise:
        resistance errors
    rhoa_noise:
        apparent resistivity errors
    '''
    data = ert.simulate(mesh, scheme, rho, verbose=True)
    rhoa = data['rhoa'].array()
    K = np.abs(scheme['k'].array())
    R = rhoa/K
    R_noise = (err_R_rel/100)*R+np.full(R.shape, err_R_abs)
    rhoa_noise = R_noise*K
    return R, rhoa, R_noise, rhoa_noise

def reduce_electrode_polarization():
    print("not implemented yet!")

def reduce_to_usable_configs(scheme_comprehensive, min_count):
    '''
    This function reduces the comprehensive dataset to only include configurations that have a minimum number of shared injections (useful for multichannel optimization).
    
    Parameters:
    ----------
    scheme_comprehensive:
        comprehensive dataset
    min_count:
        minimum number of shared injections
    Returns:
    -------
    config:
        reduced set of configurations
    '''
    data = np.stack([scheme_comprehensive['a'],scheme_comprehensive['b'],scheme_comprehensive['m'],scheme_comprehensive['n']], axis=1)
    data = pd.DataFrame(data, columns=['a','b','m','n'])
    data_grouped = pd.DataFrame({'count' : data.groupby(['a', 'b']).size()}).reset_index()
    sorted = data_grouped.sort_values('count', ascending=False)
    cut = sorted[sorted['count']>min_count]
    data_reduced = data[data[['a','b']].apply(tuple, axis=1).isin(cut[['a','b']].apply(tuple, axis=1))]
    config = np.stack([data_reduced['a'],data_reduced['b'],data_reduced['m'],data_reduced['n']], axis=1)
    return config

def easy_optimize_final(settings, scheme_start, scheme_comprehensive, rho, rhoa_comp, rhoa_noise_comp,pos,mesh_coarse, mesh_fine):
    '''
    This function runs the optimization algorithm with the given settings, schemes, grid(s), resistivity model, computed data, and noise levels.

    Parameters:
    ----------
    settings:
        dictionary containing settings
    scheme_start:
        start configuration
    scheme_comprehensive:
        comprehensive configuration
    rho:
        resistivity model
    rhoa_comp:
        apparent resistivities of the comprehensive configuration
    rhoa_noise_comp:
        apparent resistivity errors of the comprehensive configuration
    pos:
        electrode positions
    mesh_coarse:
        coarse mesh
    mesh_fine:
        fine mesh

    Returns:
    -------
    None
    '''

    #scheme_comprehensive = init_scheme(pos,comprehensive_config, "comprehensive config", settings["K_limit"])
    #scheme_start = init_scheme(pos, start_config, "start_config", settings['K_limit'])

    if settings['logparams'] == True:
        rhoa_noise_comp = rhoa_noise_comp/np.array(rhoa_comp)
    #add rhoa errors to scheme
    scheme_comprehensive['err'] = rhoa_noise_comp
    scheme_start = search_errors_from_comp(scheme_comprehensive, scheme_start, pos)

    fop_c = init_model(scheme_comprehensive, mesh_fine, irm=False)

    #calculate Jacobi matrix and R for comprehensive scheme; if logarithmic, scale by rho
    if settings['logparams']==True:
        J_c = calc_jacobian_weighted(fop_c, rho, mesh_fine, rhoa_noise_comp, settings['jacobi_cellweight'])

        x = np.tile(rho,(len(np.array(rhoa_comp)),1))
        y = np.tile(np.array(rhoa_comp), (len(rho),1))
        J_c = (x/y.T)*J_c
    else:
        J_c = calc_jacobian_weighted(fop_c, rho, mesh_fine, rhoa_noise_comp, settings['jacobi_cellweight'])

    fop_c = init_model(scheme_comprehensive, mesh_coarse, irm=False)
    #calculate constraints
    #second order
    WTW = create_constraints(fop_c, 2)
    #interpolate to coarser grid
    if mesh_fine != mesh_coarse:
        J_c_coarse = interpolate_J(J_c, mesh_fine, mesh_coarse)
    else:
        J_c_coarse = J_c
    R_c_coarse = calc_R(J_c_coarse, WTW, settings['lambda'])

    # Get sensitivites for Starting-Scheme and calculate R (sens is already weighted by errors)
    fop_start_coarse = init_model(scheme_start, mesh_coarse, irm=False)
    J_start_coarse = search_J_from_comp(J_c_coarse, scheme_comprehensive, scheme_start)
    R_start_coarse = calc_R(J_start_coarse, WTW, settings['lambda'])

    t = time.time()
    scheme_optimized, R_optimized, res_set, nconfigs = run_optimization(settings, mesh_coarse, pos, J_start_coarse, J_c_coarse, R_start_coarse, R_c_coarse, scheme_start, scheme_comprehensive, WTW)
    elapsed_time = time.time() - t
    print("Elapsed time: ",elapsed_time)

    df_opt = pd.DataFrame(list(zip(scheme_optimized['a'], scheme_optimized['b'], scheme_optimized['m'], scheme_optimized['n'])), 
                columns =['a', 'b', 'm', 'n']) 
    count_opt = len(df_opt.drop_duplicates(['a','b']).index)
    print("Number of unique current injections: ",count_opt)

    iteration = np.linspace(1,len(res_set),len(res_set))
    data = np.stack([iteration, res_set,nconfigs], axis=1)
    df_evo = pd.DataFrame(data,columns=['iteration','resolution','confignr'])
    df_evo.to_csv('optimizations/{}/{}_{}_{}/evolution_opt.csv'.format(settings['foldername'],settings['label'],settings['err_rel'],settings['err_abs']),sep=',', index=False)

    settings['elapsed_time'] = elapsed_time
    w = csv.writer(open("optimizations/{}/{}_{}_{}/settings.csv".format(settings['foldername'],settings['label'],settings['err_rel'],settings['err_abs']), "w"))
    for key, val in settings.items():
        w.writerow([key, val])
    opt_scheme = df_opt.to_numpy()
    np.savetxt('optimizations/{}/{}_{}_{}/opt_scheme.dat'.format(settings['foldername'],settings['label'],settings['err_rel'],settings['err_abs']), opt_scheme+1, fmt='%i')

def one_step(settings, mesh, scheme, scheme_comprehensive, J_c, J, R, R_c, weights, res_set, WTW):
    '''This function performs one step of the optimization algorithm
    
    Parameters:
    ----------
    settings:
        dictionary containing settings
    mesh:
        pygimli mesh object
    scheme:
        current scheme
    scheme_comprehensive:
        comprehensive dataset
    J_c:
        sensitivities of the comprehensive dataset
    J:
        sensitivities of the current scheme
    R:
        R-matrix of the current scheme
    R_c:
        R-matrix of the comprehensive dataset
    weights:
        cell weights, used to focus on certain areas of the mesh
    res_set:
        list of average relative resolutions in each iteration
    WTW:
        constraint matrix

    Returns:
    -------
    chosen_configs:
        set of measurements to be added to the scheme

    '''

    stepsize = int(np.ceil((settings['stepsize']/100)*len(scheme['k']))) # % of base set
    print('Stepsize set to {} , which equals {} % of the base set'.format(stepsize, settings['stepsize']))

    #calculate weighted relative resolution
    rel_resolution = average_rel_resolution(R, R_c, weights)
    res_set.append(rel_resolution)
    print("Average relative model resolution of current base set: ", rel_resolution)

    # Jacobi Matrix of comprehensive dataset holds
    # all sensitivities for possible configurations
    number_of_rows = J_c.shape[0]

    #select all configs from comprehensive set
    indices = np.linspace(0,number_of_rows-1,number_of_rows, dtype=int)


    #caluclate dR and Fcr for all measurements in parallel
    dR = calc_morrison_diag(J.T.dot(J),J_c, settings['lambda'], indices, WTW)
    Fcr = check_rank(mesh, dR, R_c, weights)

    print("dr and Fcr calc done")

    abmnlist = np.asarray([scheme_comprehensive['a'], scheme_comprehensive['b'], scheme_comprehensive['m'], scheme_comprehensive['n'], scheme_comprehensive['k'], scheme_comprehensive['err']])
    abmnlist = abmnlist.T

    #sort indices after rank
    fcr_indices = np.asarray(np.hstack((Fcr.reshape((len(Fcr), 1)), indices.reshape((len(indices),1)),J_c)))
    fcr_sorted = fcr_indices[fcr_indices[:,0].argsort(axis=0)]
    fcr_sorted = fcr_sorted[::-1]

    indices_sorted = fcr_sorted[:,1].astype(int)

    print("sorting done")
    
    abmn_current_scheme = np.asarray([scheme['a'], scheme['b'], scheme['m'], scheme['n'], scheme['k'], scheme['err']])
    abmn_current_scheme = abmn_current_scheme.T

    df = pd.DataFrame(list(zip(scheme['a'], scheme['b'], scheme['m'], scheme['n'])), 
                   columns =['a', 'b', 'm', 'n']) 

    unique_injections = len(df.drop_duplicates(['a','b']).index)

    print("counting unique injections done", unique_injections)

    def test_config(settings, config, glist, abmn_current_scheme, fcr_sorted, threshhold, unique_injections, k):

        '''
        This function tests a configuration to be added to the current scheme.
        If the configuration is not already in the scheme, it is checked for orthogonality.
        If the orthogonality check is passed, the configuration is added to the scheme.
        If the orthogonality check is not passed, the next configuration is tested.

        Parameters:
        ----------
        settings:
            dictionary containing settings
        config:
            configuration to be tested
        glist:
            list of sensitivities with highest rank
        abmn_current_scheme:
            current scheme
        fcr_sorted:
            sorted list of configuration ranks
        threshhold:
            orthogonality threshhold
        unique_injections:
            number of unique injections in the current scheme
        k:
            index of the configuration in the sorted list

        Returns:
        -------
        glist:
            updated list of sensitivities with highest rank
        k:
            index of the configuration in the sorted list
        unique_injections:
            updated number of unique injections in the current scheme

        '''

        #SINGLE CHANNEL OPTIMIZATION
        if settings['type'] == 'SC':
            #check if config is already in current scheme
            if (config == abmn_current_scheme).all(1).any() == False:
                glist_new = check_ortho(fcr_sorted[k], threshhold, glist)
                if len(glist_new) > len(glist):
                    print("Orthogonality check passed, adding config to scheme")
                    glist = glist_new
                    k = k+1
                    return glist, k, unique_injections
                else:
                    k = k+1
                    return glist, k, unique_injections
            #if it is, go next
            else:
                k = k+1
                return glist, k, unique_injections

        #MULTICHANNEL OPTIMIZATION
        else:
            #check if config is already in current scheme
            if (config == abmn_current_scheme).all(1).any() == False:
                #if injections are already used, check orthogonality and add if passed.
                if (config[0:2] == abmn_current_scheme[:, 0:2]).all(1).any() == True:
                    glist_new = check_ortho(fcr_sorted[k], threshhold, glist)
                    if len(glist_new) > len(glist):
                        print("Adding config with existing injections to scheme")
                        glist = glist_new
                        k = k+1
                        return glist, k, unique_injections
                    else:
                        k = k+1
                        return glist, k, unique_injections
                #else if unique injections are still smaller than the maximum allowed injections, also check orthogonality and add if passed.
                elif unique_injections < settings['max_injections']:
                    glist_new = check_ortho(fcr_sorted[k], threshhold, glist)
                    if len(glist_new) > len(glist):
                        print("Adding config with new injections to scheme")
                        glist = glist_new
                        k = k+1
                        unique_injections = unique_injections+1
                        return glist, k, unique_injections
                    else:
                        k = k+1
                        return glist, k, unique_injections
                #if maximum number of injections is reached, config cant be added. go next
                else:
                    k = k+1
                    return glist, k, unique_injections
            #if config is already in current scheme, go next
            else:
                k = k+1
                return glist, k, unique_injections
           
    k = 0
    glist = []
    threshhold = rel_resolution
    while len(glist) < stepsize:
        #go through highest ranked configs
        index = indices_sorted[k]
        #extract configuration
        config = abmnlist[index]
        
        glist, k, unique_injections = test_config(settings, config, glist, abmn_current_scheme, fcr_sorted, threshhold, unique_injections, k)
        
        if k == len(indices_sorted)-1:
            k = 0
            threshhold = threshhold+10/100*threshhold
            print("increasing threshhold")
            continue

        if len(glist) == stepsize:
            indices_chosen = np.array(glist)[:,1].astype(int)
            chosen_configs = abmnlist[indices_chosen]

            print("adding following configurations:", chosen_configs)
            print("corresponding ranks of configurations:", Fcr[indices_chosen])
                
    return chosen_configs
