#%%
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
import pandas as pd
import itertools
import os
import optimize
#%%

#MAKE SURE TO ENABLE NUMERICAL K FACTORS FOR RHIZOTRON OPTIMIZATION

settings = {
    'foldername':"rhizotron",               #MAIN FOLDER FOR OPTIMIZATION RUN
    'label':"v1",                           #PREFIX OF SUB-OPTIMIZATION-FOLDERS
    'remove_gamma':False,                   #REMOVE CROSS-CURRENT (WENNER GAMMA TYPE, A-M-B-N) CONFIGURATIONS?
    'K_limit':0.3,                          #MAXIMUM GEOMETRIC FACTOR FOR OPTIMIZATION
    'numerical_K':True,                     #USE NUMERICAL CALCULATION OF K? (IF FALSE, K IS CALCULATED ANALYTICALLY)
    'type':"MC",                            #TYPE OF OPTIMIZATION; OPTIONS ARE 'SC' (single channel) or 'MC' (multi channel)
    'logparams':True,                       #SET TO TRUE FOR LOGARITHMIC MODEL PARAMETERS, FALSE FOR LINEAR
    'jacobi_cellweight':False,              #TRUE ENABLES WEIGHTING OF JACOBIAN WITH CELL SIZES, DEFAULT IS FALSE
    'weighting_x':[0.07,0.52-0.07],         #OPTIMIZATION FOR TARGET (RECTANGLE) AREA, HERE X LIMITS
    'weighting_y':[0.06,0.52-0.08],         #OPTIMIZATION FOR TARGET (RECTANGLE) AREA, HERE Y LIMITS
    'lambda':10**(-6),                      #REGULARIZATION PARAMETER
    'res':100,                              #BACKGROUND RESISTIVITY FOR OPTIMIZATION (HOMOGENEOUS)
    'stepsize':1,                           #OPTIMIZATION STEPSIZE IN % OF CURRENT SET (e.g., 2 for 200 configurations)
    'max_injections':40,                    #MAXIMUM NUMBER OF INJECTIONS (ONLY USEFUL FOR MC OPTIMIZATION)
    'nr_configs':400,                       #DESIRED NUMBER OF CONFIGURATIONS IN FINAL OPTIMIZED SCHEME
    'reduce_comp':False,                    #IF TRUE, REDUCES SIZE OF COMPREHENSIVE CONFIG TO INJECTIONS WITH AT LEAST X AMOUNT OF POSSIBLE CURRENT DIPOLES (SEE BELOW)
    'min_possible_configs_per_inj':40,     #NUMBER OF MINIMUM CURRENT DIPOLES PER INJECTION IN COMPREHENSIVE SCHEME, ONLY ACTIVE IF reduce_comp=True
}

#%% create grid with given electrode positions
sensors = np.loadtxt("sensors_36.dat", skiprows=1)

#care should be taken that sensor positions lie on grid nodes
xn = np.linspace(0.06,0.46, 17)
xl = [0,0.03]
xr = [0.49,0.52]
xn = np.concatenate([xl,xn,xr])
yn = np.linspace(0.1,0.45, 15)
yu = [0.475,0.5,0.52]
yl = [0,0.05]
yn = np.concatenate([yl,yn,yu])
zn = np.linspace(0.,0.025,3)

#here, we create a 3D mesh becase optimization is performed in a tank

mesh_coarse = mt.createMesh3D(xn,yn,zn)
print(mesh_coarse)

mesh_inbetween = mesh_coarse.createH2()
print(mesh_inbetween)

mesh_fine = mesh_inbetween.createH2()
print(mesh_fine)

##assign mesh nodes for reference electrode
ref = [0.26, 0, 0.0125]
const = [0.26, 0.52, 0.0125]

# Find closest node to point ref
nid = mesh_inbetween.findNearestNode(ref)
n = mesh_inbetween.node(nid)
n.setMarker(-999)

nid_2 = mesh_inbetween.findNearestNode(const)
n_2 = mesh_inbetween.node(nid_2)
n_2.setMarker(-1000)

#coarse mesh
# Find closest node to point ref
nid_3 = mesh_coarse.findNearestNode(ref)
n_3 = mesh_coarse.node(nid_3)
n_3.setMarker(-999)

nid_4 = mesh_coarse.findNearestNode(const)
n_4 = mesh_coarse.node(nid_4)
n_4.setMarker(-1000)

#create comprehensive configuration with all possible measurements
nelecs = 36
#Use every possible configuration for comprehensive dataset
possible_combinations = nelecs*(nelecs-1)*(nelecs-2)*(nelecs-3)/8
print("Possible combinations with {}".format(nelecs) + " Electrodes are = {}".format(int(possible_combinations)))

eleclist = np.linspace(0,nelecs-1,nelecs)
combinations = list(itertools.permutations(eleclist, 4))
combinations = np.asarray(combinations)

comprehensive = combinations[np.where((combinations[:,0] < combinations[:,1]) & (combinations[:,0] < combinations[:,2]) & (combinations[:,2] < combinations[:,3]))]

if settings['remove_gamma'] == True:
    mask = (comprehensive[:,0] < comprehensive[:,2]) & (comprehensive[:,2] < comprehensive[:,1]) & (comprehensive[:,1] < comprehensive[:,3])
    comprehensive = comprehensive[~mask]
print("Length of array with possible configs: ", len(comprehensive))

scheme_comprehensive = optimize.init_scheme(sensors,comprehensive, "comprehensive config", settings['K_limit'], mesh=mesh_inbetween, numerical_K = settings['numerical_K'])
#%%
#HERE, WE CAN REDUCE THE COMPREHENSIVE SCHEME TO INJECTIONS WITH AT LEAST X POSSIBLE CONFIGURATIONS

if settings['reduce_comp'] == True:
    comprehensive = optimize.reduce_to_usable_configs(scheme_comprehensive, settings['min_possible_configs_per_inj'])
    scheme_comprehensive = optimize.init_scheme(sensors,comprehensive, "comprehensive config_reduced", settings['K_limit'], mesh=mesh_inbetween, numerical_K = settings['numerical_K'])

#%%
#CONFIGURE START SCHEME, MAKE SURE THE CHOSEN CONFIGS ARE IN THE COMPREHENSIVE SCHEME
config_start = np.array([
[31,36,32,35]])

comprehensive_abmn = np.stack([scheme_comprehensive['a'],scheme_comprehensive['b'],scheme_comprehensive['m'],scheme_comprehensive['n']], axis=1)
comprehensive_abmn = comprehensive_abmn+1

#check if rows of config start are in comprehensive scheme
for c in config_start:
    test = [c[0],c[1],c[2],c[3]] in comprehensive_abmn.tolist()
    if test == False:
        print("{} from start config is not in comprehensive config, aborting".format(c))
        raise ValueError("Config not in comprehensive config")

config_start = config_start-1   #the -1 is needed because pygimli starts counting electrodes at 0
scheme_start = optimize.init_scheme(sensors, config_start, "start_config", settings['K_limit'], mesh=mesh_inbetween, numerical_K = settings['numerical_K'])
#%%
#SET RESISTIVITY OF BACKGROUND
for cell in mesh_inbetween.cells():
    cell.setMarker(1)
rholist = [[1,settings['res']]]

rho = pg.solver.parseMapToCellArray(rholist, mesh_inbetween)
#%%
#SET THE DESIRED NOISE LEVEL FOR THE OPTIMIZATION(S); IF MORE THAN ONE, ALL COMBINATIONS WILL BE CALCULATED AND FOLDERS NAMED ACCORDING TO THE ERRORS
rel_errs = [0]
abs_errs = [0]
combinations_errs = list(itertools.product(rel_errs,abs_errs))    

for i in combinations_errs:
    settings['err_rel'] = i[0]
    settings['err_abs'] = i[1]

    if not os.path.isdir("optimizations/{}/{}_{}_{}".format(settings['foldername'],settings['label'],settings['err_rel'],settings['err_abs'])):
        os.makedirs("optimizations/{}/{}_{}_{}".format(settings['foldername'],settings['label'],settings['err_rel'],settings['err_abs']))

    #generate data
    Res, rhoa, R_noise, rhoa_noise_comp = optimize.create_error_weights(mesh_inbetween, scheme_comprehensive, np.array(rho), settings['err_rel'], settings['err_abs'])
    #run optimization
    optimize.easy_optimize_final(settings, scheme_start, scheme_comprehensive, rho, rhoa, rhoa_noise_comp,sensors,mesh_coarse, mesh_inbetween)
    
    #save data (linear parameters)
    data = np.stack([Res,R_noise,rhoa,rhoa_noise_comp], axis=1)
    df_data = pd.DataFrame(data,columns=['R','R_noise','Rhoa','Rhoa_noise'])
    df_data.to_csv('optimizations/{}/{}_{}_{}/data.csv'.format(settings['foldername'],settings['label'],settings['err_rel'],settings['err_abs']),sep=',', index=False)