# Optimized measurement scheme design for ERT surveys

In this git repository, a code for optimizing electrical resistivity surveys in single- and multichannel mode is presented. The algorithm to compute the optimized measurement scheme is termed "Compare-R" and implemented after [Wilkinson et al. 2012](#1) and [Loke et al. 2010](#2). The whole optimization is based on the maximization of the diagonal entries of the model resolution matrix, which is an indicator for the reconstruction capabilities of a given measurement scheme. The basic steps in the algorithm are:

1. Compute the model resolution matrix $R_b$ of a starting scheme (a few configurations) using the Jacobian matrix $G$ (here, $G_b$ contains the sensitivities of the configurations in the base set):

    $$R_b = (G_b^TG_b+\lambda C)^{-1}G_b^TG_b.$$

    $\lambda$ is a damping factor that regularizes the inversion, and $C$ denotes a matrix that contains the inversion constraints (e.g., spatial smoothing of neighbouring cells - per default, second-order smoothing constraints are enabled).
    If one wants to include error estimates in the resolution matrix, the Jacobian can be weighted with the reciprocal of the individual error estimate of each measurement over a weighting matrix $W_d$:

    $$R_b = (G_b^TW_d^TW_dG_b+\lambda C)^{-1}G_b^TW_d^TW_dG_b.$$

2. As in step 1, compute the model resolution matrix $R_c$ of a comprehensive scheme containing all possible configurations of a given electrode array (as this is computationally demanding, this is the step that will take the longest).
3. For a number of configurations, based on a predetermined step size, calculate an updated resolution matrix $R_{b+1}$, that contains the (approximated) change $\Delta R_b$ in the diagonal of the resolution matrix of the base set when a configuration is added:

    $$R_{b+1} = R_b + \Delta R_b.$$

    $\Delta R_b$ denotes the so-called "Sherman-Morrison-Rank-1-Update", and is used to avoid computationally intensive recalculation of the model resolution matrix that would be necessary for each added configuration. We calculate all $\Delta R_b$, collected as rows in one matrix $U$, using the following equations:

    $$U = [Z\oslash(\textbf{1} + \text{diag}(G_{add}Z))^T]^{T}\circ(G_{add}-Y^T),$$

    with

    $$Z = (G_b^TW_d^TW_dG_b+\lambda C)^{-1}G_{add}^T,$$

    and 

    $$Y = G_b^TW_d^TW_dG_bZ.$$

    where $G_{add}$ contains the sensitivities of the added configurations as rows (Jacobian of the added configurations) and $Z$ in a similar fashion for a number of different added test arrays. The bold $\textbf{1}$ stands for a vector containing ones with the dimension ($n \times 1$), where n is the number of configurations, and the symbols $\oslash$ and $\circ$ stand for the Hadamard (element-wise) division and multiplication, respectively. After the calculation of the matrix $U$ containing all possible updates $\Delta R_b$ for the resolution matrix diagonal, the added configurations are ranked after a ranking function:

    $$F_{CR} = \frac{1}{m} \sum_{j=1}^{m} \frac{w_j\Delta R_{b,j}}{R_{c,j,j}}.$$

    Note that $R_{c,j,j}$ is the diagonal of the resolution matrix of the comprehensive set, $w_j$ is a weighting factor that is one if it is a cell in the area of interest, and $10^{-12}$ if it is not (see for example in [Wilkinson et al. 2015](#3) or [Uhlemann et al. 2018](#4)). The higher the change in the entries of the resolution matrix, the higher the rank calculated by this function.

4. In the last step, the highest ranked configurations are added to the base set - the first one always, the second, third and so on only if their sensitivity vector is sufficiently orthogonal to the first one based on this criterion:
	
	$$\frac{|g_1\cdot g_2|}{|g_1|\cdot|g_2|} < S.$$
	
    Here, $g$ denotes the sensitivity of a given configuration, and S the average relative resolution of the base scheme compared to the comprehensive scheme:
	
	$$S = \frac{1}{n} \sum_{k=1}^{n} \frac{R_{b,k}}{R_{c,k}}$$
	
	with n being the number of model cells inside the weighted area of interest. The orhtogonality limit of S was set because it produced the best results in past studies ([Wilkinson et al. 2012](#2)).

After a certain number of added configurations (step 3 and 4), the resolution matrix of the base scheme + added configurations is recalculated. Then a new iteration of model resolution updates is performed. This procedure is repeated as long as the preset maximum number of configurations is not reached, resulting in the full optimized measurement scheme.

## Installation

To be able to install and run the code, the required packages in the requirements.txt need to be installed first. The code was written for Python version 3.11.9, so it is recommended to create a virtual environment with this version to avoid unexpected errors.

You can set up a virtual environment and install all needed packages with the virtualenvwrapper package using

```
mkvirtualenv optimization
pip install -r requirements.txt
```

If you are using anaconda, install a new virtual environment with the required packages using

```
conda env create -f optimization_env.yml
```

The optimization code can then be installed from the main folder using 

```
pip install .
```

## Basic usage

In the following, a description of the basic usage of the code is given. The sensitivities needed to compute the model resolution matrix are calculated using pyGIMLI (https://www.pygimli.org/). To understand the workflow, it is best to look at one of the example optimizations in the folder "examples". There are examples for single-channel optimization, multi-channel optimization, and optimization for a (3D) tank geometry like a rhizotron.

The function that runs the optimization requires a settings dictionary, placed at the beginning of the script, with the following adjustable parameters:

    'foldername':"output_name",             #FOLDER NAME FOR OPTIMIZATION RUN
    'label':"v1",                           #PREFIX OF SUB-OPTIMIZATION-FOLDERS THAT ARE CREATED IN MAIN FOLDER
    'remove_gamma':False,                   #REMOVE CROSS-CURRENT (WENNER GAMMA TYPE, A-M-B-N) CONFIGURATIONS?
    'K_limit':10,                           #MAXIMUM GEOMETRIC FACTOR FOR OPTIMIZATION
    'numerical_K':False,                    #USE NUMERICAL CALCULATION OF K? (IF FALSE, K IS CALCULATED ANALYTICALLY)
    'type':"MC",                            #TYPE OF OPTIMIZATION; OPTIONS ARE 'SC' (single channel) or 'MC' (multi channel)
    'logparams':True,                       #SET TO TRUE FOR LOGARITHMIC MODEL PARAMETERS, FALSE FOR LINEAR
    'jacobi_cellweight':False,              #TRUE ENABLES WEIGHTING OF JACOBIAN WITH CELL SIZES, DEFAULT IS FALSE
    'weighting_x':[0,9.75],                 #OPTIMIZATION FOR TARGET (RECTANGLE) AREA, HERE X LIMITS
    'weighting_y':[-2,-0],                  #OPTIMIZATION FOR TARGET (RECTANGLE) AREA, HERE Y LIMITS
    'lambda':10**(-6),                      #REGULARIZATION PARAMETER
    'res':100,                              #BACKGROUND RESISTIVITY FOR OPTIMIZATION (HOMOGENEOUS)
    'stepsize':1,                           #OPTIMIZATION STEPSIZE IN % OF CURRENT SET (e.g.,in the default setting of 1%, 2 for 200 configurations)
    'max_injections':40,                    #MAXIMUM NUMBER OF INJECTIONS (ONLY USEFUL FOR MC OPTIMIZATION)
    'nr_configs':400,                       #DESIRED NUMBER OF CONFIGURATIONS IN FINAL OPTIMIZED SCHEME
    'reduce_comp':True,                     #IF TRUE, REDUCES SIZE OF COMPREHENSIVE CONFIG TO INJECTIONS WITH AT LEAST X AMOUNT OF POSSIBLE CURRENT DIPOLES (SEE BELOW)
    'min_possible_configs_per_inj':10,      #NUMBER OF MINIMUM CURRENT DIPOLES PER INJECTION IN COMPREHENSIVE SCHEME, ONLY ACTIVE IF reduce_comp=True

1. In a first step, the electrode positions and the grid are created as a pygimli mesh object. This can either be done within the script, or over external tools like gmesh. It is recommended to used a grid witch unifrom cell size, as the built-in constraint generating functions of pygimli do not account for irregular cell sizes (?). A workaround could be the implementation of geostatistical constraints (https://www.pygimli.org/_tutorials_auto/3_inversion/plot_6-geostatConstraints.html).
The optimization algorithm may take two grids as arguments: A fine mesh for the calculation of the sensitivities, and a coarser mesh for calculations of R. I implemented this to reduce computation time for larger grids. The sensitivities are interpolated to the coarser mesh after calculation on the fine mesh. If the same mesh is given for both parameters, no interpolation is performed.

2. The Jacobian of the comprehensive scheme, consisting of every possible configuration for the given electrodes, is calculated on the chosen grid. Here, one can choose to remove Wenner-Gamma-configurations ('remove_gamma': False), the numerical or analytical calculation of the geometric factors ('numerical_K':False), and a maximum geometric factor ('K_limit':10). Additionally, for multichannel optimization, it is possible to reduce the comprehensive scheme to configurations that have at least X amount of possible current dipoles ('reduce_comp':True and 'min_possible_configs_per_inj':10). All of these settings can help to reduce computation time.

3. A starting scheme is chosen. Make sure that the starting scheme is part of the comprehensive scheme - if not, an error will be raised. Usually, the starting scheme only consists of a few configurations (two in the example for single and multichannel optimization).

4. A homogeneous background resistivity ('res':100 in the settings dictionary) is assigned to the cells of the (finer) grid. One could also use other resistivity models if the geology is known, or certain structures are expected.

5. The optimization is run with a combination of relative and absolute resistance noise levels in the lists "rel_errs" and "abs_errs". If you do not want to include error estimates, just enter 0 for both lists. If you want to try more than one noise level, the code will combine all possible combinations of noise levels in both lists. The output folders will be named according to the chosen noise levels.

6. After executing the script, the optimized measurement scheme is saved under "optimizations/foldername/label_rel_abs", where foldername and label are the foldername and label given in the settings dictionary, and rel and abs the relative and absolute noise levels assumed in the optimization. This folder contains
    1. The computed datapoints (and associated errors) in "data.csv".
    2. The evolution of the optimization run in "evolution_opt.csv", which stores the iteration number, number of configs, and the relative resolution of the model resolution matrix in comparison to the used comprehensive scheme (maximum relative resolution is 1).
    3. The used settings dictionary in "settings.csv".
    4. The optimized scheme, given as an (a b m n) file in "opt_scheme.dat".

## References

<a id="1">[1]</a> 
Loke, M.H., Wilkinson, P.B., Chambers, J.E., 2010. Parallel computation of optimized arrays for 2-D electrical imaging surveys: Parallel computation of optimized arrays. Geophysical Journal International 183, 1302–1315. https://doi.org/10.1111/j.1365-246X.2010.04796.x

<a id="2">[2]</a> 
Wilkinson, P.B., Loke, M.H., Meldrum, P.I., Chambers, J.E., Kuras, O., Gunn, D.A., Ogilvy, R.D., 2012. Practical aspects of applied optimized survey design for electrical resistivity tomography: Applied optimised ERT survey design. Geophysical Journal International 189, 428–440. https://doi.org/10.1111/j.1365-246X.2012.05372.x

<a id="3">[3]</a> 
Wilkinson, P.B., Uhlemann, S., Meldrum, P.I., Chambers, J.E., Carrière, S., Oxby, L.S., Loke, M.H., 2015. Adaptive time-lapse optimized survey design for electrical resistivity tomography monitoring. Geophys. J. Int. 203, 755–766. https://doi.org/10.1093/gji/ggv329

<a id="4">[4]</a> 
Uhlemann, S., Wilkinson, P.B., Maurer, H., Wagner, F.M., Johnson, T.C., Chambers, J.E., 2018. Optimized survey design for electrical resistivity tomography: combined optimization of measurement configuration and electrode placement. Geophysical Journal International 214, 108–121. https://doi.org/10.1093/gji/ggy128



