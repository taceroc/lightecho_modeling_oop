import os

#DIRECTORIES
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
CONFIG_PATH_UTILS = os.path.join(ROOT_DIR, 'utils.py') 
print(CONFIG_PATH_UTILS)

CONFIG_PATH_constants = os.path.join(ROOT_DIR, 'dust')
print(CONFIG_PATH_constants)
CUBE_NAME = "spitzer_batch_2d.npy"
# CUBE_NAME = "spitzer_batch1_2d60.npy"

PATH_TO_DUST_CUBE = os.path.join(ROOT_DIR, 'dust\\cube\\Data') 
PATH_TO_DUST_CUBE = os.path.join(PATH_TO_DUST_CUBE, CUBE_NAME) 
print(PATH_TO_DUST_CUBE)



dir_save_result = 'results'
PATH_TO_RESULTS = os.path.join(ROOT_DIR, dir_save_result) 

dir_save_simulations = 'results\\arrays'
PATH_TO_RESULTS_SIMULATIONS = os.path.join(ROOT_DIR, dir_save_simulations) 

dir_save_figures = 'results\\figures'
PATH_TO_RESULTS_FIGURES = os.path.join(ROOT_DIR, dir_save_figures) 

#