import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from configs.default_config import config
from src.models.windvelocitymodel import *
from configs.default_params import *

config["chosen_machine"] = "CREATE"
datafolder_path = os.path.join('Z:\\', "ladefense", "data")
config["plotting"]["plotting_params"] = [['X-Z',-300,5],['Y-Z',-300,5],['X-Y',5,5]]
config["plotting"]["lim_min_max"] = [(-0.3, 0.3),(-0.3, 0.3),(0, 0.25)]
config["training"]["boundary"] = [[-2520,2520,-2520,2520,0,1000],100]
config["data"]["geometry"] = 'ladefense.stl'
config["plotting"]["save_vtk"] = True

preprocess_params['nn_params']['num_epochs'] = int(1e2)
preprocess_params['datafolder_path'] = datafolder_path

model = WindVelocityModel(preprocess_params, param_grid)
model.main()