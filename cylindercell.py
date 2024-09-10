from config import config
from windvelocitymodel import *
from params import *

config["chosen_machine"] = "CREATE"

datafolder_path = os.path.join('Z:\\', "cylinder_cell", "data")
# config["plotting"]["plotting_params"] = [['X-Y',50,5]]
config["plotting"]["plotting_params"] = [['X-Z',570,5],['Y-Z',500,5],['X-Y',50,5]]
config["plotting"]["arrow"] = [True, [[500,500],[500,570]]]
config["plotting"]["save_vtk"] = True

preprocess_params['datafolder_path'] = datafolder_path
preprocess_params['make_plots'] = True

model = WindVelocityModel(preprocess_params, param_grid)
model.main()