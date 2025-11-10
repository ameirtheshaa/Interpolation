import datetime
import torch
import os

output_dir = f'output_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
os.makedirs(output_dir, exist_ok=True)

training_wind_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 150, 165, 180]
all_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
training_wind_angles = all_angles
input_params = ['Points:0', 'Points:1', 'Points:2']
output_params = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']
output_params_plotting = ['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']

plots_per_figure = 3
preprocess_params = {
    'standardize': False,
    # 'datafolder_path': datafolder_path,
    'training_wind_angles': training_wind_angles,
    'all_angles': all_angles,
    'output_params': output_params,
    'output_params_plotting': output_params_plotting,
    'input_params': input_params,
    'plots_per_figure': plots_per_figure,
    'interp_method': 'nn',
    'rank': [65,13,5],
    'make_plots': False,
    'make_all_plots': False,
    'nn_params': {
        'num_epochs': int(1e5),
        'lr': 1e-3,
        'threshold': 1e-7,
        'batch_size': 1,
        'neuron_num': 8,
        'fn': 'elu',
        'fn_': torch.nn.ELU(),
        # 'fn': 'sigmoid',
        # 'fn_': torch.nn.Sigmoid(),
        # 'fn': 'tanh',
        # 'fn_': torch.nn.Tanh(),
        # 'fn': 'gelu',
        # 'fn_': torch.nn.GELU(),
        # 'fn': 'celu',
        # 'fn_': torch.nn.CELU(),
        'load_nn': False,
        'load_nn_model_savename': os.path.join(output_dir, 'nn_savemodel_elu_4.701094837855635e-07_2024_07_05_22_40_41.pth'),
        'dynamic_lr': True,
        'grid_search':False,
        'dropout_rate': 0.2,
    },
    'single_regression_savename': os.path.join(output_dir, f'alpha_coeffs_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'),
    'savename_2d_comparison': os.path.join(output_dir, f'2dxy_COMPARISON_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'),
    'savename_2d_single': os.path.join(output_dir, f'2dxy_SINGLE_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'),
    'output_dir': output_dir,
}

param_grid = {
    'batch_size': [2**5],
    'neuron_num': [2**5, 2**6, 2**7, 2**8],
    # 'fn': ['elu', 'celu', 'gelu'],
    # 'fn_': [torch.nn.ELU(), torch.nn.CELU(), torch.nn.GELU()],
    'dropout_rate': [0.1,0.2,0.3],
}