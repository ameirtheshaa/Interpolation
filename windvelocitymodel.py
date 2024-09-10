import pandas as pd
import os
import time
import datetime
import copy
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from plotting import *
from plotting_definitions import *
from config import config
from regressionmodel import *

class WindVelocityModel:
    def __init__(self, preprocess_params, param_grid=None):
        self.preprocess_params = preprocess_params
        self.param_grid = param_grid
        self.datafolder_path = self.preprocess_params['datafolder_path']
        self.output_params = self.preprocess_params['output_params']
        self.output_params_plotting = self.preprocess_params['output_params_plotting']
        self.input_params = self.preprocess_params['input_params']
        self.training_wind_angles = self.preprocess_params['training_wind_angles']
        self.all_angles = self.preprocess_params['all_angles']
        self.plots_per_figure = self.preprocess_params['plots_per_figure']
        self.single_regression_savename = self.preprocess_params['single_regression_savename']
        self.savename_2d_comparison = self.preprocess_params['savename_2d_comparison']
        self.savename_2d_single = self.preprocess_params['savename_2d_single']
        self.rank = self.preprocess_params['rank']
        self.standardize = self.preprocess_params['standardize']
        self.make_plots = self.preprocess_params['make_plots']
        self.make_all_plots = self.preprocess_params['make_all_plots']
        self.output_dir = self.preprocess_params['output_dir']
        self.regression_model = None  # Placeholder for the RegressionModel instance

    def get_filenames_from_folder(self, extension, startname):
        return [f for f in os.listdir(self.datafolder_path) if os.path.isfile(os.path.join(self.datafolder_path, f)) and f.endswith(extension) and f.startswith(startname)]

    def concatenate_data_files_array(self, training_wind_angles, output_params):
        dfs = []
        for wind_angle_idx, wind_angle in enumerate(training_wind_angles):
            for filename in sorted(self.filenames):
                if int(filename.split('_')[-1].split('.')[0]) == wind_angle:
                    df = pd.read_csv(os.path.join(self.datafolder_path, filename))
                    df_pos = df[['Points:0','Points:1','Points:2']]
                    df['Velocity_X'] = df['Velocity:0']
                    df['Velocity_Y'] = df['Velocity:1']
                    df['Velocity_Z'] = df['Velocity:2']
                    df['Velocity_Magnitude'] = np.sqrt(df['Velocity_X']**2 + df['Velocity_Y']**2 + df['Velocity_Z']**2)
                    df['vxvy'] = np.sqrt(df['Velocity:0']**2 + df['Velocity:1']**2)/np.sqrt(df['Points:0']**2 + df['Points:1']**2 + df['Points:2']**2)
                    df['Vcom'] = df['vxvy'] + 1j * df['Velocity:2']/np.sqrt(df['Points:0']**2 + df['Points:1']**2 + df['Points:2']**2)
                    df['WindAngle'] = wind_angle
                    df = df[output_params]
                    if wind_angle_idx == 0:
                        data_array = np.zeros((len(df), len(training_wind_angles), len(output_params)))
                        self.df_pos = df_pos
                    data_array[:, wind_angle_idx, :] = df.to_numpy()
        return data_array

    def concatenate_data_files(self, training_wind_angles, output_params):
        dfs = []
        for wind_angle in training_wind_angles:
            for filename in sorted(self.filenames):
                if int(filename.split('_')[-1].split('.')[0]) == wind_angle:
                    df = pd.read_csv(os.path.join(self.datafolder_path, filename))
                    df['Velocity_X'] = df['Velocity:0']
                    df['Velocity_Y'] = df['Velocity:1']
                    df['Velocity_Z'] = df['Velocity:2']
                    df['Velocity_Magnitude'] = np.sqrt(df['Velocity_X']**2 + df['Velocity_Y']**2 + df['Velocity_Z']**2)
                    df['vxvy'] = np.sqrt(df['Velocity:0']**2 + df['Velocity:1']**2)/np.sqrt(df['Points:0']**2 + df['Points:1']**2 + df['Points:2']**2)
                    df['Vcom'] = df['vxvy'] + 1j * df['Velocity:2']/np.sqrt(df['Points:0']**2 + df['Points:1']**2 + df['Points:2']**2)
                    df['WindAngle'] = wind_angle
                    df = df[output_params]
                    dfs.append(df)
        data = pd.concat(dfs, axis=1, ignore_index=True)
        return data

    def initialize_regression_model(self, data):
        self.regression_model = RegressionModel(
            data=data,
            rank=self.rank,
            output_params=self.output_params,
            training_wind_angles=self.training_wind_angles,
            standardize=self.standardize
        )

    def plot_single_regression(self):
        theta_line = np.linspace(min(self.training_wind_angles), max(self.training_wind_angles), 1000)
        interpolated_dict = self.regression_model.predict(theta_line)
        
        if self.single_regression_savename is not None:
            os.makedirs(self.single_regression_savename, exist_ok=True)
            rows_per_figure = self.plots_per_figure
            num_figures = (self.regression_model.alpha_matrix.shape[0] + rows_per_figure - 1) // rows_per_figure
            for fig_num in range(num_figures):
                for component_index in range(interpolated_dict.shape[2]):
                    plt.figure(figsize=(32, 16))
                    for i in range(rows_per_figure):
                        row_index = fig_num * rows_per_figure + i
                        if row_index >= self.regression_model.alpha_matrix.shape[0]:
                            break
                        actual_values = self.regression_model.alpha_matrix[row_index, :, component_index]
                        interpolated_values_component = interpolated_dict[row_index, :, component_index].reshape(-1,1)
                        plt.subplot(rows_per_figure, 1, i + 1)
                        plt.plot(self.all_angles, actual_values, 'o', label=f'Actual values - Row {i + 1}')
                        plt.plot(theta_line, interpolated_values_component, '-', label=f'Interpolated Values - Row {i + 1}')
                        plt.text(min(theta_line), max(np.abs(actual_values)), f'Mode Number - {row_index + 1} for {self.veltag}', color='black')
                        plt.title(f'Alpha Values, Component {self.output_params[component_index]} - Spline Fit over Theta, for Row {row_index + 1} - {self.veltag}')
                        plt.xlabel('Theta (Degrees)')
                        plt.ylabel('Magnitude of Alpha Matrix')
                        plt.legend()
                        plt.grid(True)
                    plt.savefig(os.path.join(self.single_regression_savename, f'alpha_coeffs_{self.output_params_plotting[component_index]}_{fig_num}.png'))
                    plt.close()

    def get_plotting_data(self, angle):
        predict_dict = self.regression_model.predict([angle])
        predictions_3d = self.regression_model.inverse_standardize_data(predict_dict)
        predictions_tensor = multi_mode_dot(predictions_3d, [self.regression_model.U], modes=[0])
        predictions_tensor = predictions_tensor.squeeze(axis=1)
        return predictions_tensor

    def plot_2d_comparison_plots_single(self, angle):
        targets_135 = self.concatenate_data_files([angle], self.output_params)
        result_singlevar135 = self.get_plotting_data(angle)
        column_names = ['X', 'Y', 'Z'] + [item + "_Actual" for item in self.output_params_plotting] + [item + "_Predicted" for item in self.output_params_plotting] + ['WindAngle']
        df = pd.DataFrame(np.hstack((self.xyz, targets_135, result_singlevar135, np.full((targets_135.shape[0], 1), angle))), columns=column_names)
        df['Velocity_Magnitude_Actual'] = np.sqrt(df['Velocity_X_Actual']**2 + df['Velocity_Y_Actual']**2 + df['Velocity_Z_Actual']**2)
        df['Velocity_Magnitude_Predicted'] = np.sqrt(df['Velocity_X_Predicted']**2 + df['Velocity_Y_Predicted']**2 + df['Velocity_Z_Predicted']**2)
        plot_data_2d(config, df, [angle], os.path.join(self.datafolder_path, config["data"]["geometry"]), self.savename_2d_comparison, single=False)
        if config["plotting"]["save_vtk"]:
            predictions_column_names = [item + "_Predicted" for item in self.output_params_plotting]
            vtk_output_folder = os.path.join(self.output_dir, f'vtk_output_comparison')
            os.makedirs(vtk_output_folder, exist_ok=True)
            output_nn_to_vtk(config, angle, f'predictions_for_wind_angle_{angle}', df, predictions_column_names, vtk_output_folder)

    def plot_2d_prediction_plots_single(self, angle):
        x_new_reconstructed = self.get_plotting_data(angle)
        column_names = ['X', 'Y', 'Z'] + self.output_params_plotting + ['WindAngle']
        df = pd.DataFrame(np.hstack((self.xyz, x_new_reconstructed, np.full((x_new_reconstructed.shape[0], 1), angle))), columns=column_names)
        df['Velocity_Magnitude'] = np.sqrt(df['Velocity_X']**2 + df['Velocity_Y']**2 + df['Velocity_Z']**2)
        plot_data_2d(config, df, [angle], os.path.join(self.datafolder_path, config["data"]["geometry"]), self.savename_2d_single, single=True, df_forced=self.df_forced)
        if config["plotting"]["save_vtk"]:
            predictions_column_names = [item + "_Predicted" for item in self.output_params_plotting]
            vtk_output_folder = os.path.join(self.output_dir, f'vtk_output_preds')
            os.makedirs(vtk_output_folder, exist_ok=True)
            output_nn_to_vtk(config, 0, f'predictions_for_wind_angle_{angle}', df, predictions_column_names, vtk_output_folder)

    def main(self):
        self.veltag = 'all'
        self.filenames = self.get_filenames_from_folder('.csv', 'CFD')
        self.data = self.concatenate_data_files_array(self.training_wind_angles, self.output_params)
        print("Data Matrix Shape:", self.data.shape)
        self.initialize_regression_model(self.data)  # Initialize and prepare the regression model
        self.regression_model.execute()  # Perform decomposition and regression

        if self.make_plots:
            self.plot_single_regression()
            self.xyz = np.array(self.concatenate_data_files([0], self.input_params))
            self.df_forced = self.concatenate_data_files([135], ['Velocity_Magnitude'] + self.output_params_plotting + ['WindAngle'])
            self.df_forced.columns = ['Velocity_Magnitude'] + self.output_params_plotting + ['WindAngle']
            if not self.make_all_plots:
                self.plot_2d_comparison_plots_single(135)
            else:
                for angle in range(min(self.all_angles), max(self.all_angles) + 1, 1):
                    self.plot_2d_prediction_plots_single(angle)
                    if angle in self.all_angles:
                        self.plot_2d_comparison_plots_single(angle)