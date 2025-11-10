import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg import multi_mode_dot
from scipy.interpolate import PchipInterpolator
import numpy as np

class RegressionModel:
    def __init__(self, data, rank, output_params, training_wind_angles, standardize=True):
        self.data = data
        self.rank = rank
        self.output_params = output_params
        self.training_wind_angles = training_wind_angles
        self.standardize = standardize
        self.mean = None
        self.std = None
        self.core_tensor = None
        self.U = None
        self.V = None
        self.W = None
        self.alpha_matrix = None
        self.coeffs_core = None

    def standardize_data(self):
        if self.standardize:
            self.mean = np.mean(self.data, axis=(0, 1), keepdims=True)
            self.std = np.std(self.data, axis=(0, 1), keepdims=True)
            self.std[self.std == 0] = 1
            self.data = (self.data - self.mean) / self.std

    def inverse_standardize_data(self, data):
        return (data * self.std + self.mean) if self.standardize and self.mean is not None and self.std is not None else data

    def perform_tucker_decomposition(self):
        self.core_tensor, factors = tucker(self.data, rank=self.rank, n_iter_max=1000, tol=0.0001, verbose=True)
        self.U, self.V, self.W = factors
        self.alpha_matrix = multi_mode_dot(self.core_tensor, [self.V, self.W], modes=[1, 2])

    def perform_pchip_regression(self):
        theta = self.training_wind_angles
        self.coeffs_core = {
            component: [PchipInterpolator(theta, self.alpha_matrix[i, :, j]) for i in range(self.alpha_matrix.shape[0])]
            for j, component in enumerate(self.output_params)
        }

    def predict(self, new_angles):
        return np.array([
            [interp(new_angles) for interp in self.coeffs_core[component]]
            for component in self.output_params
        ]).transpose(1, 2, 0)

    def compute_divergence(self, tensor, positions):
        grads = [np.gradient(tensor[:, :, i], positions[:, i], axis=0) for i in range(3)]
        return np.stack(grads, axis=-1)

    def compute_alpha_divergence(self, positions):
        return self.compute_divergence(multi_mode_dot(self.alpha_matrix, [self.U], modes=[0]), positions)

    def execute(self):
        self.standardize_data()
        self.perform_tucker_decomposition()
        self.perform_pchip_regression()
        return self.core_tensor, self.U, self.V, self.W