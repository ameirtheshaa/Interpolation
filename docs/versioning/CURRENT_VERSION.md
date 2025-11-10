# Current Version: 1.0

**Release Date:** November 2024
**Status:** Production Release - Publication Ready

---

## Overview

Version 1.0 is the first production release of the wind velocity field interpolation framework, prepared for academic publication. This release implements Tucker tensor decomposition combined with PCHIP interpolation for fast prediction of wind fields around buildings at arbitrary wind angles.

---

## Core Components

### Models (`src/models/`)

#### `regressionmodel.py`
**RegressionModel** class - Core tensor decomposition and interpolation engine

**Methods:**
- `standardize_data()`: Normalize data across spatial and angular dimensions
- `perform_tucker_decomposition()`: Decompose 3D tensor into core + factor matrices
- `perform_pchip_regression()`: Fit interpolators to core tensor coefficients
- `predict(new_angles)`: Predict flow fields at arbitrary wind angles
- `execute()`: Complete pipeline execution

**Key Attributes:**
- `rank`: Tucker decomposition ranks `[R1, R2, R3]`
- `core_tensor`: Compressed core tensor
- `U, V, W`: Factor matrices (spatial, angular, variable modes)
- `coeffs_core`: PCHIP interpolator dictionary

#### `windvelocitymodel.py`
**WindVelocityModel** class - High-level orchestration and visualization

**Methods:**
- `concatenate_data_files_array()`: Load CFD data into tensor format
- `initialize_regression_model()`: Setup RegressionModel instance
- `plot_single_regression()`: Visualize interpolation quality
- `get_plotting_data()`: Generate predictions for visualization
- `plot_2d_comparison_plots_single()`: Compare actual vs predicted fields
- `main()`: Complete workflow execution

### Visualization (`src/visualization/`)

- `plotting.py`: 2D slice visualization functions
  - Contour plots with vector overlays
  - Scatter plots for raw data
  - Side-by-side actual vs. predicted comparisons

- `plotting_definitions.py`: Helper functions for plotting
  - Grid normalization
  - Geometry handling (STL files)
  - Interpolation to regular grids

### Utilities (`src/utils/`)

- `definitions.py`: Collection of utility functions
  - Data loading and preprocessing
  - GPU/device management
  - VTK export for 3D visualization
  - Scaler management
  - PCA transformers (experimental)

### Configuration (`configs/`)

- `default_config.py`: Main configuration dictionary
  - Optimizer settings (LBFGS, Adam)
  - Training parameters
  - Data paths and geometry selection
  - Plotting options

- `default_params.py`: Preprocessing parameters
  - Wind angles for training
  - Tucker decomposition ranks
  - Neural network hyperparameters (for future extensions)

### Experiments (`experiments/current/`)

- `run_cylinder.py`: Entry point for cylinder/sphere geometry
- `run_ladefense.py`: Entry point for La Defense building geometry

---

## Dependencies

### Required Libraries

**Core Scientific Computing:**
- `numpy >= 1.21.0`: Array operations
- `pandas >= 1.3.0`: Data manipulation
- `scipy >= 1.7.0`: Interpolation (PCHIP)

**Machine Learning:**
- `torch >= 1.10.0`: Tensor operations
- `scikit-learn >= 1.0.0`: Preprocessing, metrics
- `tensorly >= 0.7.0`: Tucker decomposition

**Visualization:**
- `matplotlib >= 3.4.0`: Plotting
- `plotly >= 5.3.0`: Interactive visualization

**Other:**
- `stl`: STL file handling for geometry
- `vtk >= 9.0.0`: VTK export
- `psutil`, `pynvml`, `GPUtil`: System monitoring

---

## Known Issues

### Current Limitations

1. **Tucker Rank Selection**
   - Manual specification required
   - No automatic optimization method
   - Affects compression ratio and accuracy trade-off

2. **Data Requirements**
   - Requires pre-computed CFD data at discrete angles
   - No integration with CFD solvers
   - Data must follow specific naming convention: `CFD_*_{angle}.csv`

3. **Hardcoded Paths**
   - Machine-specific paths in config (Mac, Windows)
   - Requires manual adjustment for different systems

4. **Limited Error Handling**
   - Minimal validation of input data dimensions
   - No graceful degradation for missing wind angles

5. **PCA Transformers**
   - Four experimental PCA styles included but not actively used
   - Should be deprecated or properly integrated

### Workarounds

**For Tucker Rank Selection:**
- Perform manual cross-validation
- Start with `[65, 13, 5]` as baseline
- Adjust based on reconstruction error

**For Data Paths:**
- Modify `config["chosen_machine"]` in entry scripts
- Update paths in `config["machine"]` dictionary

**For Missing Angles:**
- Ensure all angles in `config["training"]["angles_to_train"]` have corresponding CSV files

---

## Configuration Guide

### Quick Start Configuration

Minimal configuration for new geometry:

```python
# In your experiment script
config["chosen_machine"] = "mac"  # or "CREATE"
config["data"]["geometry"] = "your_geometry.stl"
config["training"]["angles_to_train"] = [0, 45, 90, 135, 180]

preprocess_params['datafolder_path'] = "/path/to/cfd/data"
preprocess_params['rank'] = [50, 10, 5]  # Adjust based on data size
```

### Advanced Configuration

**Tucker Decomposition:**
```python
preprocess_params['rank'] = [R1, R2, R3]
# R1: Spatial modes (typically largest)
# R2: Angular modes (typically ~10-20)
# R3: Variable modes (typically 5 for P, Vx, Vy, Vz, TurbVisc)
```

**Visualization:**
```python
config["plotting"]["plotting_params"] = [
    ['X-Y', z_coordinate, tolerance],
    ['X-Z', y_coordinate, tolerance],
    ['Y-Z', x_coordinate, tolerance]
]
```

---

## Usage Examples

### Basic Wind Field Interpolation

```python
from src.models.windvelocitymodel import WindVelocityModel
from configs.default_params import preprocess_params, param_grid

# Configure
preprocess_params['datafolder_path'] = "/data/cfd_results"
preprocess_params['make_plots'] = True

# Run
model = WindVelocityModel(preprocess_params, param_grid)
model.main()
```

### Predicting at New Angles

```python
from src.models.regressionmodel import RegressionModel

# After training
predictions = regression_model.predict([22.5, 67.5, 112.5])
# Returns: [n_angles, n_spatial_points, n_variables]
```

---

## Testing Status

⚠️ **No automated tests currently exist**

### Manual Testing Checklist
- [ ] Load CFD data for multiple angles
- [ ] Perform Tucker decomposition
- [ ] Interpolate at intermediate angles
- [ ] Generate visualization plots
- [ ] Export VTK files
- [ ] Compare predictions with held-out CFD data

---

## Roadmap

### Version 1.1 (Next Release)
- Add unit tests for core functions
- Automatic Tucker rank selection
- Improved path handling (environment variables)
- CLI interface for experiments

### Version 2.0 (Major Update)
- Neural network-based interpolation
- Time-varying flow support
- Uncertainty quantification
- CFD solver integration

---

## Migration from v0.1

If upgrading from the original flat structure:

1. Update imports:
   ```python
   # Old
   from regressionmodel import RegressionModel

   # New
   from src.models.regressionmodel import RegressionModel
   ```

2. Update config access:
   ```python
   # Old
   from config import config

   # New
   from configs.default_config import config
   ```

3. Update entry point execution:
   ```bash
   # Old
   python cylindercell.py

   # New
   python experiments/current/run_cylinder.py
   ```

---

## Support

For questions or issues with this version:
- Review documentation in `docs/`
- Check `docs/theory/THEORY.md` for mathematical background
- See `README.md` for usage examples
