# Version History

This document tracks the evolution of the wind velocity interpolation codebase.

## Version 1.0 (Current) - Publication Release
**Date Range:** September 2024 - Present
**Status:** Production Release

### Key Features
- Tucker tensor decomposition for reduced-order modeling of wind fields
- PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation for wind angle prediction
- Support for multiple geometries (cylinder/sphere, La Defense building)
- Comprehensive 2D visualization tools (contour plots, vector fields)
- VTK export functionality for 3D visualization
- Configurable preprocessing and model parameters

### Core Components
- **RegressionModel** (`src/models/regressionmodel.py`): Tucker decomposition and PCHIP interpolation
- **WindVelocityModel** (`src/models/windvelocitymodel.py`): Data loading, model orchestration, visualization
- **Plotting utilities** (`src/visualization/`): 2D slice visualization, comparison plots
- **Configuration system** (`configs/`): Centralized parameter management

### Methodology
- **Tensor Decomposition:** Tucker decomposition with configurable ranks `[65, 13, 5]`
- **Interpolation:** PCHIP for smooth wind angle interpolation
- **Data Structure:** 3D tensor `[spatial_points × wind_angles × flow_variables]`
- **Output Variables:** Pressure, Velocity (X, Y, Z), Turbulent Viscosity

### Results Summary
- **Spatial Points:** Variable (depends on CFD mesh)
- **Wind Angles Trained:** 0°, 15°, 30°, 45°, 60°, 75°, 90°, 105°, 120°, 150°, 165°, 180°
- **Interpolation Range:** Arbitrary angles within [0°, 180°]
- **Compression:** Significant reduction from full CFD dataset

### File Locations
```
src/models/               # Core model implementations
src/visualization/        # Plotting and visualization
src/data/                 # Data loading utilities
src/utils/                # Helper functions
configs/                  # Configuration files
experiments/current/      # Entry point scripts
```

### Known Limitations
- Requires pre-computed CFD data at discrete wind angles
- Tucker rank selection is manual (no automatic optimization)
- Limited to incompressible flow scenarios
- Geometry-specific configuration required

---

## Version 0.1 - Initial Research Prototype
**Date Range:** Pre-September 2024
**Status:** Deprecated

### Description
Early research implementation with flat file structure and experimental code paths.

### Deprecated Reason
- Lack of organization: all files in root directory
- Mixed experimental code with production code
- Hard-coded machine-specific paths
- Insufficient documentation
- No modular structure

### Migration Notes
Original flat structure reorganized into modular package structure for v1.0.

---

## Development Roadmap

### Planned Improvements (Future Versions)

**Version 1.1 - Enhanced Automation**
- Automatic Tucker rank selection via cross-validation
- Batch processing for multiple geometries
- Improved error handling and logging
- Unit test suite

**Version 2.0 - Extended Capabilities**
- Support for time-varying flows
- Neural network-based interpolation (comparison with PCHIP)
- Uncertainty quantification
- Integration with popular CFD solvers (OpenFOAM, ANSYS Fluent)

**Version 2.1 - Performance Optimization**
- GPU acceleration for tensor operations
- Parallel processing for multiple wind angles
- Memory-efficient data loading for large meshes
- Incremental Tucker decomposition for streaming data
