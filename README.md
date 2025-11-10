# Wind Velocity Field Interpolation via Tucker Decomposition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Fast, data-driven interpolation of wind fields around buildings using tensor decomposition

---

## Overview

This repository implements a **reduced-order modeling framework** for efficient interpolation of computational fluid dynamics (CFD) results at arbitrary wind angles. The method combines:

1. **Tucker Tensor Decomposition** - Compresses 3D flow field data into low-rank components
2. **PCHIP Interpolation** - Smoothly interpolates between discrete wind angles
3. **Fast Prediction** - Achieves $10^4$-$10^6\times$ speedup over re-running CFD simulations

### Key Features

- ✅ **High Accuracy:** $R^2 > 0.95$ for pressure and velocity predictions
- ✅ **Ultra-Fast:** Predictions in ~0.1-1 second vs. hours for CFD
- ✅ **Shape-Preserving:** PCHIP avoids interpolation artifacts and overshoot
- ✅ **Flexible:** Works with arbitrary CFD meshes and geometries
- ✅ **Visualization:** Built-in 2D slice plotting and VTK export

### Applications

- 🏢 **Urban Wind Assessment:** Pedestrian comfort, wind loads
- 🌬️ **Building Aerodynamics:** Pressure distribution, wake effects
- ⚡ **Wind Energy:** Turbine wake modeling
- 🏗️ **Architectural Design:** Natural ventilation optimization

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Interpolation.git
cd Interpolation

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.models.windvelocitymodel import WindVelocityModel
from configs.default_params import preprocess_params, param_grid

# Configure data path and parameters
preprocess_params['datafolder_path'] = "/path/to/cfd/data"
preprocess_params['rank'] = [65, 13, 5]  # Tucker ranks
preprocess_params['make_plots'] = True

# Run interpolation
model = WindVelocityModel(preprocess_params, param_grid)
model.main()
```

### Running Experiments

```bash
# Cylinder/sphere geometry
python experiments/current/run_cylinder.py

# La Defense building
python experiments/current/run_ladefense.py
```

---

## Documentation

- **[Mathematical Theory](docs/theory/THEORY.md)** - Complete formulation, convergence proofs
- **[Experimental Results](docs/RESULTS.md)** - Quantitative metrics, ablation studies
- **[Version History](docs/versioning/VERSION_HISTORY.md)** - Evolution of the codebase
- **[Current Version Guide](docs/versioning/CURRENT_VERSION.md)** - API reference, configuration

For full details, see the comprehensive documentation in the `docs/` directory.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{author2024tucker,
  title={Fast Wind Field Interpolation via Tucker Decomposition and PCHIP},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

