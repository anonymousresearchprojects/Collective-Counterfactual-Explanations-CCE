# Collective Algorithmic Recourse (CARLA) - Public Release

## Overview
This repository provides tools and benchmarks for collective algorithmic recourse, including implementations of state-of-the-art recourse methods, simulation scripts, and optimal transport utilities. It also includes the back-and-forth method (BFM) for optimal transport, with Python and MATLAB wrappers for C code.

## Features
- Benchmarking of algorithmic recourse methods
- Simulation and data collection scripts
- Integration with CARLA recourse library
- Back-and-forth method for optimal transport (Python, C, MATLAB)
- Utilities for unbalanced optimal transport

## Installation

### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/collective_algorithmic_recourse_anonymous.git
cd collective_algorithmic_recourse_anonymous
```

### 2. Create a Virtual Environment
We recommend using conda:
```sh
conda create --name recourse python=3.6.13 pip
conda activate recourse
```
Or with venv:
```sh
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

#### Additional dependencies (if needed):
```sh
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/carla-recourse/carla.git
pip install protobuf==3.20.0
```

### 4. (Optional) Install BFM Python bindings
```sh
pip install ./bfm/python
```

## Usage
- To run a benchmark:
  ```sh
  python benchmark.py --seed 101
  ```
- To collect data:
  ```sh
  python collection_script.py
  ```
- For BFM examples, see `bfm/python/example.ipynb` or `bfm/python/example.py`.

## Project Structure
- `src/` - Main Python source code
- `bfm/` - Back-and-forth method (C, Python, MATLAB)
- `benchmark.py` - Benchmarking script
- `collection_script.py` - Data collection script
- `requirements.txt` - Python dependencies

## Contributing
Contributions are welcome! Please open issues or pull requests for suggestions, bug reports, or improvements.

## License
This project is licensed under the MIT License. See `unbalancedTransport/LICENSE.md` for details.
