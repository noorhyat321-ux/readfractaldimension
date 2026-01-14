<p align="center">
  <img src="./Figure%201" alt="Project header image" width="600"/>
</p>

# ReadFractalDimension

ReadFractalDimension — a small toolkit to compute fractal dimensions (e.g., box-counting) for time series and simulated GBM processes.

## Features
- Compute fractal dimensions for geometric Brownian motion (GBM) and other time series.
- Includes analysis scripts and simple examples.
- Lightweight, Python-only (see requirements.txt).

## Installation
1. Ensure Python 3.8+ is installed.
2. Create and activate a virtual environment:
   `python -m venv .venv`
   `source .venv/bin/activate`  # macOS/Linux
   `.venv\\Scripts\\activate`  # Windows (PowerShell)
3. Install dependencies:
   `pip install -r requirements.txt`

## Usage
- Run the main analysis script:
  `python GBM_Fractal_Analysis.py`
- View help/options:
  `python GBM_Fractal_Analysis.py --help`

### Example
```bash
python GBM_Fractal_Analysis.py
```

## Contributing
- Open issues for bugs or feature requests.
- Submit pull requests with tests and clear descriptions.

## License
MIT (replace as appropriate for this project)
