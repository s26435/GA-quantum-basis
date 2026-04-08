# GA Quantum Basis Optimizer

A genetic algorithm framework for optimizing Gaussian basis set exponents in quantum chemistry calculations using the MOLCAS computational package.

## Overview

This project implements a hybrid genetic algorithm combined with a neural network-based population generator to optimize the number and values of exponents in quantum chemistry basis sets. The goal is to find minimal basis sets that maintain quantum chemistry calculation accuracy while reducing computational cost.

**Key Features:**
- Genetic algorithm with tournament selection and multi-point crossover
- Neural network population generator for guided exploration
- MOLCAS integration for quantum chemistry energy calculations
- SQLite caching to avoid redundant calculations
- Configurable fitness function with basis set size penalty
- Multi-core execution support

## Project Structure

```
├── main.py                  # Entry point - run with `python3 main.py`
├── requirements.txt         # Python dependencies
├── cleanup.sh              # Clean up all generated files
│
├── src/
│   ├── ga.py              # Main GA implementation
│   ├── config.py          # GA_cfg configuration dataclass
│   ├── model.py           # Population Generator neural network
│   ├── energy.py          # MOLCAS integration and energy calculations
│   ├── cache.py           # Energy caching system (SQLite)
│   ├── build_input.py     # Generate MOLCAS INPUT files
│   ├── parse_energy.py    # Extract energy from MOLCAS output
│   ├── parse_cmocorr.py   # Parse CMOCORR results
│   ├── cmocorr_util.py    # CMOCORR utility functions
│   ├── util.py            # Utility functions
│   ├── globals.py         # Global constants and blocks
│   ├── __init__.py        # Package init
│   └── INPUT/             # Template MOLCAS input files
│
├── analize.py             # Metrics visualization script
└── comp.py                # Genome comparison utility
```

## Installation

### Prerequisites
- Python 3.8+
- MOLCAS quantum chemistry package (with environment properly configured)
- NVIDIA CUDA (optional, for GPU acceleration)

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Execution

```bash
python3 main.py
```

The algorithm will:
1. Initialize population with seed exponents
2. Run energy calculations via MOLCAS
3. Evolve population over generations
4. Save results and metrics
5. Output best genome, mask, fitness, and energy

### Configuration

Modify `src/config.py` to customize the algorithm. Key parameters:

#### GA Parameters
- `population_size`: Number of individuals per generation (default: 30)
- `generations`: Total generations to run (default: 1000)
- `genome_size`: Number of exponents (default: sum of BLOCKS)

#### Fitness Function
- `start_lambda`: Weight for basis set size penalty (default: 5e-4)
- `min_mask_size`: Minimum exponents before penalty (default: 14)
- `ground_truth`: Reference energy for error calculation (default: -14.668)

#### Genetic Operators
- `elite_frac`: Fraction of top genomes copied to next gen (default: 0.2)
- `tournament_k`: Tournament selection pressure (default: 2)
- `crossover_p`: Crossover probability (default: 0.9)
- `mutation_p`: Exponent mutation probability (default: 0.4)
- `mutation_sigma`: Mutation strength (default: 0.6)
- `mask_flip_p`: Mask mutation probability (default: 0.01)

#### Population Generator
- `zdim`: Latent space dimension (default: 16)
- `gen_percent`: Fraction of population generated per generation (default: 0.5)
- `lr`: Generator learning rate (default: 1e-3)
- `weight_decay`: L1/L2 regularization (default: 1e-4)

#### System Configuration
- `device`: "cpu" or "cuda" for computation device
- `work_root`: Working directory for outputs
- `python_bin`: Path to Python executable
- `molcas_cmd`: MOLCAS command
- `db_path`: Path to caching database (default: cache/energy.sqlite)

## Output Files

The algorithm generates the following during execution:

### Metrics & Data
- **metrics.csv** - Training metrics per generation:
  - generation, generator_loss, gen_non_penalty_rate, ga_non_penalty_rate
  - best_fit, mean_fit, average_length, best_length
  - best_energy, error, current_lambda

- **populations.csv** - Detailed data for every genome:
  - exponents, masks, fitness values, energy values per generation

### Model & Cache
- **model.ckpt** - Population Generator checkpoint
- **cache/energy.sqlite** - Energy calculation cache (prevents recomputation)

### Calculation Outputs
- **gen_***/** - Directories per generation containing:
  - JSON input files
  - MOLCAS INPUT files
  - MOLCAS output files and logs

### Logging
- **out.log** - Training logs and debugging info
- **compare.png** - Visualization from comp.py

## Fitness Function

The fitness function balances energy accuracy with basis set economy:

```
Loss = E + λ × (mask_penalty)
```

Where:
- `E` is the quantum chemistry energy
- `λ` is dynamically scheduled (`start_lambda` parameter)
- `mask_penalty` is the number of active exponents
- Penalties applied if exponents < `min_mask_size`

## Utilities

### analize.py
Generates visualization of training metrics from `metrics.csv`:
```bash
python3 analize.py
```

### comp.py
Compares two genomes by their exponent values:
```bash
# Edit tmp.txt with two genomes, then run
python3 comp.py
```

### cleanup.sh
**WARNING**: Permanently deletes all generated files:
```bash
bash cleanup.sh
```

## Important Notes

- **Temp Directory**: MOLCAS may accumulate large temporary files in `/tmp/`. Periodically clean to prevent disk space issues.
- **Caching**: Energy values are cached in SQLite to avoid redundant MOLCAS calculations. Clear the cache to recalculate.
- **GPU Support**: Set `device: "cuda"` in config for GPU acceleration (requires NVIDIA GPU and CUDA).
- **Multi-core**: Configure `local_max_workers` for parallel energy calculations.

## Future Enhancements

- [ ] Configuration preset saving/loading
- [ ] Option to disable Population Generator training
- [ ] Alternative mask lambda schedulers
- [ ] Automated workspace cleaning
- [ ] Command-line interface (CLI)
- [ ] Python library packaging
- [ ] MPI parallelization for large-scale runs

## Dependencies

Key dependencies (see requirements.txt):
- **torch** - Neural network Population Generator
- **numpy** - Numerical computations
- **pandas** - Data handling
- **matplotlib** - Visualization
- **tqdm** - Progress bars

## License
[here](LICENSE)

## Author
Jan Wolski
