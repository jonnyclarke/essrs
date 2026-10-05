# Examples

In this file I provide direction on how to build the package and run the examples provided.
This is currently a work in progress hence only the python notebooks are documented.

## Python Notebooks

## Setup & Run Python Notebooks

Assuming Rust and Python are already installed:

# 1. Clone the repository
```bash
git clone https://github.com/jonnyclarke/essrs
cd essrs
```

# 2. Create and activate the uv virtual environment
```bash
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

# 3. Sync Python dependencies
```bash
uv sync
```

# 4. Build/install the Rust components
```bash
uv run maturin develop --release
```

# 5. Run the project
Navigate to `examples_py` directory and open the notebooks from there.
