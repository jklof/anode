# ANode
Node based realtime audio processing

## setup

Create the complete development environment with a single command
```bash
conda env create -f environment.yml
```

Activate the new environment
```bash
conda activate anode-dev
```
## compile and install
This command does two magic things:
 - Triggers scikit-build-core, which runs CMake to compile C++ code.
 - Installs Python package in a way that any changes to the .py
    files are immediately reflected without needing to reinstall.

```bash
pip install -e . -v
```

## run
```bash
python main.py
```

## test
Always run with the environment activated — activation puts `Library\bin`
on `PATH`, which native-library discovery depends on (without it, e.g.
`import soundfile` fails even though `libsndfile` is installed).
```bash
conda activate anode-dev
python -m pytest tests/ -q
```

## remove the environment
```bash
conda deactivate
conda env remove --name anode-dev
```
