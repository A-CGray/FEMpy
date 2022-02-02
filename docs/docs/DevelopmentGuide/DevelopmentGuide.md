# Contributors Guide

## Code formatting and Linting

The following checks will be run on any PR using [pre-commit](pre-commit.com) before it can be merged into the main branch:

- Black code formatter, with 120 column line width and target python versions 3.7-3.10
- Flake8 linter, using the [MDO Lab's configuration file](https://raw.githubusercontent.com/mdolab/.github/master/.flake8)
- `check-yaml` - checks yaml files for parseable syntax.
- `check-json` - checks json files for parseable syntax.
- `check-added-large-files` - prevents giant files from being committed.
- `mixed-line-ending` - replaces or checks mixed line ending.
- `check-merge-conflict` - checks for files that contain merge conflict strings.
- `debug-statements` - checks for debugger imports and py37+ `breakpoint()` calls in python source.

If you want to contribute to FEMpy please install FEMpy in editable mode and setup the pre-commit hooks to ensure consistent code formatting.
The full installation process is then:

```shell
# Install FEMpy in editable mode
pip install -e .[dev]

# Download the MDO Lab's flake8 configuration
wget https://raw.githubusercontent.com/mdolab/.github/master/.flake8

# Install the pre-commit hooks
pre-commit install
```

If the `wget` command doesn't work on your OS, simply [download the file here](https://raw.githubusercontent.com/mdolab/.github/master/.flake8) and place it in the root directory of your FEMpy repository.
