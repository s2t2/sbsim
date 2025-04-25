# Google Smart Buildings Control

This repository accompanies Goldfeder, J., Sipple, J., Real-World Data and Calibrated
Simulation Suite for Offline Training of Reinforcement Learning Agents to Optimize
Energy and Emission in Office Buildings, currently under review at Neurips 2024,
and builds off of Goldfeder, J., Sipple, J., (2023).
[A Lightweight Calibrated Simulation Enabling Efficient Offline Learning for Optimal Control of Real Buildings](https://dl.acm.org/doi/10.1145/3600100.3625682),
BuildSys ’23, November 15–16, 2023, Istanbul, Turkey

## Getting Started

The best place to jump in is the Soft Actor Critic Demo notebook,
available in notebooks/SAC_Demo.ipynb

This will walk you through:

1. Creating an RL (gym compatible) environment

2. Visualizing the env

3. Training an agent using the [Tensorflow Agents Library](https://www.tensorflow.org/agents)

Before you run this notebook, make sure to go through the setup instructions below to ensure the notebook runs successfully.

## Setup

Follow these steps to setup locally before you run the `notebooks/SAC_Demo.ipynb` notebook. Note: this will only work on linux, as some libraries are not supported by other operating systems.

1. Clone the repository

2. Ensure you have `protoc` and `ffmpeg` installed, as well as `python >=3.10.12 and <3.12`. You can install these running `sudo apt install -y protobuf-compiler` and `sudo apt install -y ffmpeg`

3. Create a virtual environment by running `python -m venv .venv`. Activate the environment `source .venv/bin/activate`. Then, install poetry with `pip install poetry`

4. Install the dependencies by running `poetry install --with dev`

5. Build the `.proto` files at `smart_control/proto`into python files by running `cd smart_control/proto && protoc --python_out=. smart_control_building.proto smart_control_normalization.proto smart_control_reward.proto && cd ../..`

6. Modify the value of `VIDEO_PATH_ROOT` at `smart_control/simulator/constants.py`. This is the path where simulation videos will be stored

7. Now in the `notebooks/SAC_Demo.ipynb` notebook, modify the values of `data_path`, `metrics_path`, `output_data_path` and `root_dir`. In particular, `data_path` should point to the `sim_config.gin` file at `smart_control/configs/sim_config.gin`

8. Now you are ready to run the `notebooks/SAC_Demo.ipynb` notebook

## Real World Data

In addition to our calibrated simulator, we released 6 years of data on 3 buildings, for further calibration, and to use, in conjunction with the simulator, for training and evaluating RL models. The dataset is part of [Tensorflow Datasets](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/datasets/smart_buildings_dataset)

## Contributing

We welcome your contributions to this repository!

All open source contributors will need to sign Google's [Contributor License Agreement (CLA)](https://cla.developers.google.com/).

Contributors are encouraged to consult the sections below for more information about code documenation, testing, and formatting.

### Documentation

We encourage you to document your code using docstrings. Specifically we use the [Google Docstring Guidelines](https://google.github.io/styleguide/pyguide.html#381-docstrings) outlined in the Google Python Style Guide.

### Testing

We encourage you to add tests to ensure your code is working as expected.

Running tests:

```sh
# run all tests:
pytest

# disable warnings:
pytest --disable-pytest-warnings

# run specific test files:
pytest --disable-pytest-warnings path/to/your/test.py

# run specific tests:
pytest --disable-pytest-warnings -k your_test_name_here
```

### Linting

#### Style Formatting

We are using the [`pyink` formatter](https://github.com/google/pyink) to format code according to [Google Python Style Guidelines](https://google.github.io/styleguide/pyguide.html). The formatter will automatically update files inplace.

The formatter will run automatically as a pre-commit hook (see "Pre-commit Hooks" section below for more information and setup instructions).

Additionally, for contributors using the VS Code text editor, we have configured a VS Code workspace settings file to run the formatter whenever a file is saved. NOTE: this requires the [`ms-python.black-formatter` extension](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) for VS Code.

If you would like to run the formatter manually:

```sh
# format all the files:
pyink .

# format a specific file or directory:
pyink /path/to/file/or/dir
```

If you would like to perform a dry run:

```sh
# check if a file would be changed:
pyink . --check

# see what changes would be made:
pyink . --diff
```

If you would like to prevent certain lines of code from being formatted (for example to leave a long line as-is), it is possible to [ignore formatting](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#ignoring-sections) by:

  + addding a trailing comment of `# fmt: skip` to the right of the line, or
  + wrapping multiple lines of code between `# fmt: off` and `# fmt: on` comments

#### Import Sorting

We are using [`isort`](https://pycqa.github.io/isort/) to control the sort order of import statements, specifically grouping the "smart_control" local module imports separately in their own section below the package imports.

The import sorter will run automatically as a pre-commit hook (see "Pre-commit Hooks" section below for more information and setup instructions).

Additionally, for contributors using the VS Code text editor, we have configured a VS Code workspace settings file to run the import sorter whenever a file is saved. NOTE: this requires the [`ms-python.isort` extension](https://marketplace.visualstudio.com/items?itemName=ms-python.isort) for VS Code.

If you would like to run the import sorter manually:

```sh
# sort all the files:
isort .

# sort a specific file:
isort /path/to/file.py

# sort with verbose outputs (helpful for troubleshooting):
isort -v .
```


### Pre-commit Hooks

We are using pre-commit hooks to perform code formatting and import sorting. These actions will take place on each commit.

To enable the pre-commit hooks, you must perform a one-time setup by running `pre-commit install`. This will update ".git/hooks/pre-commit".


If you would like to run the pre-commit hooks without making a commit:

```sh
# run for all files:
pre-commit run --all-files

# run for a specific set of file(s):
pre-commit run --files path/to/my_file.py path/to/other_file.py
```

If you run into issues and need to clear the cache:

```sh
pre-commit clean
```

## [License](LICENSE)
