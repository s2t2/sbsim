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

1. Creating an RL (gym compatible) envronment

2. Visualizing the env

3. Training an agent using the [Tensorflow Agents Library](https://www.tensorflow.org/agents)

Before you run this notebook, make sure to go through the setup instructions below to ensure the notebook runs successfully.

## Setup

Follow these steps to setup locally before you run the `notebooks/SAC_Demo.ipynb` notebook. 

> NOTE: this will only work on Linux, as some libraries are not supported by other operating systems.

1. Clone the repository.

2. Ensure you have `protoc` and `ffmpeg` installed, as well as `python >=3.10.12 and <3.12`. You can install these running `sudo apt install -y protobuf-compiler` and `sudo apt install -y ffmpeg`.

3. Create a virtual environment by running `python -m venv .venv`. Activate the environment `source .venv/bin/activate`. Then, install poetry with `pip install poetry`.

> NOTE: on Google machines you may need to use `python3` instead of `python`.

> NOTE: on Google machines you may need to first install venv by running `apt install python3.12-venv` (prefixed with `sudo` as necessary).

4. Install the dependencies by running `poetry install --with dev`.

> NOTE: you may need to first use a Python version that is compatible with this project, by running `poetry env use 3.11` (for example if you want to use Python version 3.11).

> NOTE: on Google machines, you may need to first install a compatible Python version by running `pyenv install 3.11` (for example if you want to use Python version 3.11), and then `pyenv versions` to list the installed versions, and then `pyenv global 3.11.11` to use the specific version that was installed.

> NOTE: on Google machines you may need to first [install and configure pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation).

5. Build the `.proto` files at `smart_control/proto` into python files by running `cd smart_control/proto && protoc --python_out=. smart_control_building.proto smart_control_normalization.proto smart_control_reward.proto && cd ../..`.

6. Create a local `.env` file in the root directory of this repo, and specify the following environment variables (see example `.env` file contents below):

  + `VIDEO_PATH_ROOT`: used by `smart_control/simulator/constants.py`, this is the path where simulation videos will be stored.
  + `DATA_PATH`: used by demo notebooks, this should point to the directory in this repository where the `sim_config.gin` file is located (currently `smart_control/configs/resources/sb1`).
  + `METRICS_PATH`: used by demo notebooks
  + `OUTPUT_DATA_PATH`: used by demo notebooks
  + `ROOT_DIR`: used by demo notebooks


```sh
# example .env file ...

VIDEO_PATH_ROOT="/path/to/sbsim/geometric_sim_videos"

DATA_PATH="/path/to/sbsim/smart_control/configs/resources/sb1/"
METRICS_PATH="/path/to/sbsim/metrics" 
OUTPUT_DATA_PATH='/path/to/sbsim/smart_control/sb_colab_demo'
ROOT_DIR="/path/to/sbsim/root" 

# fall back to using Python instead of C extensions (for the wrapt package):
# WRAPT_DISABLE_EXTENSIONS="true"
```

7. Now you are ready to run the `notebooks/SAC_Demo.ipynb` notebook. 

> NOTE: if you would like to use jupyter to run the notebook, first create a kernel by running `poetry run python -m ipykernel install --user --name=sbsim-kernel`, then choose this kernel when running the notebook in VS Code. Otherwise you can run `poetry run jupyter notebook` to run the notebook with Jupyter.

## Real World Data

In addition to our calibrated simulator, we released 6 years of data on 3 buildings, for further calibration, and to use, in conjunction with the simulator, for training and evaluating RL models. The dataset is part of [Tensorflow Datasets](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/datasets/smart_buildings_dataset)
