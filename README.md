# Google Smart Buildings Control

This repository accompanies Goldfeder, J., Sipple, J., Real-World Data and
Calibrated Simulation Suite for Offline Training of Reinforcement Learning
Agents to Optimize Energy and Emission in Office Buildings, currently under
review at Neurips 2024, and builds off of Goldfeder, J., Sipple, J., (2023).
[A Lightweight Calibrated Simulation Enabling Efficient Offline Learning for Optimal Control of Real Buildings](https://dl.acm.org/doi/10.1145/3600100.3625682),
BuildSys '23, November 15–16, 2023, Istanbul, Turkey

## Getting Started

The best place to jump in is the Soft Actor Critic Demo notebook, available in
notebooks/SAC_Demo.ipynb

This will walk you through:

1. Creating an RL (gym compatible) environment

2. Visualizing the env

3. Training an agent using the
   [Tensorflow Agents Library](https://www.tensorflow.org/agents)

Before you run this notebook, make sure to go through the setup instructions
below to ensure the notebook runs successfully.

## Setup

Follow these steps to setup locally before you run the
`notebooks/SAC_Demo.ipynb` notebook. Note: this will only work on linux, as some
libraries are not supported by other operating systems.

1. Clone the repository

2. Ensure you have `protoc` and `ffmpeg` installed, as well as
   `python >=3.10.12 and <3.12`. You can install these running
   `sudo apt install -y protobuf-compiler` and `sudo apt install -y ffmpeg`

3. Create a virtual environment by running `python -m venv .venv`. Activate the
   environment `source .venv/bin/activate`. Then, install poetry with
   `pip install poetry`

4. Install the dependencies by running `poetry install --with dev`

5. Build the `.proto` files at `smart_control/proto` into python files by
   running:

   ```bash
   cd smart_control/proto
   protoc --python_out=. smart_control_building.proto \
     smart_control_normalization.proto \
     smart_control_reward.proto
   cd ../..
   ```

6. By default, simulation videos are stored in the "simulator/videos" directory
   (which is ignored from version control). If you would like to customize this
   location, use the `SIM_VIDEOS_DIRPATH` environment variable. You can pass
   this environment variable at runtime, or create a local ".env" file and set
   your desired value there:

   ```bash
   # this is the ".env" file:

   SIM_VIDEOS_DIRPATH="/cns/oz-d/home/smart-buildings-control-team/smart-buildings/geometric_sim_videos/"
   ```

7. Now in the `notebooks/SAC_Demo.ipynb` notebook, modify the values of
   `data_path`, `metrics_path`, `output_data_path` and `root_dir`. In
   particular, `data_path` should point to the `sim_config.gin` file at
   `smart_control/configs/sim_config.gin`

8. Now you are ready to run the `notebooks/SAC_Demo.ipynb` notebook

## 🐳 Docker (Alternative Setup)

To avoid OS compatibility issues, use the pre-configured Docker environment
(Linux/amd64):

**Build:**

```bash
docker build -t sbsim-env .
```

Run the container:

```bash
docker run -it -p 8888:8888 -v $(pwd):/workspace sbsim-env
```

Access Jupyter:

Open http://localhost:8888

The container will copy the SBSim code into /workspace/sbsim on first run. Use
-v to persist changes.

## Real World Data

In addition to our calibrated simulator, we released 6 years of data on 3
buildings, for further calibration, and to use, in conjunction with the
simulator, for training and evaluating RL models. The dataset is part of
[Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/smart_buildings).

## Documentation

Here is an
[Unofficial Community-run Documentation Site](https://gitwyd.github.io/sbsim_documentation/)
containing more information about the project and the codebase.

## [Contributing](docs/contributing.md)

## [License](LICENSE)
