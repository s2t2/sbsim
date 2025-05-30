# Setup Guide

This document provides instructions for getting the repository set up for local
development. By default, we use Linux OS, however we are also providing a
"Dockerfile" to facilitate running the code on non-Linux (Mac and Windows)
systems.

## Linux OS

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

## Docker (Alternative Setup)

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
