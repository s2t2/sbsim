# Setup Guide

This document provides instructions for getting the repository set up for local
development. By default, we use Linux OS, as some dependencies are not supported
by other operating systems, however we are also providing a "Dockerfile" to
facilitate running the code on non-Linux (Mac and Windows) systems.

This project requires the following dependencies:

- Python (>=3.10.12 and \<3.12)
- [Protocol Buffer Compiler](https://grpc.io/docs/protoc-installation/)
  ("protoc")
- [FFmpeg](https://ffmpeg.org/)

## Repository Setup

Clone the repository, for example using an SSH approach:

```sh
git clone git@github.com:google/sbsim.git
cd sbsim
```

## Linux Package Installation

Install Linux package dependencies:

```sh
sudo apt install -y protobuf-compiler
sudo apt install -y ffmpeg
```

## Virtual Environment Setup

Create a virtual environment:

```sh
python -m venv .venv
```

Activate the environment:

```sh
source .venv/bin/activate
```

## Python Package Installation

Install poetry:

```sh
pip install poetry
```

Use poetry to install dependencies:

```sh
poetry install --with dev
```

## Protocol Buffer Compilation

Build the ".proto" files defined in the "smart_control/proto" directory into
Python files:

```bash
cd smart_control/proto

protoc --python_out=. smart_control_building.proto \
  smart_control_normalization.proto \
  smart_control_reward.proto

cd ../..
```

## Environment Variable Setup

By default, simulation videos are stored in the "simulator/videos" directory
(which is ignored from version control). If you would like to customize this
location, use the `SIM_VIDEOS_DIRPATH` environment variable.

You can pass environment variable(s) at runtime, or create a local ".env" file
and set your desired value(s) there:

```bash
# this is the ".env" file:

SIM_VIDEOS_DIRPATH="/cns/oz-d/home/smart-buildings-control-team/smart-buildings/geometric_sim_videos/"
```

## Notebook Setup

If you are running the Demo notebooks in the "notebook" directory, you must
modify the values of `data_path`, `metrics_path`, `output_data_path` and
`root_dir` in those notebooks. Specifically, the `data_path` should point to the
"sim_config.gin" file located at "smart_control/configs/sim_config.gin".

> NOTE: in the future we plan on refactoring notebook code to leverage the local
> module code and simplify this notebook setup experience. See
> [issue #83](https://github.com/google/sbsim/issues/83) (contributions
> welcome)!

<hr>

# Docker (Alternative Setup)

To avoid OS compatibility issues, and get the repository set up on non-Linux
environments, you can use the pre-configured Docker environment (Linux/amd64)
specified by the "Dockerfile".

Build the image:

```bash
docker build -t sbsim-env .
```

Run the container, in interactive mode, with open ports:

```bash
docker run -it -p 8888:8888 -v $(pwd):/workspace sbsim-env
```

> NOTE: the container will copy the repository into "/workspace/sbsim" on the
> first run. Use -v to persist changes.

To access Jupyter notebooks, visit http://localhost:8888 in the browser.

To run scripts or tests inside the actively running docker container:

```sh
# activate the virtual environment:
source /opt/venv/bin/activate

# navigate to the repository:
cd /workspace/sbsim

# running scripts:
python path/to/script.py

# running tests:
pytest
```

To stop the container:

```sh
docker stop sbsim-env
```

> NOTE: in the future we would like to further update these instructions and
> improve the Dockerfile. See
> [issue #80](https://github.com/google/sbsim/issues/80) (contributions
> welcome)!
