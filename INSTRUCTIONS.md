# Docker Setup and Usage Instructions

This document provides instructions on how to build the Docker image for this project and how to run containers for development, including running Jupyter notebooks, executing scripts, and running tests.

## Prerequisites

*   Docker installed on your system.

## Building the Docker Image

1.  **Navigate to the project root directory**:
    Open your terminal and change to the directory containing `Dockerfile-2` and your project files.

2.  **Build the image**:
    Run the following command to build the Docker image. Replace `your-image-name` with a name you prefer (e.g., `smart_control_app`).

    ```bash
    docker build -t your-image-name -f Dockerfile-2 .
    ```

    This process might take some time, especially on the first build, as it downloads the base image and installs all dependencies.

## Running the Docker Container

Once the image is built, you can run a container based on it. The `ENTRYPOINT` for this image is set to `poetry run`, which means any command you provide will be executed within the project's Poetry-managed virtual environment.

### General Usage

The general command to run a container is:

```bash
docker run -it --rm \
    -p 8888:8888 \
    -v $(pwd):/workspace \
    your-image-name <command_and_args>
```

*   `-it`: Runs the container in interactive mode with a TTY.
*   `--rm`: Automatically removes the container when it exits.
*   `-p 8888:8888`: Maps port 8888 on your host to port 8888 in the container (useful for Jupyter). You can change this if needed.
*   `-v $(pwd):/workspace`: Mounts the current directory on your host to `/workspace` in the container. This allows you to edit files locally and see changes reflected in the container immediately.
*   `your-image-name`: The name you gave your image during the build step.
*   `<command_and_args>`: The command and its arguments you want to run inside the container (e.g., `python script.py`, `pytest`, `jupyter notebook ...`). These will be automatically prefixed by `poetry run`.

### 1. Running Jupyter Notebooks

To start a Jupyter Notebook server that you can access from your host machine (this is the default command if none is provided):

```bash
docker run -it --rm \
    -p 8888:8888 \
    -v $(pwd):/workspace \
    your-image-name
```
This uses the default `CMD` specified in the `Dockerfile-2`: `jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token='' --no-browser --notebook-dir=/workspace/smart_control/notebooks`.

Then, open your browser and navigate to `http://localhost:8888`. The notebooks are located in the `smart_control/notebooks` directory.

If you need to override the default command for Jupyter (e.g., to change the notebook directory):
```bash
docker run -it --rm \
    -p 8888:8888 \
    -v $(pwd):/workspace \
    your-image-name jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token='' --no-browser --notebook-dir=/workspace/some_other_notebook_dir
```

### 2. Executing Python Scripts

To run a Python script located in your project (e.g., a training script):

```bash
docker run -it --rm \
    -v $(pwd):/workspace \
    your-image-name python smart_control/reinforcement_learning/scripts/train.py --your --script --arguments
```
Replace `smart_control/reinforcement_learning/scripts/train.py` with the path to your script and add any necessary arguments. The command `python ...` will be executed as `poetry run python ...`.

### 3. Running Tests (pytest)

To run all tests using pytest:

```bash
docker run -it --rm \
    -v $(pwd):/workspace \
    your-image-name pytest smart_control
```
The command `pytest smart_control` will be executed as `poetry run pytest smart_control`.

*   If you want to run specific tests, you can modify the `pytest` command:
    ```bash
    docker run -it --rm \
        -v $(pwd):/workspace \
        your-image-name pytest smart_control/environment/environment_test.py
    ```

### 4. Accessing a Shell (Bash) in the Container

To get an interactive bash shell inside the container for debugging or manual operations:

```bash
docker run -it --rm \
    -v $(pwd):/workspace \
    your-image-name bash
```
This will drop you into a shell (executed as `poetry run bash`) in the `/workspace` directory, with the Poetry environment effectively active for subsequent commands run within that shell.

## Notes

*   **Protobuf Files**: The `.proto` files are compiled during the image build process. If you modify them, you will need to rebuild the image.
*   **Dependencies**: If you change `pyproject.toml` or `poetry.lock` (e.g., add or update dependencies), you should rebuild the image to include these changes.
*   **Permissions**: If you encounter permission issues with files created in the mounted volume, ensure the user inside the Docker container has the correct permissions or run the container with user mapping (`--user $(id -u):$(id -g)`). However, the current setup runs as root inside the container, which simplifies some things but is less secure for shared environments.
```
