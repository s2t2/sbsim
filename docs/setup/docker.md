# Docker Setup Guide

To get the repository set up on non-Linux
environments, you can use the pre-configured Docker environment ("Linux/amd64")
specified by the "Dockerfile".

First, install [Docker Desktop](https://www.docker.com/products/docker-desktop/).

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

To access Jupyter notebooks, visit
[http://localhost:8888](http://localhost:8888) in the browser.

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
