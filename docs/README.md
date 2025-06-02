# Documentation Site Setup and Maintenance

This document outlines how to set up the MkDocs documentation site locally and
how to maintain it. The site is built using [MkDocs](https://www.mkdocs.org/)
with the [mkdocstrings](https://mkdocstrings.github.io/) plugin to generate
documentation from Python docstrings.

## One-Time Setup (or after cloning)

1. **Ensure Poetry is installed:** If you don't have Poetry, follow the
   installation instructions on the
   [Poetry website](https://python-poetry.org/docs/#installation).
2. **Install project dependencies:** Navigate to the project root directory (the
   one containing `pyproject.toml`) and run:
   ```bash
   poetry install --with dev
   ```
   This command installs all project dependencies, including those required for
   building the documentation (MkDocs, mkdocstrings, etc.), as defined in
   `pyproject.toml`.

## Building and Serving Locally

1. **Navigate to the project root directory.**
2. **Serve the documentation site:** To start a local development server that
   auto-reloads when changes are detected, run:
   ```bash
   poetry run mkdocs serve
   ```
   This will typically make the site available at `http://127.0.0.1:8000/`. The
   command should be run from the project root directory.
3. **Build the static site:** To generate the static HTML files (e.g., for
   deployment), run:
   ```bash
   poetry run mkdocs build
   ```
   The output will be placed in the `site/` directory by default. This command
   should also be run from the project root.

## How Docstrings are Collected

- The documentation for Python modules and functions is generated from their
  docstrings.
- We follow the
  [Google Python Style Docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings).
- The `mkdocstrings` plugin is configured in `docs/mkdocs.yml`. It looks for
  Python modules within the `smart_control/` directory.
- The main page that pulls in all the API documentation is `docs/api.md`. It
  uses a `::: smart_control` block to tell `mkdocstrings` to document the entire
  `smart_control` package recursively.

## Maintaining the Documentation

- **Keep docstrings updated:** When you add or modify code in the
  `smart_control` package, ensure the docstrings are clear, accurate, and follow
  the Google style.
- **Navigation:** To add new top-level pages or change the navigation structure,
  edit the `nav` section in `docs/mkdocs.yml`.
- **Content Pages:** For new static content pages (like this README or
  tutorials), create a new Markdown file in the `docs/` directory and add it to
  the `nav` in `mkdocs.yml`.
- **Configuration:** Advanced configuration for `mkdocstrings` or the site theme
  can be done in `docs/mkdocs.yml`.
- **Dependencies:** If you need new MkDocs plugins, add them to the
  `[tool.poetry.group.dev.dependencies]` in `pyproject.toml` and run
  `poetry lock && poetry install`. Then, configure the plugin in
  `docs/mkdocs.yml`.

## GitHub Actions Deployment

A GitHub Actions workflow is set up in `.github/workflows/deploy-docs.yml`. This
workflow automatically builds and deploys the documentation to GitHub Pages
whenever changes are pushed to the `main` branch.
