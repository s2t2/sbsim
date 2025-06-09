#!/bin/bash
set -e

# Activate the virtual environment
source /workspace/.venv/bin/activate

# If the first argument is "jupyter", "python", "pytest", or "bash",
# then execute it. Otherwise, assume the user wants to run pytest.
if [ "$1" = "jupyter" ] || [ "$1" = "python" ] || [ "$1" = "pytest" ] || [ "$1" = "bash" ]; then
    exec "$@"
else
    # Default to running pytest if no specific command is given, or an unknown command.
    # This could also be changed to 'bash' to just open a shell.
    echo "Running tests..."
    exec pytest smart_control/tests
fi
