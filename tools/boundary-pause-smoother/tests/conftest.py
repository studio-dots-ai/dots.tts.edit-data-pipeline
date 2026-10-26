# This file isolates the test directory from the parent boundary-pause-smoother package.
# Without this, pytest tries to import boundary-pause-smoother/__init__.py which uses
# relative imports that fail outside the package context.
