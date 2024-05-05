#!/bin/bash

set -euo pipefail

# Go to project root
cd "$(dirname "$(dirname "$0")")"
# Build Cython libraries
python setup.py build_ext --inplace