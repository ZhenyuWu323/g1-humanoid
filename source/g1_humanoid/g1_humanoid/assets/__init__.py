# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing asset and sensor configurations."""

import os

# Define the base path for this assets package
ASSETS_DIR = os.path.abspath(os.path.dirname(__file__))
"""Path to the assets directory."""

ASSETS_DATA_DIR = os.path.join(ASSETS_DIR, "data")
"""Path to the data directory inside assets."""

# Check if data directory exists
if not os.path.exists(ASSETS_DATA_DIR):
    raise FileNotFoundError(f"Assets data directory not found: {ASSETS_DATA_DIR}")

# Import other submodules or specific assets configurations
try:
    from .g1_inspire import *  # Replace with actual submodules if needed
    from .g1_with_plate import *
    from .g1_humanoid import *
except ImportError:
    pass  # If no submodules exist, this can be safely ignored
