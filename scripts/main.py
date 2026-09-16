'''
@file    :   main.py
@create date : 2025-07-22 16:17:49
@modify date 2026-06-19 10:29:20
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    Convenience wrapper script for BioAirMet training and testing.
    Delegates to bioairmet.main which contains the actual logic.
    Use this file directly (python scripts/main.py ...) when running from
    the project source tree without installing the package.
]
'''

import os
import sys

# Allow running without installing the package by adding src/ to sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, 'src'))

from bioairmet.main import main  # noqa: E402 (import after path setup)

if __name__ == '__main__':
    main()
