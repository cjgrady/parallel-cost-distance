"""
@summary: Test the Work Queue multitile module
@author: CJ Grady
@version: 1.0
@status: alpha
"""
import glob
from hashlib import md5
import numpy as np
import os
from random import randint
import shutil
import tempfile
import traceback

from slr.multiTile.wqMultiTile import MultiTileWqParallelDijkstraLCP

from tests.helpers.testConstants import (EVEN_TILE_DIR, EVEN_TILE_COSTS_DIR, 
                                        UNEVEN_TILE_DIR, UNEVEN_TILE_COSTS_DIR)
                           

NUM_WORKERS = 1 # Travis core limit
TEST_SURFACE = [
[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9, 9, 9, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0]
]

COST_SURFACE = [
[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9, 9, 9, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0],
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0]
]

TEST_SURFACE2 = [
 [ 3,  7, 12, 16, 16, 17, 15, 11, 11, 11,   15, 15, 13, 12,  8,  7,  5,  5,  5,  5], 
 [ 6,  7, 13, 17, 18, 18, 16, 13, 11, 13,   16, 15, 13, 12,  8,  7,  5,  5,  5,  5], 
 [ 3, 11, 13, 17, 18, 18, 16, 11, 11, 13,   16, 15, 13, 12, 10,  7,  6,  6,  6,  5], 
 [ 3,  4, 14, 15, 16, 16, 15, 10, 11, 15,   15, 14, 19, 11, 11,  9,  6,  7,  6,  5], 
 [ 3,  6, 12, 13, 11, 11,  9,  5,  6,  8,   11, 16, 22, 16, 14, 10,  6,  6,  6,  5], 
 [ 2,  8,  8, 13,  9,  9,  5,  6,  8,  8,   16, 15, 16, 15, 15,  7,  7,  7,  7,  7], 
 [ 5, 11, 10,  9,  7,  4,  5,  4,  5,  7,   15, 17, 17, 15, 12,  7,  7,  7,  7,  7], 
 [ 7, 11, 13, 13, 11,  7,  4,  4,  5, 12,   14, 16, 18, 13, 13, 12,  9,  7,  7,  7], 
 [ 8, 10, 11, 12, 10,  5,  7,  4,  4, 11,   13, 18, 16, 18, 16, 14, 10, 11, 10, 10], 
 [ 9, 11, 11, 11,  9,  3,  3,  8,  0,  7,    9, 11, 16, 16, 16, 14,  9, 10, 11, 11], 
 
 [ 9, 11, 11, 11,  9,  3,  8, 10,  3,  6,   10, 11, 13, 15, 17, 15, 10, 11, 11, 12], 
 [ 9, 11, 12, 11,  9,  3,  3,  3,  3,  9,   10, 11, 13, 15, 14, 13,  8,  8,  9, 10], 
 [ 9, 11, 11, 11,  9,  3,  3,  3,  9,  9,    8, 11, 16, 15, 14, 13,  8,  9, 10, 10], 
 [ 9, 11, 11, 12, 10,  4,  4,  4,  8, 10,    9, 12, 16, 16, 14, 13,  8,  9, 10, 11], 
 [ 8, 11, 12, 11,  9,  4,  8,  8,  9, 14,   13, 10, 14, 14, 15, 14,  8,  9, 10, 10], 
 [ 8, 12, 12, 12,  9,  9,  9,  9, 10, 14,   13, 17, 15, 15, 13, 11,  7,  7,  7,  8], 
 [ 8, 11, 12, 11,  8,  9,  9, 11, 12, 16,   13, 14, 14, 14, 13, 11,  6,  6,  6,  7], 
 [ 4,  9, 10, 10,  9,  9, 10, 12, 15, 17,   13, 13, 14, 13, 12, 10,  5,  5,  6,  6], 
 [ 4,  7,  8,  7,  4,  8,  8, 10, 11, 10,   10, 12, 10, 12, 10,  5,  5,  4,  5,  5], 
 [ 4,  4,  4,  4,  4,  8,  8,  8,  9,  8,   10, 10,  7,  8, 10,  5,  5,  0,  4,  4]
]


COST_SURFACE2 = [
 [ 9,  9, 12, 16, 16, 17, 15, 11, 11, 11,   15, 15, 13, 12, 10, 10, 10, 10, 10, 10], 
 [ 9,  9, 13, 17, 18, 18, 16, 13, 11, 13,   16, 15, 13, 12, 10, 10, 10, 10, 10, 10], 
 [ 9, 11, 13, 17, 18, 18, 16, 11, 11, 13,   16, 15, 13, 12, 10, 10, 10, 10, 10, 10], 
 [ 9,  9, 14, 15, 16, 16, 15, 10, 11, 15,   15, 15, 19, 11, 11, 10, 10, 10, 10, 10], 
 [ 9,  9, 12, 13, 11, 11,  9,  6,  6,  8,   11, 16, 22, 16, 14, 10, 10, 10, 10, 10], 
 [ 9,  9,  9, 13,  9,  9,  5,  6,  8,  8,   16, 16, 16, 15, 15, 10, 10, 10, 10, 10], 
 [ 9, 11, 10,  9,  7,  5,  5,  4,  5,  7,   15, 17, 17, 15, 12, 10, 10, 10, 10, 10], 
 [ 9, 11, 13, 13, 11,  7,  4,  4,  5, 12,   14, 16, 18, 13, 13, 12, 10, 10, 10, 10], 
 [ 9, 10, 11, 12, 10,  5,  7,  4,  4, 11,   13, 18, 16, 18, 16, 14, 10, 11, 10, 10], 
 [ 9, 11, 11, 11,  9,  3,  3,  8,  0,  7,    9, 11, 16, 16, 16, 14, 10, 10, 11, 11], 
 
 [ 9, 11, 11, 11,  9,  3,  8, 10,  3,  6,   10, 11, 13, 15, 17, 15, 10, 11, 11, 12], 
 [ 9, 11, 12, 11,  9,  3,  3,  3,  3,  9,   10, 11, 13, 15, 14, 13,  8,  8,  9, 10], 
 [ 9, 11, 11, 11,  9,  3,  3,  3,  9,  9,    9, 11, 16, 15, 14, 13,  8,  9, 10, 10], 
 [ 9, 11, 11, 12, 10,  4,  4,  4,  8, 10,    9, 12, 16, 16, 14, 13,  8,  9, 10, 11], 
 [ 9, 11, 12, 11,  9,  4,  8,  8,  9, 14,   13, 12, 14, 14, 15, 14,  8,  9, 10, 10], 
 [ 9, 12, 12, 12,  9,  9,  9,  9, 10, 14,   13, 17, 15, 15, 13, 11,  7,  7,  7,  8], 
 [ 9, 11, 12, 11,  9,  9,  9, 11, 12, 16,   13, 14, 14, 14, 13, 11,  6,  6,  6,  7], 
 [ 9,  9, 10, 10,  9,  9, 10, 12, 15, 17,   13, 13, 14, 13, 12, 10,  5,  5,  6,  6], 
 [ 9,  9,  9,  9,  9,  9,  9, 10, 11, 10,   10, 12, 10, 12, 10,  5,  5,  4,  5,  5], 
 [ 9,  9,  9,  9,  9,  9,  9,  9,  9,  9,   10, 10, 10, 10, 10,  5,  5,  0,  4,  4]
]


# .............................................................................
class TestWqParallelLCP(object):
   """
   @summary: This class tests the Work Queue Parallel Dijkstra implementation
   """
   # ............................
   def setup(self):
      """
      @summary: Set up the test
      """
      self.cDir = self._getTemporaryDirectory(randint(0, 100000))
      self.oDir = self._getTemporaryDirectory(randint(0, 100000))
      # We just want a name, so let it be deleted
      tf = tempfile.NamedTemporaryFile(delete=True)
      self.summaryFn = tf.name
      tf.close()
   
   # ............................
   def teardown(self):
      """
      @summary: Tear down the test
      """
      # Delete cost directory
      try:
         shutil.rmtree(self.cDir)
      except:
         pass
      
      # Delete output directory
      try:
         shutil.rmtree(self.oDir)
      except:
         pass
      
      # Delete summary file
      try:
         shutil.rmtree(self.summaryFn)
      except:
         pass
   
   # ............................
   def _checkOutputs(self, cmpDir):
      """
      @summary: Check that the outputs are what we expect from the compare 
                   directory
      @param cmpDir: This is a test data directory created by splitting the 
                        cost surface generated with the single surface mode
      @return: Boolean indicating if all of the files match
      """
      # Need to check that the files match
      assert len(glob.glob(os.path.join(self.cDir, "*.asc"))) == len(glob.glob(os.path.join(cmpDir, "*.asc")))
      
      # Need to check that the contents match
      for fn in glob.iglob(os.path.join(self.cDir, "*.asc")):
         n = os.path.basename(fn)
         print "Checking:", n
         a1 = np.loadtxt(fn, comments='', skiprows=6, dtype=int)
         a2 = np.loadtxt(os.path.join(cmpDir, n), comments='', skiprows=6, dtype=int)
         if not np.array_equal(a1, a2):
            print a1.tolist()
            print
            print a2.tolist()
            return False
      return True
         
   # ............................
   def _getTemporaryDirectory(self, r):
      """
      @summary: Creates a temporary directory for testing
      """
      n = md5(str(r)).hexdigest()
      d = os.path.join(tempfile.gettempdir(), n)
      os.mkdir(d)
      return d
   
   # ............................
   #def test_even_tiles(self):
   #   """
   #   @summary: Test that the outputs are what we expect when using even tile
   #                sizes
   #   """
   #   # Create instance
   #   myInstance = MultiTileWqParallelDijkstraLCP(EVEN_TILE_DIR, 
   #                        self.cDir, self.oDir, 10.0, .2, 
   #                        summaryFn=self.summaryFn)
   #   # Run
   #   print "Starting workers"
   #   myInstance.startWorkers(NUM_WORKERS)
   #   try:
   #      myInstance.calculate()
   #   except Exception, e:
   #      print "stop workers (error)"
   #      myInstance.stopWorkers()
   #      raise e
   #   
   #   # Only on success
   #   print "Stopping workers"
   #   myInstance.stopWorkers()
   #   print "Done"
   #   
   #   # Compare outputs
   #   assert self._checkOutputs(EVEN_TILE_COSTS_DIR)
   
   # ............................
   #def test_uneven_tiles(self):
   #   """
   #   @summary: Test that the outputs are what we expect when using uneven tile
   #                sizes
   #   """
   #   # Create instance
   #   myInstance = MultiTileWqParallelDijkstraLCP(UNEVEN_TILE_DIR, 
   #                        self.cDir, self.oDir, 20.0, .2, 
   #                        summaryFn=self.summaryFn)
   #   # Run
   #   print "Starting workers"
   #   myInstance.startWorkers(NUM_WORKERS)
   #   try:
   #      myInstance.calculate()
   #   except Exception, e:
   #      traceback.print_exc()
   #      print "stop workers (error)"
   #      myInstance.stopWorkers()
   #      raise e
   #   
   #   # Only on success
   #   print "Stopping workers"
   #   myInstance.stopWorkers()
   #   print "Done"
   #   
   #   # Compare outputs
   #   assert self._checkOutputs(UNEVEN_TILE_COSTS_DIR)
   
   # ............................
   def test_small(self):
      """
      @summary: This is the same test data used for explicitly performing this
                   procedure in single tile mode
      """
      inDir = self._getTemporaryDirectory(randint(0, 100000))
      realCostsDir = self._getTemporaryDirectory(randint(0, 100000))
      
      t1Fn = os.path.join(inDir, 'grid0.0-10.0-10.0-20.0.asc')
      t2Fn = os.path.join(inDir, 'grid10.0-10.0-20.0-20.0.asc')
      t3Fn = os.path.join(inDir, 'grid0.0-0.0-10.0-10.0.asc')
      t4Fn = os.path.join(inDir, 'grid10.0-0.0-10.0-10.0.asc')
      c1Fn = os.path.join(realCostsDir, 'grid0.0-10.0-10.0-20.0.asc')
      c2Fn = os.path.join(realCostsDir, 'grid10.0-10.0-20.0-20.0.asc')
      c3Fn = os.path.join(realCostsDir, 'grid0.0-0.0-10.0-10.0.asc')
      c4Fn = os.path.join(realCostsDir, 'grid10.0-0.0-10.0-10.0.asc')
      
      
      # Write out tiles
      with open(t1Fn, 'w') as t1F:
         t1F.write("ncols   20\n")
         t1F.write("nrows   20\n")
         t1F.write("xllcorner   0\n")
         t1F.write("yllcorner   10\n")
         t1F.write("cellsize   0.5\n")
         t1F.write("NODATA_value   -999\n")
         for line in TEST_SURFACE[:20]:
            t1F.write("{0}\n".format(' '.join([str(i) for i in line[:20]])))
   
      with open(t2Fn, 'w') as t2F:
         t2F.write("ncols   20\n")
         t2F.write("nrows   20\n")
         t2F.write("xllcorner   10\n")
         t2F.write("yllcorner   10\n")
         t2F.write("cellsize   0.5\n")
         t2F.write("NODATA_value   -999\n")
         for line in TEST_SURFACE[:20]:
            t2F.write("{0}\n".format(' '.join([str(i) for i in line[20:]])))

      with open(t3Fn, 'w') as t3F:
         t3F.write("ncols   20\n")
         t3F.write("nrows   20\n")
         t3F.write("xllcorner   0\n")
         t3F.write("yllcorner   0\n")
         t3F.write("cellsize   0.5\n")
         t3F.write("NODATA_value   -999\n")
         for line in TEST_SURFACE[20:]:
            t3F.write("{0}\n".format(' '.join([str(i) for i in line[:20]])))
   
      with open(t4Fn, 'w') as t4F:
         t4F.write("ncols   20\n")
         t4F.write("nrows   20\n")
         t4F.write("xllcorner   10\n")
         t4F.write("yllcorner   0\n")
         t4F.write("cellsize   0.5\n")
         t4F.write("NODATA_value   -999\n")
         for line in TEST_SURFACE[20:]:
            t4F.write("{0}\n".format(' '.join([str(i) for i in line[20:]])))

      # Write out cost tiles
      with open(c1Fn, 'w') as c1F:
         c1F.write("ncols   20\n")
         c1F.write("nrows   20\n")
         c1F.write("xllcorner   0\n")
         c1F.write("yllcorner   10\n")
         c1F.write("cellsize   0.5\n")
         c1F.write("NODATA_value   -999\n")
         for line in COST_SURFACE[:20]:
            c1F.write("{0}\n".format(' '.join([str(i) for i in line[:20]])))
   
      with open(c2Fn, 'w') as c2F:
         c2F.write("ncols   20\n")
         c2F.write("nrows   20\n")
         c2F.write("xllcorner   10\n")
         c2F.write("yllcorner   10\n")
         c2F.write("cellsize   0.5\n")
         c2F.write("NODATA_value   -999\n")
         for line in COST_SURFACE[:20]:
            c2F.write("{0}\n".format(' '.join([str(i) for i in line[20:]])))

      with open(c3Fn, 'w') as c3F:
         c3F.write("ncols   20\n")
         c3F.write("nrows   20\n")
         c3F.write("xllcorner   0\n")
         c3F.write("yllcorner   0\n")
         c3F.write("cellsize   0.5\n")
         c3F.write("NODATA_value   -999\n")
         for line in COST_SURFACE[20:]:
            c3F.write("{0}\n".format(' '.join([str(i) for i in line[:20]])))
   
      with open(c4Fn, 'w') as c4F:
         c4F.write("ncols   20\n")
         c4F.write("nrows   20\n")
         c4F.write("xllcorner   10\n")
         c4F.write("yllcorner   0\n")
         c4F.write("cellsize   0.5\n")
         c4F.write("NODATA_value   -999\n")
         for line in COST_SURFACE[20:]:
            c4F.write("{0}\n".format(' '.join([str(i) for i in line[20:]])))


      myInstance = MultiTileWqParallelDijkstraLCP(inDir, 
                           self.cDir, self.oDir, 10.0, 1.0, 
                           summaryFn=self.summaryFn)
      # Run
      print "Starting workers"
      myInstance.startWorkers(NUM_WORKERS)
      try:
         myInstance.calculate()
      except Exception, e:
         traceback.print_exc()
         print "stop workers (error)"
         myInstance.stopWorkers()
         raise e
      
      # Only on success
      print "Stopping workers"
      myInstance.stopWorkers()
      print "Done"
      
      # Compare outputs
      assert self._checkOutputs(realCostsDir)

      # Clean up
      shutil.rmtree(realCostsDir)
      shutil.rmtree(inDir)

   # ............................
   def test_small2(self):
      """
      @summary: This is the same test data used for explicitly performing this
                   procedure in single tile mode
      """
      inDir = self._getTemporaryDirectory(randint(0, 100000))
      realCostsDir = self._getTemporaryDirectory(randint(0, 100000))
      
      t1Fn = os.path.join(inDir, 'grid0.0-10.0-10.0-20.0.asc')
      t2Fn = os.path.join(inDir, 'grid10.0-10.0-20.0-20.0.asc')
      t3Fn = os.path.join(inDir, 'grid0.0-0.0-10.0-10.0.asc')
      t4Fn = os.path.join(inDir, 'grid10.0-0.0-20.0-10.0.asc')
      c1Fn = os.path.join(realCostsDir, 'grid0.0-10.0-10.0-20.0.asc')
      c2Fn = os.path.join(realCostsDir, 'grid10.0-10.0-20.0-20.0.asc')
      c3Fn = os.path.join(realCostsDir, 'grid0.0-0.0-10.0-10.0.asc')
      c4Fn = os.path.join(realCostsDir, 'grid10.0-0.0-20.0-10.0.asc')
      
      
      # Write out tiles
      with open(t1Fn, 'w') as t1F:
         t1F.write("ncols   10\n")
         t1F.write("nrows   10\n")
         t1F.write("xllcorner   0\n")
         t1F.write("yllcorner   10\n")
         t1F.write("cellsize   1\n")
         t1F.write("NODATA_value   -999\n")
         for line in TEST_SURFACE2[:10]:
            t1F.write("{0}\n".format(' '.join([str(i) for i in line[:10]])))
   
      with open(t2Fn, 'w') as t2F:
         t2F.write("ncols   10\n")
         t2F.write("nrows   10\n")
         t2F.write("xllcorner   10\n")
         t2F.write("yllcorner   10\n")
         t2F.write("cellsize   1\n")
         t2F.write("NODATA_value   -999\n")
         for line in TEST_SURFACE2[:10]:
            t2F.write("{0}\n".format(' '.join([str(i) for i in line[10:]])))

      with open(t3Fn, 'w') as t3F:
         t3F.write("ncols   10\n")
         t3F.write("nrows   10\n")
         t3F.write("xllcorner   0\n")
         t3F.write("yllcorner   0\n")
         t3F.write("cellsize   1\n")
         t3F.write("NODATA_value   -999\n")
         for line in TEST_SURFACE2[10:]:
            t3F.write("{0}\n".format(' '.join([str(i) for i in line[:10]])))
   
      with open(t4Fn, 'w') as t4F:
         t4F.write("ncols   10\n")
         t4F.write("nrows   10\n")
         t4F.write("xllcorner   10\n")
         t4F.write("yllcorner   0\n")
         t4F.write("cellsize   1\n")
         t4F.write("NODATA_value   -999\n")
         for line in TEST_SURFACE2[10:]:
            t4F.write("{0}\n".format(' '.join([str(i) for i in line[10:]])))

      # Write out cost tiles
      with open(c1Fn, 'w') as c1F:
         c1F.write("ncols   10\n")
         c1F.write("nrows   10\n")
         c1F.write("xllcorner   0\n")
         c1F.write("yllcorner   10\n")
         c1F.write("cellsize   1\n")
         c1F.write("NODATA_value   -999\n")
         for line in COST_SURFACE2[:10]:
            c1F.write("{0}\n".format(' '.join([str(i) for i in line[:10]])))
   
      with open(c2Fn, 'w') as c2F:
         c2F.write("ncols   10\n")
         c2F.write("nrows   10\n")
         c2F.write("xllcorner   10\n")
         c2F.write("yllcorner   10\n")
         c2F.write("cellsize   1\n")
         c2F.write("NODATA_value   -999\n")
         for line in COST_SURFACE2[:10]:
            c2F.write("{0}\n".format(' '.join([str(i) for i in line[10:]])))

      with open(c3Fn, 'w') as c3F:
         c3F.write("ncols   10\n")
         c3F.write("nrows   10\n")
         c3F.write("xllcorner   0\n")
         c3F.write("yllcorner   0\n")
         c3F.write("cellsize   1\n")
         c3F.write("NODATA_value   -999\n")
         for line in COST_SURFACE2[10:]:
            c3F.write("{0}\n".format(' '.join([str(i) for i in line[:10]])))
   
      with open(c4Fn, 'w') as c4F:
         c4F.write("ncols   10\n")
         c4F.write("nrows   10\n")
         c4F.write("xllcorner   10\n")
         c4F.write("yllcorner   0\n")
         c4F.write("cellsize   1\n")
         c4F.write("NODATA_value   -999\n")
         for line in COST_SURFACE2[10:]:
            c4F.write("{0}\n".format(' '.join([str(i) for i in line[10:]])))


      myInstance = MultiTileWqParallelDijkstraLCP(inDir, 
                           self.cDir, self.oDir, 10.0, 1.0, 
                           summaryFn=self.summaryFn)
      # Run
      print "Starting workers"
      myInstance.startWorkers(NUM_WORKERS)
      try:
         myInstance.calculate()
      except Exception, e:
         traceback.print_exc()
         print "stop workers (error)"
         myInstance.stopWorkers()
         raise e
      
      # Only on success
      print "Stopping workers"
      myInstance.stopWorkers()
      print "Done"
      
      # Compare outputs
      assert self._checkOutputs(realCostsDir)

      # Clean up
      shutil.rmtree(realCostsDir)
      shutil.rmtree(inDir)
