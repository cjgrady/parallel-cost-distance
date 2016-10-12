"""
@summary: Tests single tile Dijkstra (parallel) module
@author: CJ Grady
"""
from hashlib import md5
import numpy as np
import os
from random import randint
import shutil
from tempfile import NamedTemporaryFile, gettempdir

from slr.common.costFunctions import seaLevelRiseCostFn

from slr.singleTile.parallelDijkstra import SingleTileParallelDijkstraLCP

from tests.helpers.testConstants import (L_COST_RASTER, L_INPUT_RASTER)

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

# .............................................................................
def _getTemporaryDirectory(r):
   """
   @summary: Creates a temporary directory for testing
   """
   n = md5(str(r)).hexdigest()
   d = os.path.join(gettempdir(), n)
   os.mkdir(d)
   return d

# .............................................................................
def test_calculate_large_evenSteps_4workers():
   """
   @summary: Tests that calculations work for a single tile (large)
   """
   # We just need a name, not a file
   tf = NamedTemporaryFile(delete=True)
   cAryFn = tf.name
   tf.close()
   myInstance = SingleTileParallelDijkstraLCP(L_INPUT_RASTER, cAryFn, 
                              seaLevelRiseCostFn)
   myInstance.setMaxWorkers(4)
   myInstance.setStepSize(.50)
   myInstance.findSourceCells()
   myInstance.calculate()
   
   computedArray = np.loadtxt(cAryFn, dtype=int, comments='', skiprows=6)
   costArray = np.loadtxt(L_COST_RASTER, dtype=int, comments='', skiprows=6)
   
   # Delete temp file
   os.remove(cAryFn)
   
   assert np.array_equal(computedArray, costArray)
   
# .............................................................................
def test_calculate_large_evenSteps_16workers():
   """
   @summary: Tests that calculations work for a single tile (large)
   """
   # We just need a name, not a file
   tf = NamedTemporaryFile(delete=True)
   cAryFn = tf.name
   tf.close()
   myInstance = SingleTileParallelDijkstraLCP(L_INPUT_RASTER, cAryFn, 
                              seaLevelRiseCostFn)
   myInstance.setMaxWorkers(16)
   myInstance.setStepSize(.50)
   myInstance.findSourceCells()
   myInstance.calculate()
   
   computedArray = np.loadtxt(cAryFn, dtype=int, comments='', skiprows=6)
   costArray = np.loadtxt(L_COST_RASTER, dtype=int, comments='', skiprows=6)
   
   # Delete temp file
   os.remove(cAryFn)
   
   assert np.array_equal(computedArray, costArray)
   
# .............................................................................
def test_calculate_large_unevenSteps_4workers():
   """
   @summary: Tests that calculations work for a single tile (large)
   """
   # We just need a name, not a file
   tf = NamedTemporaryFile(delete=True)
   cAryFn = tf.name
   tf.close()
   myInstance = SingleTileParallelDijkstraLCP(L_INPUT_RASTER, cAryFn, 
                              seaLevelRiseCostFn)
   myInstance.setMaxWorkers(4)
   myInstance.setStepSize(.33)
   myInstance.findSourceCells()
   myInstance.calculate()
   
   computedArray = np.loadtxt(cAryFn, dtype=int, comments='', skiprows=6)
   costArray = np.loadtxt(L_COST_RASTER, dtype=int, comments='', skiprows=6)
   
   # Delete temp file
   os.remove(cAryFn)
   
   assert np.array_equal(computedArray, costArray)
   
# .............................................................................
def test_calculate_large_unevenSteps_16workers():
   """
   @summary: Tests that calculations work for a single tile (large)
   """
   # We just need a name, not a file
   tf = NamedTemporaryFile(delete=True)
   cAryFn = tf.name
   tf.close()
   myInstance = SingleTileParallelDijkstraLCP(L_INPUT_RASTER, cAryFn, 
                              seaLevelRiseCostFn)
   myInstance.setMaxWorkers(16)
   myInstance.setStepSize(.33)
   myInstance.findSourceCells()
   myInstance.calculate()
   
   computedArray = np.loadtxt(cAryFn, comments='', skiprows=6, dtype=int)
   costArray = np.loadtxt(L_COST_RASTER, comments='', skiprows=6, dtype=int)
   
   # Delete temp file
   os.remove(cAryFn)
   
   print "Computed array shape:", computedArray.shape
   print "Cost array shape:", costArray.shape
   print computedArray
   print costArray
   print np.where(computedArray != costArray)[0]
   assert np.array_equal(computedArray, costArray)

# .............................................................................
def test_check_result_even_steps():
   """
   @summary: Test with a simple surface to verify result
   """
   inF = NamedTemporaryFile(delete=False)
   inFn = inF.name
   inF.write("ncols   40\n")
   inF.write("nrows   40\n")
   inF.write("xllcorner   0\n")
   inF.write("yllcorner   0\n")
   inF.write("cellsize   0.5\n")
   inF.write("NODATA_value   -999\n")
   for line in TEST_SURFACE:
      inF.write("{0}\n".format(' '.join([str(i) for i in line])))
   inF.close()
   
   outF = NamedTemporaryFile(delete=True)
   outFn = outF.name
   outF.close()
   
   myInstance = SingleTileParallelDijkstraLCP(inFn, outFn, seaLevelRiseCostFn)
   myInstance.setMaxWorkers(46)
   myInstance.setStepSize(.5)
   myInstance.findSourceCells()
   myInstance.calculate()
   
   computedArray = np.loadtxt(outFn, comments='', skiprows=6, dtype=int)
   costArray = np.array(COST_SURFACE, dtype=int)
   
   os.remove(inFn)
   os.remove(outFn)
   
   print computedArray.tolist()
   print costArray.tolist()
   assert np.array_equal(computedArray, costArray)
   
   # Check sides
   outDir = _getTemporaryDirectory(str(randint(0, 999)))
   
   
   taskId = 'myTest'
   myInstance.writeChangedVectors(outDir, ts=0.5, taskId=taskId)

   # Load arrays
   computedTop = np.load(os.path.join(outDir, "%s-toTop.npy" % taskId))
   computedBottom = np.load(os.path.join(outDir, "%s-toBottom.npy" % taskId))
   computedLeft = np.load(os.path.join(outDir, "%s-toLeft.npy" % taskId))
   computedRight = np.load(os.path.join(outDir, "%s-toRight.npy" % taskId))
   
   # Remove outputs
   shutil.rmtree(outDir)

   # Check top
   costTop = np.array(COST_SURFACE[0], dtype=int)
   assert np.array_equal(computedTop, costTop)
   
   # Check bottom
   costBottom = np.array(COST_SURFACE[-1], dtype=int)
   assert np.array_equal(computedBottom, costBottom)
   
   # Check left
   costLeft = np.array([l[0] for l in COST_SURFACE], dtype=int)
   assert np.array_equal(computedLeft, costLeft)
   
   # Check right
   costRight = np.array([l[-1] for l in COST_SURFACE], dtype=int)
   assert np.array_equal(computedRight, costRight)
   
# .............................................................................
def test_check_result_uneven_steps():
   """
   @summary: Test with a simple surface to verify result
   """
   inF = NamedTemporaryFile(delete=False)
   inFn = inF.name
   inF.write("ncols   40\n")
   inF.write("nrows   40\n")
   inF.write("xllcorner   0\n")
   inF.write("yllcorner   0\n")
   inF.write("cellsize   0.5\n")
   inF.write("NODATA_value   -999\n")
   for line in TEST_SURFACE:
      inF.write("{0}\n".format(' '.join([str(i) for i in line])))
   inF.close()
   
   outF = NamedTemporaryFile(delete=True)
   outFn = outF.name
   outF.close()
   
   myInstance = SingleTileParallelDijkstraLCP(inFn, outFn, seaLevelRiseCostFn)
   myInstance.setMaxWorkers(46)
   myInstance.setStepSize(.3)
   myInstance.findSourceCells()
   myInstance.calculate()
   
   computedArray = np.loadtxt(outFn, comments='', skiprows=6, dtype=int)
   costArray = np.array(COST_SURFACE, dtype=int)
   
   os.remove(inFn)
   os.remove(outFn)
   
   assert np.array_equal(computedArray, costArray)
   
   # Check sides
   outDir = _getTemporaryDirectory(str(randint(0, 999)))
   
   
   taskId = 'myTest'
   myInstance.writeChangedVectors(outDir, ts=20.0, taskId=taskId)

   # Load arrays
   computedTop = np.load(os.path.join(outDir, "%s-toTop.npy" % taskId))
   computedBottom = np.load(os.path.join(outDir, "%s-toBottom.npy" % taskId))
   computedLeft = np.load(os.path.join(outDir, "%s-toLeft.npy" % taskId))
   computedRight = np.load(os.path.join(outDir, "%s-toRight.npy" % taskId))
   
   # Remove outputs
   shutil.rmtree(outDir)

   # Check top
   costTop = np.array(COST_SURFACE[0], dtype=int)
   assert np.array_equal(computedTop, costTop)
   
   # Check bottom
   costBottom = np.array(COST_SURFACE[-1], dtype=int)
   assert np.array_equal(computedBottom, costBottom)
   
   # Check left
   costLeft = np.array([l[0] for l in COST_SURFACE], dtype=int)
   assert np.array_equal(computedLeft, costLeft)
   
   # Check right
   costRight = np.array([l[-1] for l in COST_SURFACE], dtype=int)
   assert np.array_equal(computedRight, costRight)
   
# .............................................................................
def test_split_tiles_flow():
   """
   @summary: Make sure that water flows from one tile to another correctly
   """
   t1F = NamedTemporaryFile(delete=False)
   tile1Fn = t1F.name
   t1F.write("ncols   20\n")
   t1F.write("nrows   20\n")
   t1F.write("xllcorner   0\n")
   t1F.write("yllcorner   0\n")
   t1F.write("cellsize   0.5\n")
   t1F.write("NODATA_value   -999\n")
   for line in TEST_SURFACE[:20]:
      t1F.write("{0}\n".format(' '.join([str(i) for i in line[:20]])))
   t1F.close()
   
   t2F = NamedTemporaryFile(delete=False)
   tile2Fn = t2F.name
   t2F.write("ncols   20\n")
   t2F.write("nrows   20\n")
   t2F.write("xllcorner   0\n")
   t2F.write("yllcorner   0\n")
   t2F.write("cellsize   0.5\n")
   t2F.write("NODATA_value   -999\n")
   for line in TEST_SURFACE[:20]:
      t2F.write("{0}\n".format(' '.join([str(i) for i in line[20:]])))
   t2F.close()

   # Set up output directory
   outDir = _getTemporaryDirectory(str(randint(0, 999)))

   # Check that first tile was written correctly
   inTile1 = np.loadtxt(tile1Fn, comments='', skiprows=6, dtype=int)
   checkTile1 = np.array([[i for i in line[:20]] for line in TEST_SURFACE[:20]], dtype=int)
   assert np.array_equal(inTile1, checkTile1)
   
   # Check that the second tile was written correctly
   inTile2 = np.loadtxt(tile2Fn, comments='', skiprows=6, dtype=int)
   checkTile2 = np.array([[i for i in line[20:]] for line in TEST_SURFACE[:20]], dtype=int)
   assert np.array_equal(inTile2, checkTile2)
   
   # Set up cost files
   c1F = NamedTemporaryFile(delete=True)
   cost1Fn = c1F.name
   c1F.close() 
   
   c2F = NamedTemporaryFile(delete=True)
   cost2Fn = c2F.name
   c2F.close() 
   
   # Compute tile 1
   inst1 = SingleTileParallelDijkstraLCP(tile1Fn, cost1Fn, seaLevelRiseCostFn)
   inst1.setMaxWorkers(50)
   inst1.setStepSize(.5)
   inst1.findSourceCells()
   assert len(inst1.sourceCells) == 14
   inst1.calculate()
   
   # Check output vectors for tile 1
   task1Id = 'task1'
   inst1.writeChangedVectors(outDir, ts=10.0, taskId=task1Id)
   
   # Assume that we are using the COST_SURFACE upper-left quadrant and this was 
   #    the first task run

   # Load task 1arrays
   task1Top = np.load(os.path.join(outDir, "%s-toTop.npy" % task1Id))
   task1Bottom = np.load(os.path.join(outDir, "%s-toBottom.npy" % task1Id))
   task1Left = np.load(os.path.join(outDir, "%s-toLeft.npy" % task1Id))
   task1Right = np.load(os.path.join(outDir, "%s-toRight.npy" % task1Id))
   
   # Check task 1 outputs
   # Top should be: [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
   assert np.array_equal(task1Top, 
            np.array(
               [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
               dtype=int))
   
   # Bottom should be: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9]
   assert np.array_equal(task1Bottom, 
            np.array(
               [1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9], 
               dtype=int))
   
   # Left should be: [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
   assert np.array_equal(task1Left, 
            np.array(
               [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
               dtype=int))
   
   # Right should be: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9]
   assert np.array_equal(task1Right, 
            np.array(
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9], 
               dtype=int))
   
   
   # Compute tile 2
   inst2 = SingleTileParallelDijkstraLCP(tile2Fn, cost2Fn, seaLevelRiseCostFn)
   inst2.setMaxWorkers(50)
   inst2.setStepSize(.5)
   inst2.findSourceCells()
   assert len(inst2.sourceCells) == 0 # No source cells in tile
   inst2.calculate()
   
   # Check output vectors for tile 2
   task2Id = 'task2'
   inst2.writeChangedVectors(outDir, ts=10.0, taskId=task2Id)
   
   # Assume that we are using the COST_SURFACE upper-right quadrant and this was 
   #    the first task run

   # There are no source cells in this quadrant, so we shouldn't have vectors
   assert not os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task2Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task2Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task2Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task2Id))
   
   # Check that cost surface for tile 2 is all no data
   assert np.array_equal(np.loadtxt(cost2Fn, dtype=int, comments='', skiprows=6),
               -9999 * np.ones((20, 20), dtype=int))
   

   # Check tile 2 with source vector from task 1
   inst3 = SingleTileParallelDijkstraLCP(tile2Fn, cost2Fn, seaLevelRiseCostFn)
   inst3.setMaxWorkers(50)
   inst3.setStepSize(.5)
   inst3.addSourceVector(task1Right, 0) # Source vector from tile 1 (to the left)
   assert len(inst3.sourceCells) == 20 # All from vector
   inst3.calculate()

   # Check output vectors, should create one for every direction
   task3Id = 'task3'
   inst3.writeChangedVectors(outDir, ts=10.0, taskId=task3Id)
   
   # Load task 1arrays
   task3Top = np.load(os.path.join(outDir, "%s-toTop.npy" % task3Id))
   task3Bottom = np.load(os.path.join(outDir, "%s-toBottom.npy" % task3Id))
   task3Left = np.load(os.path.join(outDir, "%s-toLeft.npy" % task3Id))
   task3Right = np.load(os.path.join(outDir, "%s-toRight.npy" % task3Id))
   
   # Top should be: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
   assert np.array_equal(task3Top, 
            np.array(
               [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 
               dtype=int))
   
   # Bottom should be: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
   assert np.array_equal(task3Bottom, 
            np.array(
               [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 
               dtype=int))
   
   # Left should be: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9, 2, 2, 2, 2, 2]
   assert np.array_equal(task3Left, 
            np.array(
               [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9, 2, 2, 2, 2, 2], 
               dtype=int))
   
   # Right should be: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
   assert np.array_equal(task3Right, 
            np.array(
               [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 
               dtype=int))
   
   
   # Flow back from tile 2 to tile 1
   inst4 = SingleTileParallelDijkstraLCP(tile1Fn, cost1Fn, seaLevelRiseCostFn)
   inst4.setMaxWorkers(50)
   inst4.setStepSize(.5)
   inst4.addSourceVector(task3Left, 2) # Add sources from tile 2 (from the right)
   assert len(inst4.sourceCells) == 5 # There are 5 changed cells
   inst4.calculate()
   
   # Check output vectors for tile 1
   task4Id = 'task4'
   inst4.writeChangedVectors(outDir, ts=10.0, taskId=task4Id)
   
   # Only the bottom vector should have changed, the right side did, but it was 
   #    due to source cells

   assert not os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task4Id))
   assert os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task4Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task4Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task4Id))
   
   
   # Check final cost surfaces
   tile1CostFinal = np.loadtxt(cost1Fn, dtype=int, comments='', skiprows=6)
   assert np.array_equal(tile1CostFinal,
            np.array([[i for i in line[:20]] for line in COST_SURFACE[:20]], 
                     dtype=int))
   tile2CostFinal = np.loadtxt(cost2Fn, dtype=int, comments='', skiprows=6)
   
   cmpTile2 = np.array([[i for i in line[20:]] for line in COST_SURFACE[:20]], 
                     dtype=int)
   
   assert np.array_equal(tile2CostFinal, cmpTile2)
   
   # Clean up
   os.remove(tile1Fn)
   os.remove(tile2Fn)
   os.remove(cost1Fn)
   os.remove(cost2Fn)
   shutil.rmtree(outDir)
   
# .............................................................................
def test_quad_tiles_flow():
   """
   @summary: Make sure works correctly with four tiles
   """
   # Create input tiles
   t1F = NamedTemporaryFile(delete=False)
   tile1Fn = t1F.name
   t1F.write("ncols   20\n")
   t1F.write("nrows   20\n")
   t1F.write("xllcorner   0\n")
   t1F.write("yllcorner   0\n")
   t1F.write("cellsize   0.5\n")
   t1F.write("NODATA_value   -999\n")
   for line in TEST_SURFACE[:20]:
      t1F.write("{0}\n".format(' '.join([str(i) for i in line[:20]])))
   t1F.close()
   
   t2F = NamedTemporaryFile(delete=False)
   tile2Fn = t2F.name
   t2F.write("ncols   20\n")
   t2F.write("nrows   20\n")
   t2F.write("xllcorner   0\n")
   t2F.write("yllcorner   0\n")
   t2F.write("cellsize   0.5\n")
   t2F.write("NODATA_value   -999\n")
   for line in TEST_SURFACE[:20]:
      t2F.write("{0}\n".format(' '.join([str(i) for i in line[20:]])))
   t2F.close()

   t3F = NamedTemporaryFile(delete=False)
   tile3Fn = t3F.name
   t3F.write("ncols   20\n")
   t3F.write("nrows   20\n")
   t3F.write("xllcorner   0\n")
   t3F.write("yllcorner   -10\n")
   t3F.write("cellsize   0.5\n")
   t3F.write("NODATA_value   -999\n")
   for line in TEST_SURFACE[20:]:
      t3F.write("{0}\n".format(' '.join([str(i) for i in line[:20]])))
   t3F.close()
   
   t4F = NamedTemporaryFile(delete=False)
   tile4Fn = t4F.name
   t4F.write("ncols   20\n")
   t4F.write("nrows   20\n")
   t4F.write("xllcorner   10\n")
   t4F.write("yllcorner   -10\n")
   t4F.write("cellsize   0.5\n")
   t4F.write("NODATA_value   -999\n")
   for line in TEST_SURFACE[20:]:
      t4F.write("{0}\n".format(' '.join([str(i) for i in line[20:]])))
   t4F.close()


   # Set up output directory
   outDir = _getTemporaryDirectory(str(randint(0, 999)))

   # Check that first tile was written correctly
   inTile1 = np.loadtxt(tile1Fn, comments='', skiprows=6, dtype=int)
   checkTile1 = np.array([[i for i in line[:20]] for line in TEST_SURFACE[:20]], dtype=int)
   assert np.array_equal(inTile1, checkTile1)
   
   # Check that the second tile was written correctly
   inTile2 = np.loadtxt(tile2Fn, comments='', skiprows=6, dtype=int)
   checkTile2 = np.array([[i for i in line[20:]] for line in TEST_SURFACE[:20]], dtype=int)
   assert np.array_equal(inTile2, checkTile2)
   
   # Check the third
   inTile3 = np.loadtxt(tile3Fn, comments='', skiprows=6, dtype=int)
   checkTile3 = np.array([[i for i in line[:20]] for line in TEST_SURFACE[20:]], dtype=int)
   assert np.array_equal(inTile3, checkTile3)
   
   # Check that the fourth tile was written correctly
   inTile4 = np.loadtxt(tile4Fn, comments='', skiprows=6, dtype=int)
   checkTile4 = np.array([[i for i in line[20:]] for line in TEST_SURFACE[20:]], dtype=int)
   assert np.array_equal(inTile4, checkTile4)
   
   # Set up cost files
   c1F = NamedTemporaryFile(delete=True)
   cost1Fn = c1F.name
   c1F.close() 
   
   c2F = NamedTemporaryFile(delete=True)
   cost2Fn = c2F.name
   c2F.close() 

   c3F = NamedTemporaryFile(delete=True)
   cost3Fn = c3F.name
   c3F.close() 
   
   c4F = NamedTemporaryFile(delete=True)
   cost4Fn = c4F.name
   c4F.close() 

   # Tile 1 - task 1
   #   - check source cells
   #   - should generate vectors
   inst1 = SingleTileParallelDijkstraLCP(tile1Fn, cost1Fn, seaLevelRiseCostFn)
   inst1.setMaxWorkers(50)
   inst1.setStepSize(.5)
   inst1.findSourceCells()
   assert len(inst1.sourceCells) == 14
   inst1.calculate()
   
   task1Id = 'task1'
   inst1.writeChangedVectors(outDir, ts=10.0, taskId=task1Id)
   assert os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task1Id))
   assert os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task1Id))
   assert os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task1Id))
   assert os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task1Id))
   # Load vectors we will use later
   task1Right = np.load(os.path.join(outDir, "%s-toRight.npy" % task1Id))
   task1Bottom = np.load(os.path.join(outDir, "%s-toBottom.npy" % task1Id))
   
   # Tile 2 - task 2
   #   - no source cells
   #   - no vectors
   #   - no data cost surface
   inst2 = SingleTileParallelDijkstraLCP(tile2Fn, cost2Fn, seaLevelRiseCostFn)
   inst2.setMaxWorkers(50)
   inst2.setStepSize(.5)
   inst2.findSourceCells()
   assert len(inst2.sourceCells) == 0
   inst2.calculate()
   
   task2Id = 'task2'
   inst2.writeChangedVectors(outDir, ts=10.0, taskId=task2Id)
   assert not os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task2Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task2Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task2Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task2Id))
   
   assert np.array_equal(np.loadtxt(cost2Fn, dtype=int, comments='', skiprows=6),
               -9999 * np.ones((20, 20), dtype=int))

   # Tile 3 - task 3
   #   - no source cells
   #   - no vectors
   #   - no data cost surface
   inst3 = SingleTileParallelDijkstraLCP(tile3Fn, cost3Fn, seaLevelRiseCostFn)
   inst3.setMaxWorkers(50)
   inst3.setStepSize(.5)
   inst3.findSourceCells()
   assert len(inst3.sourceCells) == 0
   inst3.calculate()
   
   task3Id = 'task3'
   inst3.writeChangedVectors(outDir, ts=10.0, taskId=task3Id)
   assert not os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task3Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task3Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task3Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task3Id))
   
   assert np.array_equal(np.loadtxt(cost2Fn, dtype=int, comments='', skiprows=6),
               -9999 * np.ones((20, 20), dtype=int))

   # Tile 4 - task 4
   #  - check source cells
   #  - should generate vectors
   inst4 = SingleTileParallelDijkstraLCP(tile4Fn, cost4Fn, seaLevelRiseCostFn)
   inst4.setMaxWorkers(50)
   inst4.setStepSize(.5)
   inst4.findSourceCells()
   assert len(inst4.sourceCells) == 10
   inst4.calculate()
   
   task4Id = 'task4'
   inst4.writeChangedVectors(outDir, ts=10.0, taskId=task4Id)
   assert os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task4Id))
   assert os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task4Id))
   assert os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task4Id))
   assert os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task4Id))
   # We'll use these later
   task4Top = np.load(os.path.join(outDir, "%s-toTop.npy" % task4Id))
   task4Left = np.load(os.path.join(outDir, "%s-toLeft.npy" % task4Id))
   
   # Tile 2 from 1 and 4 - task 5
   #  - some number of source cells
   #  - generates vectors
   inst5 = SingleTileParallelDijkstraLCP(tile2Fn, cost2Fn, seaLevelRiseCostFn)
   inst5.setMaxWorkers(50)
   inst5.setStepSize(.5)
   inst5.addSourceVector(task1Right, 0)
   assert len(inst5.sourceCells) == 20
   inst5.addSourceVector(task4Top, 3)
   print inst5.sourceCells
   assert len(inst5.sourceCells) == 40 
   inst5.calculate()
   
   task5Id = 'task5'
   inst5.writeChangedVectors(outDir, ts=10.0, taskId=task5Id)
   assert os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task5Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task5Id))
   assert os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task5Id))
   assert os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task5Id))

   task5Left = np.load(os.path.join(outDir, "%s-toLeft.npy" % task5Id))
   task5Bottom = np.load(os.path.join(outDir, "%s-toBottom.npy" % task5Id))

   # Tile 3 from 1 and 4 - task 6
   #  - source cells
   #  - vectors
   inst6 = SingleTileParallelDijkstraLCP(tile3Fn, cost3Fn, seaLevelRiseCostFn)
   inst6.setMaxWorkers(50)
   inst6.setStepSize(.5)
   inst6.addSourceVector(task1Bottom, 1)
   assert len(inst6.sourceCells) == 20
   inst6.addSourceVector(task4Left, 4)
   assert len(inst6.sourceCells) == 39 
   inst6.calculate()
   
   task6Id = 'task6'
   inst6.writeChangedVectors(outDir, ts=10.0, taskId=task6Id)
   assert os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task6Id))
   assert os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task6Id))
   assert os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task6Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task6Id))

   task6Top = np.load(os.path.join(outDir, "%s-toTop.npy" % task6Id))
   
   # Tile 1 from 2 and 3 - task 7
   #  - source cells
   #  - vectors
   inst7 = SingleTileParallelDijkstraLCP(tile1Fn, cost1Fn, seaLevelRiseCostFn)
   inst7.setMaxWorkers(50)
   inst7.setStepSize(.5)
   inst7.addSourceVector(task5Left, 2)
   assert len(inst7.sourceCells) == 5
   inst7.addSourceVector(task6Top, 3)
   assert len(inst7.sourceCells) == 8 
   inst7.calculate()
   
   task7Id = 'task7'
   inst7.writeChangedVectors(outDir, ts=10.0, taskId=task7Id)
   assert not os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task7Id))
   assert os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task7Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task7Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task7Id))
   task7Bottom = np.load(os.path.join(outDir, "%s-toBottom.npy" % task7Id))
   
   # Tile 4 from 2 - task 8
   #  - no source cells
   #  - no vectors
   inst8 = SingleTileParallelDijkstraLCP(tile4Fn, cost4Fn, seaLevelRiseCostFn)
   inst8.setMaxWorkers(50)
   inst8.setStepSize(.5)
   inst8.addSourceVector(task5Bottom, 1)
   assert len(inst8.sourceCells) == 0
   inst8.calculate()
   
   task8Id = 'task8'
   inst8.writeChangedVectors(outDir, ts=10.0, taskId=task8Id)
   assert not os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task8Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task8Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task8Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task8Id))
   
   # Tile 3 from 1 - task 9
   #  - source cells
   #  - no vectors
   inst9 = SingleTileParallelDijkstraLCP(tile3Fn, cost3Fn, seaLevelRiseCostFn)
   inst9.setMaxWorkers(50)
   inst9.setStepSize(.5)
   inst9.addSourceVector(task7Bottom, 1)
   assert len(inst9.sourceCells) == 0
   inst9.calculate()
   
   task9Id = 'task9'
   inst9.writeChangedVectors(outDir, ts=10.0, taskId=task9Id)
   assert not os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task9Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task9Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task9Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task9Id))
   
   # Check final
   tile1CostFinal = np.loadtxt(cost1Fn, dtype=int, comments='', skiprows=6)
   assert np.array_equal(tile1CostFinal,
            np.array([[i for i in line[:20]] for line in COST_SURFACE[:20]], 
                     dtype=int))

   tile2CostFinal = np.loadtxt(cost2Fn, dtype=int, comments='', skiprows=6)
   assert np.array_equal(tile2CostFinal,
            np.array([[i for i in line[20:]] for line in COST_SURFACE[:20]], 
                     dtype=int))

   tile3CostFinal = np.loadtxt(cost3Fn, dtype=int, comments='', skiprows=6)
   assert np.array_equal(tile3CostFinal,
            np.array([[i for i in line[:20]] for line in COST_SURFACE[20:]], 
                     dtype=int))

   tile4CostFinal = np.loadtxt(cost4Fn, dtype=int, comments='', skiprows=6)
   assert np.array_equal(tile4CostFinal,
            np.array([[i for i in line[20:]] for line in COST_SURFACE[20:]], 
                     dtype=int))
   
   # Clean up
   os.remove(tile1Fn)
   os.remove(tile2Fn)
   os.remove(tile3Fn)
   os.remove(tile4Fn)
   os.remove(cost1Fn)
   os.remove(cost2Fn)
   os.remove(cost3Fn)
   os.remove(cost4Fn)
   shutil.rmtree(outDir)
