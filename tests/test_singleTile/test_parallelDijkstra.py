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
   print np.where(computedArray != costArray)
   for i in np.where(computedArray != costArray):
      print computedArray[i], costArray[i]
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
   outDir = _getTemporaryDirectory(str(randint(0, 99999)))
   
   
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
   outDir = _getTemporaryDirectory(str(randint(0, 99999)))
   
   
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
   outDir = _getTemporaryDirectory(str(randint(0, 99999)))

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
   t1F.write("yllcorner   10\n")
   t1F.write("cellsize   0.5\n")
   t1F.write("NODATA_value   -999\n")
   for line in TEST_SURFACE[:20]:
      t1F.write("{0}\n".format(' '.join([str(i) for i in line[:20]])))
   t1F.close()
   
   t2F = NamedTemporaryFile(delete=False)
   tile2Fn = t2F.name
   t2F.write("ncols   20\n")
   t2F.write("nrows   20\n")
   t2F.write("xllcorner   10\n")
   t2F.write("yllcorner   10\n")
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
   t3F.write("yllcorner   0\n")
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
   t4F.write("yllcorner   0\n")
   t4F.write("cellsize   0.5\n")
   t4F.write("NODATA_value   -999\n")
   for line in TEST_SURFACE[20:]:
      t4F.write("{0}\n".format(' '.join([str(i) for i in line[20:]])))
   t4F.close()


   # Set up output directory
   outDir = _getTemporaryDirectory(str(randint(0, 99999)))

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
   assert os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task5Id))
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
   inst6.addSourceVector(task4Left, 2)
   assert len(inst6.sourceCells) == 40 
   inst6.calculate()
   
   task6Id = 'task6'
   inst6.writeChangedVectors(outDir, ts=10.0, taskId=task6Id)
   assert os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task6Id))
   assert os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task6Id))
   assert os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task6Id))
   assert os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task6Id))

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

# .............................................................................
def test_quad_tiles_flow2():
   """
   @summary: Make sure works correctly with four tiles
   """
   # Create input tiles
   t1F = NamedTemporaryFile(delete=False)
   tile1Fn = t1F.name
   t1F.write("ncols   10\n")
   t1F.write("nrows   10\n")
   t1F.write("xllcorner   0\n")
   t1F.write("yllcorner   10\n")
   t1F.write("cellsize   1\n")
   t1F.write("NODATA_value   -999\n")
   for line in TEST_SURFACE2[:10]:
      t1F.write("{0}\n".format(' '.join([str(i) for i in line[:10]])))
   t1F.close()
   
   t2F = NamedTemporaryFile(delete=False)
   tile2Fn = t2F.name
   t2F.write("ncols   10\n")
   t2F.write("nrows   10\n")
   t2F.write("xllcorner   10\n")
   t2F.write("yllcorner   10\n")
   t2F.write("cellsize   1\n")
   t2F.write("NODATA_value   -999\n")
   for line in TEST_SURFACE2[:10]:
      t2F.write("{0}\n".format(' '.join([str(i) for i in line[10:]])))
   t2F.close()

   t3F = NamedTemporaryFile(delete=False)
   tile3Fn = t3F.name
   t3F.write("ncols   10\n")
   t3F.write("nrows   10\n")
   t3F.write("xllcorner   0\n")
   t3F.write("yllcorner   0\n")
   t3F.write("cellsize   1\n")
   t3F.write("NODATA_value   -999\n")
   for line in TEST_SURFACE2[10:]:
      t3F.write("{0}\n".format(' '.join([str(i) for i in line[:10]])))
   t3F.close()
   
   t4F = NamedTemporaryFile(delete=False)
   tile4Fn = t4F.name
   t4F.write("ncols   10\n")
   t4F.write("nrows   10\n")
   t4F.write("xllcorner   10\n")
   t4F.write("yllcorner   0\n")
   t4F.write("cellsize   1\n")
   t4F.write("NODATA_value   -999\n")
   for line in TEST_SURFACE2[10:]:
      t4F.write("{0}\n".format(' '.join([str(i) for i in line[10:]])))
   t4F.close()


   # Set up output directory
   outDir = _getTemporaryDirectory(str(randint(0, 99999)))

   # Check that first tile was written correctly
   inTile1 = np.loadtxt(tile1Fn, comments='', skiprows=6, dtype=int)
   checkTile1 = np.array([[i for i in line[:10]] for line in TEST_SURFACE2[:10]], dtype=int)
   assert np.array_equal(inTile1, checkTile1)
   
   # Check that the second tile was written correctly
   inTile2 = np.loadtxt(tile2Fn, comments='', skiprows=6, dtype=int)
   checkTile2 = np.array([[i for i in line[10:]] for line in TEST_SURFACE2[:10]], dtype=int)
   assert np.array_equal(inTile2, checkTile2)
   
   # Check the third
   inTile3 = np.loadtxt(tile3Fn, comments='', skiprows=6, dtype=int)
   checkTile3 = np.array([[i for i in line[:10]] for line in TEST_SURFACE2[10:]], dtype=int)
   assert np.array_equal(inTile3, checkTile3)
   
   # Check that the fourth tile was written correctly
   inTile4 = np.loadtxt(tile4Fn, comments='', skiprows=6, dtype=int)
   checkTile4 = np.array([[i for i in line[10:]] for line in TEST_SURFACE2[10:]], dtype=int)
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


   # Tasks

   # Tile 1 - task 1
   #   - look for source cells
   #   - should generate vectors
   inst1 = SingleTileParallelDijkstraLCP(tile1Fn, cost1Fn, seaLevelRiseCostFn)
   inst1.setMaxWorkers(50)
   inst1.setStepSize(0.5)
   inst1.findSourceCells()
   assert len(inst1.sourceCells) == 1
   inst1.calculate()
   
   task1Id = 'task1'
   inst1.writeChangedVectors(outDir, ts=1.0, taskId=task1Id)
   assert os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task1Id))
   assert os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task1Id))
   assert os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task1Id))
   assert os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task1Id))
   # Load vectors we will use later
   task1Top = np.load(os.path.join(outDir, "%s-toTop.npy" % task1Id))
   task1Left = np.load(os.path.join(outDir, "%s-toLeft.npy" % task1Id))
   task1Right = np.load(os.path.join(outDir, "%s-toRight.npy" % task1Id))
   task1Bottom = np.load(os.path.join(outDir, "%s-toBottom.npy" % task1Id))
   
   # Check tile 1 costs after task 1
   t1cost = np.loadtxt(cost1Fn, comments='', dtype=int, skiprows=6)
   t1check = np.array([
                       [10, 10, 12, 16, 16, 17, 15, 11, 11, 11], 
                       [10, 10, 13, 17, 18, 18, 16, 13, 11, 13], 
                       [10, 11, 13, 17, 18, 18, 16, 11, 11, 13], 
                       [10, 10, 14, 15, 16, 16, 15, 10, 11, 15], 
                       [10, 10, 12, 13, 11, 11,  9,  6,  6,  8], 
                       [10, 10, 10, 13,  9,  9,  5,  6,  8,  8], 
                       [10, 11, 10,  9,  7,  5,  5,  4,  5,  7], 
                       [10, 11, 13, 13, 11,  7,  4,  4,  5, 12], 
                       [10, 10, 11, 12, 10,  7,  7,  4,  4, 11], 
                       [10, 11, 11, 11,  9,  7,  7,  8,  0,  7]], dtype=int)
   assert np.array_equal(t1cost, t1check)
   t1cost = None
   t1check = None
   
   # Check vectors
   t1LeftCheck = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=int)
   t1RightCheck = np.array([11, 13, 13, 15, 8, 8, 7, 12, 11, 7], dtype=int)
   t1TopCheck = np.array([10, 10, 12, 16, 16, 17, 15, 11, 11, 11], dtype=int)
   t1BottomCheck = np.array([10, 11, 11, 11, 9, 7, 7, 8, 0, 7], dtype=int)
   
   assert np.array_equal(task1Top, t1TopCheck)
   assert np.array_equal(task1Bottom, t1BottomCheck)
   assert np.array_equal(task1Left, t1LeftCheck)
   assert np.array_equal(task1Right, t1RightCheck)
   
   
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
               -9999 * np.ones((10, 10), dtype=int))

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
               -9999 * np.ones((10, 10), dtype=int))

   # Tile 4 - task 4
   #  - check source cells
   #  - should generate vectors
   inst4 = SingleTileParallelDijkstraLCP(tile4Fn, cost4Fn, seaLevelRiseCostFn)
   inst4.setMaxWorkers(50)
   inst4.setStepSize(.5)
   inst4.findSourceCells()
   assert len(inst4.sourceCells) == 1
   inst4.calculate()
   
   task4Id = 'task4'
   inst4.writeChangedVectors(outDir, ts=1.0, taskId=task4Id)
   assert os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task4Id))
   assert os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task4Id))
   assert os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task4Id))
   assert os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task4Id))
   # Load vectors we will use later
   task4Top = np.load(os.path.join(outDir, "%s-toTop.npy" % task4Id))
   task4Left = np.load(os.path.join(outDir, "%s-toLeft.npy" % task4Id))
   task4Right = np.load(os.path.join(outDir, "%s-toRight.npy" % task4Id))
   task4Bottom = np.load(os.path.join(outDir, "%s-toBottom.npy" % task4Id))
   
   # Check tile 4 costs after task 4
   t4cost = np.loadtxt(cost4Fn, comments='', dtype=int, skiprows=6)
   t4check = np.array([
                       [13, 13, 13, 15, 17, 15, 10, 11, 11, 12], 
                       [13, 13, 13, 15, 14, 13,  8,  8,  9, 10], 
                       [13, 13, 16, 15, 14, 13,  8,  9, 10, 10], 
                       [13, 13, 16, 16, 14, 13,  8,  9, 10, 11], 
                       [13, 13, 14, 14, 15, 14,  8,  9, 10, 10], 
                       [13, 17, 15, 15, 13, 11,  7,  7,  7,  8], 
                       [13, 14, 14, 14, 13, 11,  6,  6,  6,  7], 
                       [13, 13, 14, 13, 12, 10,  5,  5,  6,  6], 
                       [10, 12, 10, 12, 10,  5,  5,  4,  5,  5], 
                       [10, 10, 10, 10, 10,  5,  5,  0,  4,  4]], dtype=int)
   assert np.array_equal(t4cost, t4check)
   t4cost = None
   t4check = None
   
   # Check vectors
   t4LeftCheck = np.array([13, 13, 13, 13, 13, 13, 13, 13, 10, 10], dtype=int)
   t4RightCheck = np.array([12, 10, 10, 11, 10, 8, 7, 6, 5, 4], dtype=int)
   t4TopCheck = np.array([13, 13, 13, 15, 17, 15, 10, 11, 11, 12], dtype=int)
   t4BottomCheck = np.array([10, 10, 10, 10, 10, 5, 5, 0, 4, 4], dtype=int)
   
   assert np.array_equal(task4Top, t4TopCheck)
   assert np.array_equal(task4Bottom, t4BottomCheck)
   assert np.array_equal(task4Left, t4LeftCheck)
   assert np.array_equal(task4Right, t4RightCheck)
   
   # Tile 2 - task 5
   #   - add source cells from tile 1 (from left)
   #   - should generate vectors
   inst5 = SingleTileParallelDijkstraLCP(tile2Fn, cost2Fn, seaLevelRiseCostFn)
   inst5.setMaxWorkers(50)
   inst5.setStepSize(1.0)
   inst5.addSourceVector(task1Right, 0)
   assert len(inst5.sourceCells) == 10
   inst5.calculate()
   
   task5Id = 'task5'
   inst5.writeChangedVectors(outDir, ts=1.0, taskId=task5Id)
  
   # Check tile 2 costs after task 5
   t5cost = np.loadtxt(cost2Fn, comments='', dtype=int, skiprows=6)
   t5check = np.array([
                       [15, 15, 15, 15, 15, 15, 15, 15, 15, 15], 
                       [16, 15, 15, 15, 15, 15, 15, 15, 15, 15], 
                       [16, 15, 15, 15, 15, 15, 15, 15, 15, 15], 
                       [15, 15, 19, 15, 15, 15, 15, 15, 15, 15], 
                       [11, 16, 22, 16, 15, 15, 15, 15, 15, 15], 
                       [16, 16, 16, 15, 15, 15, 15, 15, 15, 15], 
                       [15, 17, 17, 15, 15, 15, 15, 15, 15, 15], 
                       [14, 16, 18, 15, 15, 15, 15, 15, 15, 15], 
                       [13, 18, 16, 18, 16, 15, 15, 15, 15, 15], 
                       [ 9, 11, 16, 16, 16, 15, 15, 15, 15, 15]], dtype=int)
   assert np.array_equal(t5cost, t5check)
   t5cost = None
   t5check = None
   
   assert os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task5Id))
   assert os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task5Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task5Id))
   assert os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task5Id))
   # Load vectors we will use later
   task5Top = np.load(os.path.join(outDir, "%s-toTop.npy" % task5Id))
   task5Right = np.load(os.path.join(outDir, "%s-toRight.npy" % task5Id))
   task5Bottom = np.load(os.path.join(outDir, "%s-toBottom.npy" % task5Id))

   # Check vectors
   t5RightCheck = np.array([15, 15, 15, 15, 15, 15, 15, 15, 15, 15], dtype=int)
   t5TopCheck = np.array([15, 15, 15, 15, 15, 15, 15, 15, 15, 15], dtype=int)
   t5BottomCheck = np.array([9, 11, 16, 16, 16, 15, 15, 15, 15, 15], dtype=int)
   
   assert np.array_equal(task5Top, t5TopCheck)
   assert np.array_equal(task5Bottom, t5BottomCheck)
   assert np.array_equal(task5Right, t5RightCheck)
   
   # Tile 3 - task 6
   #   - add source cells from tile 1 (from top)
   #   - should generate vectors
   inst6 = SingleTileParallelDijkstraLCP(tile3Fn, cost3Fn, seaLevelRiseCostFn)
   inst6.setMaxWorkers(50)
   inst6.setStepSize(.5)
   inst6.addSourceVector(task1Bottom, 1)
   assert len(inst6.sourceCells) == 10
   inst6.calculate()
   
   task6Id = 'task6'
   inst6.writeChangedVectors(outDir, ts=1.0, taskId=task6Id)
   assert os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task6Id))
   assert os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task6Id))
   assert os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task6Id))
   assert os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task6Id))
   # Load vectors we will use later
   task6Top = np.load(os.path.join(outDir, "%s-toTop.npy" % task6Id))
   task6Left = np.load(os.path.join(outDir, "%s-toLeft.npy" % task6Id))
   task6Right = np.load(os.path.join(outDir, "%s-toRight.npy" % task6Id))
   task6Bottom = np.load(os.path.join(outDir, "%s-toBottom.npy" % task6Id))
   
   # Check tile 3 costs after task 6
   t6cost = np.loadtxt(cost3Fn, comments='', dtype=int, skiprows=6)
   t6check = np.array([
                       [ 9, 11, 11, 11,  9,  3,  8, 10,  3,  6], 
                       [ 9, 11, 12, 11,  9,  3,  3,  3,  3,  9], 
                       [ 9, 11, 11, 11,  9,  3,  3,  3,  9,  9], 
                       [ 9, 11, 11, 12, 10,  4,  4,  4,  8, 10], 
                       [ 9, 11, 12, 11,  9,  4,  8,  8,  9, 14], 
                       [ 9, 12, 12, 12,  9,  9,  9,  9, 10, 14], 
                       [ 9, 11, 12, 11,  9,  9,  9, 11, 12, 16], 
                       [ 9,  9, 10, 10,  9,  9, 10, 12, 15, 17], 
                       [ 9,  9,  9,  9,  9,  9,  9, 10, 11, 10], 
                       [ 9,  9,  9,  9,  9,  9,  9,  9,  9,  9]], dtype=int)
   assert np.array_equal(t6cost, t6check)
   t6cost = None
   t6check = None
   
   # Check vectors
   t6LeftCheck = np.array([9, 9, 9, 9, 9, 9, 9, 9, 9, 9], dtype=int)
   t6RightCheck = np.array([6, 9, 9, 10, 14, 14, 16, 17, 10, 9], dtype=int)
   t6TopCheck = np.array([9, 11, 11, 11, 9, 3, 8, 10, 3, 6], dtype=int)
   t6BottomCheck = np.array([9, 9, 9, 9, 9, 9, 9, 9, 9, 9], dtype=int)
   
   assert np.array_equal(task6Top, t6TopCheck)
   assert np.array_equal(task6Bottom, t6BottomCheck)
   assert np.array_equal(task6Left, t6LeftCheck)
   assert np.array_equal(task6Right, t6RightCheck)
   

   # Tile 2 - task 7
   #   - add source cells from tile 4 (from bottom)
   #   - should generate vectors
   inst7 = SingleTileParallelDijkstraLCP(tile2Fn, cost2Fn, seaLevelRiseCostFn)
   inst7.setMaxWorkers(50)
   inst7.setStepSize(1.0)
   inst7.addSourceVector(task4Top, 3)
   assert len(inst7.sourceCells) == 4
   inst7.calculate()
   
   task7Id = 'task7'
   inst7.writeChangedVectors(outDir, ts=1.0, taskId=task7Id)
   assert os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task7Id))
   assert os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task7Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task7Id))
   assert os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task7Id))
   # Load vectors we will use later
   task7Top = np.load(os.path.join(outDir, "%s-toTop.npy" % task7Id))
   task7Right = np.load(os.path.join(outDir, "%s-toRight.npy" % task7Id))
   task7Bottom = np.load(os.path.join(outDir, "%s-toBottom.npy" % task7Id))
   
   # Check tile 2 costs after task 7
   t7cost = np.loadtxt(cost2Fn, comments='', dtype=int, skiprows=6)
   t7check = np.array([
                       [15, 15, 13, 12, 10, 10, 10, 10, 10, 10], 
                       [16, 15, 13, 12, 10, 10, 10, 10, 10, 10], 
                       [16, 15, 13, 12, 10, 10, 10, 10, 10, 10], 
                       [15, 15, 19, 11, 11, 10, 10, 10, 10, 10], 
                       [11, 16, 22, 16, 14, 10, 10, 10, 10, 10], 
                       [16, 16, 16, 15, 15, 10, 10, 10, 10, 10], 
                       [15, 17, 17, 15, 12, 10, 10, 10, 10, 10], 
                       [14, 16, 18, 13, 13, 12, 10, 10, 10, 10], 
                       [13, 18, 16, 18, 16, 14, 10, 11, 10, 10], 
                       [ 9, 11, 16, 16, 16, 14, 10, 10, 11, 11]], dtype=int)
   assert np.array_equal(t7cost, t7check)
   t7cost = None
   t7check = None
   
   # Check vectors
   t7RightCheck = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 11], dtype=int)
   t7TopCheck = np.array([15, 15, 13, 12, 10, 10, 10, 10, 10, 10], dtype=int)
   t7BottomCheck = np.array([9, 11, 16, 16, 16, 14, 10, 10, 11, 11], dtype=int)
   
   assert np.array_equal(task7Top, t7TopCheck)
   assert np.array_equal(task7Bottom, t7BottomCheck)
   assert np.array_equal(task7Right, t7RightCheck)
   
   # Tile 3 - task 8
   #   - add source cells from tile 4 (from right)
   #   - no vectors
   inst8 = SingleTileParallelDijkstraLCP(tile3Fn, cost3Fn, seaLevelRiseCostFn)
   inst8.setMaxWorkers(50)
   inst8.setStepSize(.5)
   inst8.addSourceVector(task4Left, 2)
   assert len(inst8.sourceCells) == 0
   inst8.calculate()
   
   task8Id = 'task8'
   inst8.writeChangedVectors(outDir, ts=1.0, taskId=task8Id)
   assert not os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task8Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task8Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task8Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task8Id))
   
   # Check tile 3 costs after task 8
   t8cost = np.loadtxt(cost3Fn, comments='', dtype=int, skiprows=6)
   t8check = np.array([
                       [ 9, 11, 11, 11,  9,  3,  8, 10,  3,  6], 
                       [ 9, 11, 12, 11,  9,  3,  3,  3,  3,  9], 
                       [ 9, 11, 11, 11,  9,  3,  3,  3,  9,  9], 
                       [ 9, 11, 11, 12, 10,  4,  4,  4,  8, 10], 
                       [ 9, 11, 12, 11,  9,  4,  8,  8,  9, 14], 
                       [ 9, 12, 12, 12,  9,  9,  9,  9, 10, 14], 
                       [ 9, 11, 12, 11,  9,  9,  9, 11, 12, 16], 
                       [ 9,  9, 10, 10,  9,  9, 10, 12, 15, 17], 
                       [ 9,  9,  9,  9,  9,  9,  9, 10, 11, 10], 
                       [ 9,  9,  9,  9,  9,  9,  9,  9,  9,  9]], dtype=int)
   assert np.array_equal(t8cost, t8check)
   t8cost = None
   t8check = None
   
   # Tile 1 - task 9
   #   - add source cells from tile 3 (from bottom)
   #   - should generate vectors
   inst9 = SingleTileParallelDijkstraLCP(tile1Fn, cost1Fn, seaLevelRiseCostFn)
   inst9.setMaxWorkers(50)
   inst9.setStepSize(.5)
   inst9.addSourceVector(task6Top, 3)
   assert len(inst9.sourceCells) == 2
   inst9.calculate()
   
   task9Id = 'task9'
   inst9.writeChangedVectors(outDir, ts=1.0, taskId=task9Id)
   assert os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task9Id))
   assert os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task9Id))
   assert os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task9Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task9Id))
   # Load vectors we will use later
   task9Top = np.load(os.path.join(outDir, "%s-toTop.npy" % task9Id))
   task9Left = np.load(os.path.join(outDir, "%s-toLeft.npy" % task9Id))
   task9Bottom = np.load(os.path.join(outDir, "%s-toBottom.npy" % task9Id))
   
   # Check tile 1 costs after task 9
   t9cost = np.loadtxt(cost1Fn, comments='', dtype=int, skiprows=6)
   t9check = np.array([
                       [ 9,  9, 12, 16, 16, 17, 15, 11, 11, 11], 
                       [ 9,  9, 13, 17, 18, 18, 16, 13, 11, 13], 
                       [ 9, 11, 13, 17, 18, 18, 16, 11, 11, 13], 
                       [ 9,  9, 14, 15, 16, 16, 15, 10, 11, 15], 
                       [ 9,  9, 12, 13, 11, 11,  9,  6,  6,  8], 
                       [ 9,  9,  9, 13,  9,  9,  5,  6,  8,  8], 
                       [ 9, 11, 10,  9,  7,  5,  5,  4,  5,  7], 
                       [ 9, 11, 13, 13, 11,  7,  4,  4,  5, 12], 
                       [ 9, 10, 11, 12, 10,  5,  7,  4,  4, 11], 
                       [ 9, 11, 11, 11,  9,  3,  3,  8,  0,  7]], dtype=int)
   assert np.array_equal(t9cost, t9check)
   t9cost = None
   t9check = None
   
   # Check vectors
   t9LeftCheck = np.array([9, 9, 9, 9, 9, 9, 9, 9, 9, 9], dtype=int)
   t9TopCheck = np.array([9, 9, 12, 16, 16, 17, 15, 11, 11, 11], dtype=int)
   t9BottomCheck = np.array([9, 11, 11, 11, 9, 3, 3, 8, 0, 7], dtype=int)
   
   assert np.array_equal(task9Top, t9TopCheck)
   assert np.array_equal(task9Bottom, t9BottomCheck)
   assert np.array_equal(task9Left, t9LeftCheck)

   
   # Tile 4 - task 10
   #   - add source cells from tile 3 (from left)
   #   - should generate vectors
   inst10 = SingleTileParallelDijkstraLCP(tile4Fn, cost4Fn, seaLevelRiseCostFn)
   inst10.setMaxWorkers(50)
   inst10.setStepSize(.5)
   inst10.addSourceVector(task6Right, 0)
   assert len(inst10.sourceCells) == 4
   inst10.calculate()
   
   task10Id = 'task10'
   inst10.writeChangedVectors(outDir, ts=1.0, taskId=task10Id)
   assert os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task10Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task10Id))
   assert os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task10Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task10Id))
   # Load vectors we will use later
   task10Top = np.load(os.path.join(outDir, "%s-toTop.npy" % task10Id))
   task10Left = np.load(os.path.join(outDir, "%s-toLeft.npy" % task10Id))
   
   # Check tile 4 costs after task 10
   t10cost = np.loadtxt(cost4Fn, comments='', dtype=int, skiprows=6)
   t10check = np.array([
                       [10, 11, 13, 15, 17, 15, 10, 11, 11, 12], 
                       [10, 11, 13, 15, 14, 13,  8,  8,  9, 10], 
                       [ 9, 11, 16, 15, 14, 13,  8,  9, 10, 10], 
                       [ 9, 12, 16, 16, 14, 13,  8,  9, 10, 11], 
                       [13, 12, 14, 14, 15, 14,  8,  9, 10, 10], 
                       [13, 17, 15, 15, 13, 11,  7,  7,  7,  8], 
                       [13, 14, 14, 14, 13, 11,  6,  6,  6,  7], 
                       [13, 13, 14, 13, 12, 10,  5,  5,  6,  6], 
                       [10, 12, 10, 12, 10,  5,  5,  4,  5,  5], 
                       [10, 10, 10, 10, 10,  5,  5,  0,  4,  4]], dtype=int)
   assert np.array_equal(t10cost, t10check)
   t10cost = None
   t10check = None
   
   # Check vectors
   t10LeftCheck = np.array([10, 10, 9, 9, 13, 13, 13, 13, 10, 10], dtype=int)
   t10TopCheck = np.array([10, 11, 13, 15, 17, 15, 10, 11, 11, 12], dtype=int)
   
   assert np.array_equal(task10Top, t10TopCheck)
   assert np.array_equal(task10Left, t10LeftCheck)
   
   
   # Tile 4 - task 11
   #   - add source cells from tile 2 (from top)
   #   - should generate vectors
   inst11 = SingleTileParallelDijkstraLCP(tile4Fn, cost4Fn, seaLevelRiseCostFn)
   inst11.setMaxWorkers(50)
   inst11.setStepSize(.5)
   inst11.addSourceVector(task7Bottom, 1)
   assert len(inst11.sourceCells) == 0
   inst11.calculate()
   
   task11Id = 'task11'
   inst11.writeChangedVectors(outDir, ts=1.0, taskId=task11Id)
   assert not os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task11Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task11Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task11Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task11Id))
   # Load vectors we will use later
   
   # Check tile 4 costs after task 11
   t11cost = np.loadtxt(cost4Fn, comments='', dtype=int, skiprows=6)
   t11check = np.array([
                       [10, 11, 13, 15, 17, 15, 10, 11, 11, 12], 
                       [10, 11, 13, 15, 14, 13,  8,  8,  9, 10], 
                       [ 9, 11, 16, 15, 14, 13,  8,  9, 10, 10], 
                       [ 9, 12, 16, 16, 14, 13,  8,  9, 10, 11], 
                       [13, 12, 14, 14, 15, 14,  8,  9, 10, 10], 
                       [13, 17, 15, 15, 13, 11,  7,  7,  7,  8], 
                       [13, 14, 14, 14, 13, 11,  6,  6,  6,  7], 
                       [13, 13, 14, 13, 12, 10,  5,  5,  6,  6], 
                       [10, 12, 10, 12, 10,  5,  5,  4,  5,  5], 
                       [10, 10, 10, 10, 10,  5,  5,  0,  4,  4]], dtype=int)
   assert np.array_equal(t11cost, t11check)
   t11cost = None
   t11check = None
   
   # Tile 3 - task 12
   #   - add source cells from tile 1 (from top)
   #   - should generate vectors
   inst12 = SingleTileParallelDijkstraLCP(tile3Fn, cost3Fn, seaLevelRiseCostFn)
   inst12.setMaxWorkers(50)
   inst12.setStepSize(.5)
   inst12.addSourceVector(task9Bottom, 1)
   assert len(inst12.sourceCells) == 0
   inst12.calculate()
   
   task12Id = 'task12'
   inst12.writeChangedVectors(outDir, ts=1.0, taskId=task12Id)
   assert not os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task12Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task12Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task12Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task12Id))
   # Load vectors we will use later
   
   # Check tile 3 costs after task 12
   t12cost = np.loadtxt(cost3Fn, comments='', dtype=int, skiprows=6)
   t12check = np.array([
                       [ 9, 11, 11, 11,  9,  3,  8, 10,  3,  6], 
                       [ 9, 11, 12, 11,  9,  3,  3,  3,  3,  9], 
                       [ 9, 11, 11, 11,  9,  3,  3,  3,  9,  9], 
                       [ 9, 11, 11, 12, 10,  4,  4,  4,  8, 10], 
                       [ 9, 11, 12, 11,  9,  4,  8,  8,  9, 14], 
                       [ 9, 12, 12, 12,  9,  9,  9,  9, 10, 14], 
                       [ 9, 11, 12, 11,  9,  9,  9, 11, 12, 16], 
                       [ 9,  9, 10, 10,  9,  9, 10, 12, 15, 17], 
                       [ 9,  9,  9,  9,  9,  9,  9, 10, 11, 10], 
                       [ 9,  9,  9,  9,  9,  9,  9,  9,  9,  9]], dtype=int)
   assert np.array_equal(t12cost, t12check)
   t12cost = None
   t12check = None
   
   # Tile 3 - task 13
   #   - add source cells from tile 4 (from right)
   #   - should generate vectors
   inst13 = SingleTileParallelDijkstraLCP(tile3Fn, cost3Fn, seaLevelRiseCostFn)
   inst13.setMaxWorkers(50)
   inst13.setStepSize(.5)
   inst13.addSourceVector(task10Left, 2)
   assert len(inst13.sourceCells) == 0
   inst13.calculate()
   
   task13Id = 'task12'
   inst13.writeChangedVectors(outDir, ts=1.0, taskId=task13Id)
   assert not os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task13Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task13Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task13Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task13Id))
   # Load vectors we will use later
   
   # Check tile 3 costs after task 13
   t13cost = np.loadtxt(cost3Fn, comments='', dtype=int, skiprows=6)
   t13check = np.array([
                       [ 9, 11, 11, 11,  9,  3,  8, 10,  3,  6], 
                       [ 9, 11, 12, 11,  9,  3,  3,  3,  3,  9], 
                       [ 9, 11, 11, 11,  9,  3,  3,  3,  9,  9], 
                       [ 9, 11, 11, 12, 10,  4,  4,  4,  8, 10], 
                       [ 9, 11, 12, 11,  9,  4,  8,  8,  9, 14], 
                       [ 9, 12, 12, 12,  9,  9,  9,  9, 10, 14], 
                       [ 9, 11, 12, 11,  9,  9,  9, 11, 12, 16], 
                       [ 9,  9, 10, 10,  9,  9, 10, 12, 15, 17], 
                       [ 9,  9,  9,  9,  9,  9,  9, 10, 11, 10], 
                       [ 9,  9,  9,  9,  9,  9,  9,  9,  9,  9]], dtype=int)
   assert np.array_equal(t13cost, t13check)
   t13cost = None
   t13check = None
   
   # Tile 2 - task 14
   #   - add source cells from tile 4 (from bottom)
   #   - should generate vectors
   inst14 = SingleTileParallelDijkstraLCP(tile2Fn, cost2Fn, seaLevelRiseCostFn)
   inst14.setMaxWorkers(50)
   inst14.setStepSize(.5)
   inst14.addSourceVector(task10Top, 3)
   assert len(inst14.sourceCells) == 0
   inst14.calculate()
   
   task14Id = 'task14'
   inst14.writeChangedVectors(outDir, ts=1.0, taskId=task14Id)
   assert not os.path.exists(os.path.join(outDir, "%s-toTop.npy" % task14Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toBottom.npy" % task14Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toLeft.npy" % task14Id))
   assert not os.path.exists(os.path.join(outDir, "%s-toRight.npy" % task14Id))
   # Load vectors we will use later
   
   # Check tile 3 costs after task 12
   t14cost = np.loadtxt(cost2Fn, comments='', dtype=int, skiprows=6)
   t14check = np.array([
                       [15, 15, 13, 12, 10, 10, 10, 10, 10, 10], 
                       [16, 15, 13, 12, 10, 10, 10, 10, 10, 10], 
                       [16, 15, 13, 12, 10, 10, 10, 10, 10, 10], 
                       [15, 15, 19, 11, 11, 10, 10, 10, 10, 10], 
                       [11, 16, 22, 16, 14, 10, 10, 10, 10, 10], 
                       [16, 16, 16, 15, 15, 10, 10, 10, 10, 10], 
                       [15, 17, 17, 15, 12, 10, 10, 10, 10, 10], 
                       [14, 16, 18, 13, 13, 12, 10, 10, 10, 10], 
                       [13, 18, 16, 18, 16, 14, 10, 11, 10, 10], 
                       [ 9, 11, 16, 16, 16, 14, 10, 10, 11, 11]], dtype=int)
   print t14cost
   print t14check
   assert np.array_equal(t14cost, t14check)
   t14cost = None
   t14check = None
   


   
   # Check final
   tile1CostFinal = np.loadtxt(cost1Fn, dtype=int, comments='', skiprows=6)
   assert np.array_equal(tile1CostFinal,
            np.array([[i for i in line[:10]] for line in COST_SURFACE2[:10]], 
                     dtype=int))

   tile2CostFinal = np.loadtxt(cost2Fn, dtype=int, comments='', skiprows=6)
   assert np.array_equal(tile2CostFinal,
            np.array([[i for i in line[10:]] for line in COST_SURFACE2[:10]], 
                     dtype=int))

   tile3CostFinal = np.loadtxt(cost3Fn, dtype=int, comments='', skiprows=6)
   assert np.array_equal(tile3CostFinal,
            np.array([[i for i in line[:10]] for line in COST_SURFACE2[10:]], 
                     dtype=int))

   tile4CostFinal = np.loadtxt(cost4Fn, dtype=int, comments='', skiprows=6)
   assert np.array_equal(tile4CostFinal,
            np.array([[i for i in line[10:]] for line in COST_SURFACE2[10:]], 
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
