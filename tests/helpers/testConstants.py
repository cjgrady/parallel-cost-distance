"""
@summary: This module contains constants to be used for testing
@author: CJ Grady
@version: 1.0
@status: alpha
"""
import os


if os.environ.has_key('TEST_DATA_DIR'):
   TEST_DATA_PATH = os.environ['TEST_DATA_DIR']
else:
   TEST_DATA_PATH = os.path.join(os.path.abspath(os.curdir), 'testData')

SURFACES_PATH = os.path.join(TEST_DATA_PATH, 'surfaces')
TILES_PATH = os.path.join(TEST_DATA_PATH, 'tiles')
COSTS_PATH = os.path.join(TEST_DATA_PATH, 'costs')
VECTORS_PATH = os.path.join(TEST_DATA_PATH, 'vectors')

# 10 x 10
S_INPUT_RASTER = os.path.join(SURFACES_PATH, 'small.asc')
S_COST_RASTER = os.path.join(COSTS_PATH, 'smallCost.asc')
S_INPUT_RASTER_NO_SOURCES = os.path.join(SURFACES_PATH, 'smallNoSources.asc')

# 100 x 100
M_INPUT_RASTER = os.path.join(SURFACES_PATH, 'medium.asc')
M_COST_RASTER = os.path.join(COSTS_PATH, 'mediumCost.asc')
M_INPUT_RASTER_NO_SOURCES = os.path.join(SURFACES_PATH, 'mediumNoSources.asc')

# 1000 x 1000
L_INPUT_RASTER = os.path.join(SURFACES_PATH, 'large.asc')
L_COST_RASTER = os.path.join(COSTS_PATH, 'largeCost.asc')
L_INPUT_RASTER_NO_SOURCES = os.path.join(SURFACES_PATH, 'largeNoSources.asc')

# 10000 x 10000
XL_INPUT_RASTER = os.path.join(SURFACES_PATH, 'xlarge.asc')
XL_COST_RASTER = os.path.join(COSTS_PATH, 'xlargeCost.asc')

# Below sea-level raster
BELOW_SEA_LEVEL_RASTER = os.path.join(SURFACES_PATH, 'underSL.asc')

# Test vector
TEST_VECTOR = os.path.join(VECTORS_PATH, 'testVector1.npy')