"""
@summary: Tests single tile Dijkstra module
@author: CJ Grady
"""
import numpy as np
import os
from tempfile import NamedTemporaryFile

from slr.common.costFunctions import seaLevelRiseCostFn

from slr.singleTile.dijkstra import SingleTileSerialDijkstraLCP

from tests.helpers.testConstants import (L_COST_RASTER, L_INPUT_RASTER, 
         M_COST_RASTER, M_INPUT_RASTER, S_COST_RASTER, S_INPUT_RASTER)

# .............................................................................
# .                              Calculate Tests                              .
# .............................................................................
def test_calculate_small():
   """
   @summary: Tests that calculations work for a single tile (small)
   """
   # We just need a name, not a file
   tf = NamedTemporaryFile(delete=True)
   cAryFn = tf.name
   tf.close()
   myInstance = SingleTileSerialDijkstraLCP(S_INPUT_RASTER, cAryFn, 
                              seaLevelRiseCostFn)
   myInstance.findSourceCells()
   myInstance.calculate()
   
   computedArray = np.loadtxt(cAryFn, dtype=int, comments='', skiprows=6)
   costArray = np.loadtxt(S_COST_RASTER, dtype=int, comments='', skiprows=6)
   
   # Delete temp file
   os.remove(cAryFn)
   
   assert np.array_equiv(computedArray, costArray)
   
# .............................................................................
def test_calculate_medium():
   """
   @summary: Tests that calculations work for a single tile (medium)
   """
   # We just need a name, not a file
   tf = NamedTemporaryFile(delete=True)
   cAryFn = tf.name
   tf.close()
   myInstance = SingleTileSerialDijkstraLCP(M_INPUT_RASTER, cAryFn, 
                              seaLevelRiseCostFn)
   myInstance.findSourceCells()
   myInstance.calculate()
   
   computedArray = np.loadtxt(cAryFn, dtype=int, comments='', skiprows=6)
   costArray = np.loadtxt(M_COST_RASTER, dtype=int, comments='', skiprows=6)
   
   # Delete temp file
   os.remove(cAryFn)
   
   assert np.array_equiv(computedArray, costArray)
   
# .............................................................................
def test_calculate_large():
   """
   @summary: Tests that calculations work for a single tile (large)
   """
   # We just need a name, not a file
   tf = NamedTemporaryFile(delete=True)
   cAryFn = tf.name
   tf.close()
   myInstance = SingleTileSerialDijkstraLCP(L_INPUT_RASTER, cAryFn, 
                              seaLevelRiseCostFn)
   myInstance.findSourceCells()
   myInstance.calculate()
   
   computedArray = np.loadtxt(cAryFn, dtype=int, comments='', skiprows=6)
   costArray = np.loadtxt(L_COST_RASTER, dtype=int, comments='', skiprows=6)
   
   # Delete temp file
   os.remove(cAryFn)
   
   assert np.array_equiv(computedArray, costArray)
   
