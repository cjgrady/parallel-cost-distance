"""
@summary: Tests single tile Dijkstra (parallel) module
@author: CJ Grady
"""
import numpy as np
import os
from tempfile import NamedTemporaryFile

from slr.common.costFunctions import seaLevelRiseCostFn

from slr.singleTile.parallelDijkstra import SingleTileParallelDijkstraLCP

from tests.helpers.testConstants import (L_COST_RASTER, L_INPUT_RASTER, 
         XL_COST_RASTER, XL_INPUT_RASTER)


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
   myInstance.setStepSize(50)
   myInstance.findSourceCells()
   myInstance.calculate()
   
   computedArray = np.loadtxt(cAryFn, dtype=int, comments='', skiprows=6)
   costArray = np.loadtxt(L_COST_RASTER, dtype=int, comments='', skiprows=6)
   
   # Delete temp file
   os.remove(cAryFn)
   
   assert np.array_equiv(computedArray, costArray)
   
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
   myInstance.setStepSize(50)
   myInstance.findSourceCells()
   myInstance.calculate()
   
   computedArray = np.loadtxt(cAryFn, dtype=int, comments='', skiprows=6)
   costArray = np.loadtxt(L_COST_RASTER, dtype=int, comments='', skiprows=6)
   
   # Delete temp file
   os.remove(cAryFn)
   
   assert np.array_equiv(computedArray, costArray)
   
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
   myInstance.setStepSize(3)
   myInstance.findSourceCells()
   myInstance.calculate()
   
   computedArray = np.loadtxt(cAryFn, dtype=int, comments='', skiprows=6)
   costArray = np.loadtxt(L_COST_RASTER, dtype=int, comments='', skiprows=6)
   
   # Delete temp file
   os.remove(cAryFn)
   
   assert np.array_equiv(computedArray, costArray)
   
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
   myInstance.setStepSize(33)
   myInstance.findSourceCells()
   myInstance.calculate()
   
   computedArray = np.loadtxt(cAryFn, dtype=int, comments='', skiprows=6)
   costArray = np.loadtxt(L_COST_RASTER, dtype=int, comments='', skiprows=6)
   
   # Delete temp file
   os.remove(cAryFn)
   
   assert np.array_equiv(computedArray, costArray)
   
# .............................................................................
def test_calculate_xlarge_evenSteps():
   """
   @summary: Tests that calculations work for a single tile (xlarge)
   """
   # We just need a name, not a file
   tf = NamedTemporaryFile(delete=True)
   cAryFn = tf.name
   tf.close()
   myInstance = SingleTileParallelDijkstraLCP(XL_INPUT_RASTER, cAryFn, 
                              seaLevelRiseCostFn)
   myInstance.setMaxWorkers(32)
   myInstance.setStepSize(100)
   myInstance.findSourceCells()
   myInstance.calculate()
   
   computedArray = np.loadtxt(cAryFn, dtype=int, comments='', skiprows=6)
   costArray = np.loadtxt(XL_COST_RASTER, dtype=int, comments='', skiprows=6)
   
   # Delete temp file
   os.remove(cAryFn)
   
   assert np.array_equiv(computedArray, costArray)

# .............................................................................
def test_calculate_xlarge_unevenSteps():
   """
   @summary: Tests that calculations work for a single tile (xlarge)
   """
   # We just need a name, not a file
   tf = NamedTemporaryFile(delete=True)
   cAryFn = tf.name
   tf.close()
   myInstance = SingleTileParallelDijkstraLCP(XL_INPUT_RASTER, cAryFn, 
                              seaLevelRiseCostFn)
   myInstance.setMaxWorkers(32)
   myInstance.setStepSize(333)
   myInstance.findSourceCells()
   myInstance.calculate()
   
   computedArray = np.loadtxt(cAryFn, dtype=int, comments='', skiprows=6)
   costArray = np.loadtxt(XL_COST_RASTER, dtype=int, comments='', skiprows=6)
   
   # Delete temp file
   os.remove(cAryFn)
   
   assert np.array_equiv(computedArray, costArray)
