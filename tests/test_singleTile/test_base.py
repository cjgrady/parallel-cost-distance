"""
@summary: This module tests the single tile base module
@author: CJ Grady
@version: 1.0
@status: beta
"""
from nose.tools import raises 
import numpy as np
import os
from slr.common.costFunctions import seaLevelRiseCostFn
from slr.singleTile.base import SingleTileLCP

from tests.helpers.testConstants import (BELOW_SEA_LEVEL_RASTER, 
               L_INPUT_RASTER, M_INPUT_RASTER, M_INPUT_RASTER_NO_SOURCES, 
               M_COST_RASTER, S_INPUT_RASTER, TEST_VECTOR)

# .............................................................................
# .                             Constructor Tests                             .
# .............................................................................
def test_constructor():
   """
   @summary: This test checks that the constructor will at least work
   """
   myInstance = SingleTileLCP(M_INPUT_RASTER_NO_SOURCES, M_COST_RASTER, 
                              seaLevelRiseCostFn)

# .............................................................................
def test_constructor_existingCostGrid():
   """
   @summary: This test checks that the constructor will work with an existing
                cost surface
   """
   myInstance = SingleTileLCP(M_INPUT_RASTER, M_COST_RASTER, 
                              seaLevelRiseCostFn)

# .............................................................................
def test_constructor_noCostGrid():
   """
   @summary: This test checks that the constructor works when the cost surface
                has not been created yet
   """
   myInstance = SingleTileLCP(M_INPUT_RASTER_NO_SOURCES, '', 
                              seaLevelRiseCostFn)

# .............................................................................
@raises(IOError)
def test_constructor_noInputGrid():
   """
   @summary: This test checks that the constructor throws an IOError when no
                input surface is provided
   """
   myInstance = SingleTileLCP('', '', seaLevelRiseCostFn)

# .............................................................................
# .                              Calculate Tests                              .
# .............................................................................
@raises(NotImplementedError)
def test_calculate():
   """
   @summary: The base class does not have the _calculate method implemented, so
                this should fail
   """
   myInstance = SingleTileLCP(M_INPUT_RASTER_NO_SOURCES, 'someCostGrid.npy', 
                              seaLevelRiseCostFn)
   myInstance.calculate()
   
# .............................................................................
# .                          Add Source Vector Tests                          .
# .............................................................................
@raises(Exception)
def test_addSourceVector_badSide():
   """
   @summary: This test checks that an error is thrown when asking for to add 
                source cells from an invalid side
   """
   myInstance = SingleTileLCP(M_INPUT_RASTER, 'someCostGrid.npy', 
                              seaLevelRiseCostFn)
   myInstance.addSourceVector(np.load(TEST_VECTOR), 9)

# .............................................................................
def test_addSourceVector_matchLength():
   """
   @summary: This test checks that things operate properly when adding a valid
                source vector
   """
   myInstance = SingleTileLCP(M_INPUT_RASTER, 'someCostGrid.npy', 
                              seaLevelRiseCostFn)
   myInstance.addSourceVector(np.load(TEST_VECTOR), 1)


# .............................................................................
def test_addSourceVector_noChange():
   """
   @summary: This test checks that things operate properly when adding a valid
                source vector that does not have a lower cost for any values
   """
   myInstance = SingleTileLCP(M_INPUT_RASTER, 'someCostGrid.npy', 
                              seaLevelRiseCostFn)
   myInstance.addSourceVector(np.load(TEST_VECTOR), 2)


# .............................................................................
def test_addSourceVector_shrinkVector():
   """
   @summary: This test checks that an input vector can be used that is longer
                than the side it is being compared with (should be shrank)
   """
   myInstance = SingleTileLCP(S_INPUT_RASTER, 'someCostGrid.npy', 
                              seaLevelRiseCostFn)
   myInstance.addSourceVector(np.load(TEST_VECTOR), 3)

# .............................................................................
def test_addSourceVector_stretchVector():
   """
   @summary: This test checks that an input vector that is shorter than the 
                side it is to be compared with is stretched to match
   """
   myInstance = SingleTileLCP(L_INPUT_RASTER, 'someCostGrid.npy', 
                              seaLevelRiseCostFn)
   myInstance.addSourceVector(np.load(TEST_VECTOR), 0)

# .............................................................................
# .                         Find Source Values Tests                          .
# .............................................................................
def test_findSourceValues():
   """
   @summary: This test checks that source values can be found for proper inputs
   """
   myInstance = SingleTileLCP(M_INPUT_RASTER, '', 
                              seaLevelRiseCostFn)
   myInstance.findSourceCells()
   assert len(myInstance.sourceCells) > 0

# .............................................................................
def test_findSourceValues_noSourceValues():
   """
   @summary: This test checks that things operate correctly when an input 
                surface does not have source values in it
   """
   myInstance = SingleTileLCP(M_INPUT_RASTER_NO_SOURCES, '', 
                              seaLevelRiseCostFn)
   myInstance.findSourceCells()
   print myInstance.sourceCells
   assert len(myInstance.sourceCells) == 0

# .............................................................................
def test_findSourceValues_underSeaLevel():
   """
   @summary: This test checks that things operate correctly when an input 
                surface is under sea level
   """
   myInstance = SingleTileLCP(BELOW_SEA_LEVEL_RASTER, '', 
                              seaLevelRiseCostFn)
   myInstance.findSourceCells()
   assert len(myInstance.sourceCells) == 10000

