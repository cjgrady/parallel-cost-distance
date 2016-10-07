"""
@summary: This module tests the cost functions module
@author: CJ Grady
@status: alpha
@version: 1.0
"""
from slr.common.costFunctions import seaLevelRiseCostFn

# .............................................................................
def test_simple():
   """
   @summary: Just a basic test that this function exists and works
   """
   assert seaLevelRiseCostFn(4, 1, 8, 9) == 8
   
# .............................................................................
def test_currentCostGreater():
   """
   @summary: Test that the cost function works when the already incurred cost 
               is greater than the connected cell height
   """
   assert seaLevelRiseCostFn(10, 3, 6, 1) == 10
   
# .............................................................................
def test_connectedCellCostGreater():
   """
   @summary: Test that the cost function works when the connected cell height
                is greater than the currently incurred cost
   """
   assert seaLevelRiseCostFn(4, 4, 10, 1) == 10
