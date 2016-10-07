"""
@summary: This module contains potential cost functions to be used for 
             calculating cost distance
@author: CJ Grady
@version: 1.0
@status: beta
@note: All cost functions should take arguments:
        - currentCost: The cost already incurred
        - cellValue: The value at the origin cell
        - connCellValue: The value at the connected cell in question
        - cellSize: The size of the cells
@note: To start, we will only have sea level rise
"""

# .............................................................................
def seaLevelRiseCostFn(currentCost, cellValue, connCellValue, cellSize):
   """
   @summary: This cost function finds the cost to inundate a cell based on the 
                current cost to get to that cell and the value of the connected 
                cell
   @param currentCost: The cost to reach this point
   @param cellValue: The value in the origin cell (elevation in this case)
   @param connCellValue: The value in the destination cell (elevation in this case)
   @param cellSize: The size of the cell (not used for this calculation) 
   """
   return max(currentCost, connCellValue)
   
