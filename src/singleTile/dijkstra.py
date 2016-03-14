"""
@summary: This module contains a more traditional implementation of Dijkstra's
             algorithm for graph traversal that does not operate in parallel
@author: CJ Grady
@status: alpha
"""
import heapq

from singleTile.base import SingleTileLCP

# .............................................................................
class SingleTileSerialDijkstraLCP(SingleTileLCP):
   """
   @summary: This is the serial Dijkstra implementation
   """
   # ..........................
   def _calculate(self):
      """
      @summary: Calculate the least cost path for all cells in the grid to a 
                   source cell
      @note: This method should be implemented in subclasses
      """
      hq = []
      
      # ........................
      def addCell(x, y, cost):
         """
         @summary: Add a cell to the heap if appropriate
         """
         if int(self.cMtx[y][x]) == int(self.noDataValue) or self.cMtx[y][x] > cost:
            heapq.heappush(hq, (cost, x, y))
   
      # ........................
      def addNeighbors(x, y, cost):
         cellCost = self.inMtx[y][x]
         if cellCost != self.noDataValue:
            if x - 1 >= 0:
               addCell(x-1, y, self.costFn(cost, cellCost, self.inMtx[y][x-1], self.cellSize))
            if x + 1 < len(self.inMtx[y]):
               addCell(x+1, y, self.costFn(cost, cellCost, self.inMtx[y][x+1], self.cellSize))
            if y-1 >= 0:
               addCell(x, y-1, self.costFn(cost, cellCost, self.inMtx[y-1][x], self.cellSize))
            if y+1 < len(self.inMtx):
               addCell(x, y+1, self.costFn(cost, cellCost, self.inMtx[y+1][x], self.cellSize))

      # .........................
      # Dijkstra
      
      for x, y in self.sourceCells:
         if int(self.cMtx[y][x]) == int(self.noDataValue):
            c = self.inMtx[y][x]
         else:
            c = self.cMtx[y][x]
         print "Adding source cell:", x, y, c
         print self.inMtx[y][x]
         print self.cMtx[y][x]
         addNeighbors(x, y, c)
            
      while len(hq) > 0:
         cost, x, y = heapq.heappop(hq)
         
         if self.cMtx[y][x] == self.noDataValue or cost < self.cMtx[y][x]:
            self.cMtx[y][x] = cost
            addNeighbors(x, y, cost)
      