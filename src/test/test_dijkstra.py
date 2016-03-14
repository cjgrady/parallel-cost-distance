"""
@summary: Tests for serial Dijstra single tile
@author: CJ Grady
@status: alpha
"""
from singleTile.dijkstra import SingleTileSerialDijkstraLCP
# .............................................................................
if __name__ == "__main__":
   
   inFn = '/home/cjgrady/git/irksome-broccoli/testData/testGrid.asc'
   outFn = '/home/cjgrady/git/irksome-broccoli/testData/outputGrid.asc'
   
   def costFn(i, x, y, z):
      c = max(i, y)
      #print i, x, y, z, c
      return c
   
   #costFn = lambda x,y,z: min(x,y)
   t1 = SingleTileSerialDijkstraLCP(inFn, outFn, costFn)
   
   t1.findSourceCells()
   
   print "t1"
   print t1.sourceCells
   
   print "Attempting to calculate"
   t1.calculate()
   
   