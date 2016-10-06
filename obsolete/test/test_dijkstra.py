"""
@summary: Tests for serial Dijstra single tile
@author: CJ Grady
@status: alpha
"""
import time

from slr.singleTile.dijkstra import SingleTileSerialDijkstraLCP
# .............................................................................
if __name__ == "__main__":
   
   t1 = time.time()
   inFn = '/home/cjgrady/thesis/fl_east_gom_crm_v1.asc'
   outFn = '/home/cjgrady/thesis/serialDijkstraFL.asc'
   
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
   
   t2 = time.time()
   print "Elapsed time:", t2-t1