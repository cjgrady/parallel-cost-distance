"""
@summary: Simple test to make sure that the base single tile class works
"""
import numpy

from singleTile.base import SingleTileLCP

if __name__ == "__main__":
   
   inFn = 'testSurface.asc'
   outFn = 'testOut.asc'
   
   costFn = lambda x,y,z: min(x,y)
   t1 = SingleTileLCP(inFn, outFn, costFn)
   t2 = SingleTileLCP(inFn, outFn, costFn)
   t3 = SingleTileLCP(inFn, outFn, costFn)
   
   t1.findSourceCells()
   
   v1 = numpy.array([5, 3, 1, 2, 5])
   v2 = numpy.array([8, 3, 4, 1, 2, 4])
   
   t2.addSourceVector(v1, 0)
   t3.addSourceVector(v2, 1)
   
   print "t1"
   print t1.sourceCells
   
   print
   print "t2"
   print t2.sourceCells
   print
   
   print "t3"
   print t3.sourceCells
   print
   