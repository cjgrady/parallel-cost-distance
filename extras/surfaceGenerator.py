"""
@summary: This script will generate a surface using cones and ellipsoids
@author: CJ Grady
"""
from math import sqrt
import numpy as np
from random import randint

# .............................................................................
class SurfaceGenerator(object):
   """
   @summary: This class is used to generate a surface to be used for testing
   """
   # ..................................
   def __init__(self, nrows, ncols, xll, yll, cellSize, defVal=1):
      if defVal == 0:
         self.grid = np.zeros((nrows, ncols), dtype=int)
      else:
         self.grid = np.ones((nrows, ncols), dtype=int) * defVal

      self.headers = """\
ncols   {0}
nrows   {1}
xllcorner   {2}
yllcorner   {3}
cellsize   {4}
NODATA_value   {5}
""".format(ncols, nrows, xll, yll, cellSize, -9999)
      
   # ..................................
   def addCone(self, xCnt, yCnt, rad, height, maxHeight=None):
      def insideCircle(x, y):
         return ((1.0*x - xCnt)**2 / rad**2) + ((1.0*y - yCnt)**2 / rad**2) <= 1

      def getZ(x, y):
         # % 1 - rad * total height
         
         xPart = (1.0*x - xCnt)**2
         yPart = (1.0*y - yCnt)**2 
         dist = sqrt(xPart + yPart)
         
         val = height * (1 - dist / rad)
         
         if maxHeight is not None:
            val = max(val, maxHeight)
         
         return val
   
      for x in range(max(0, (xCnt - rad)), min(self.grid.shape[1], (xCnt + rad))):
         for y in range(max(0, (yCnt - rad)), min(self.grid.shape[0], (yCnt + rad))):
            if insideCircle(x, y):
               #self.grid[y,x] = max(getZ(x, y), self.grid[y,x])
               try:
                  self.grid[y,x] = self.grid[y,x] + getZ(x, y)
               except:
                  print "An exception occurred when setting the value of a cell"

   # ..................................
   def addEllipsoid(self, xCnt, yCnt, xRad, yRad, height, maxHeight=None):
      def insideEllipse(x, y):
         return ((1.0*x - xCnt)**2 / xRad**2) + ((1.0*y - yCnt)**2 / yRad**2) <= 1
      
      def getZ(x, y):
         xPart = (1.0*x - xCnt)**2 / xRad**2
         yPart = (1.0*y - yCnt)**2 / yRad**2
         
         t = 1 - xPart - yPart
         
         t2 = height**2 * t
         
         val = sqrt(t2)
         
         if maxHeight is not None:
            val = max(val, maxHeight)
         return val
      
      for x in range(max(0, (xCnt - xRad)), min(self.grid.shape[1], (xCnt + xRad))):
         for y in range(max(0, (yCnt - yRad)), min(self.grid.shape[0], (yCnt + yRad))):
            if insideEllipse(x, y):
               #self.grid[y,x] = max(getZ(x, y), self.grid[y,x])
               try:
                  self.grid[y,x] = self.grid[y,x] + getZ(x, y)
               except:
                  print "An exception occurred when setting the value of a cell"
               
   # ..................................
   def addRandom(self, numCones=0, numEllipsoids=0, maxHeight=100, maxRad=100):
      # TODO: Add mesas?
      maxX, maxY = self.grid.shape
      # Add cones
      for i in range(numCones):
         h = randint(1, maxHeight)
         self.addCone(randint(0, maxX), randint(0, maxY), randint(1, maxRad), h, maxHeight=randint(1, h))
      
      # Add ellipsoids
      for i in range(numEllipsoids):
         h = randint(1, maxHeight)
         self.addEllipsoid(randint(0, maxX), randint(0, maxY), randint(1, maxRad), randint(1, maxRad), h, maxHeight=randint(1, h))
      
   # ..................................
   def writeGrid(self, fn):
      np.savetxt(fn, self.grid, fmt="%i", header=self.headers, comments='')
      
# .............................................................................
if __name__ == "__main__":
   #s = SurfaceGenerator(20, 20, 10, 10, 10, defVal=0)
   #s = SurfaceGenerator(1000, 1000, 0, 0, 0.05, defVal=0)
   s = SurfaceGenerator(20, 20, 0, 0, 1.0, defVal=0)
   #s.addEllipsoid(5, 5, 4, 3, 10)
   #s.addEllipsoid(7, 7, 2, 5, 10)
   #s.addCone(14, 14, 5, 20)
   #s.writeGrid('/home/cjgrady/testSurface.asc')
   s.addRandom(numCones=10, numEllipsoids=10, maxHeight=10, maxRad=10)
   #s.writeGrid('/home/cjgrady/git/irksome-broccoli/testData/surfaces/testSurface.asc')
   print s.grid.tolist()
   