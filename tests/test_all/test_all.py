"""
@summary: This module tests the entire library to make sure results are 
             consistent across computation methods
@author: CJ Grady
"""
import glob
from hashlib import md5
import numpy as np
import os
from random import randint
import shutil
import tempfile

from extras.surfaceGenerator import SurfaceGenerator
from slr.common.costFunctions import seaLevelRiseCostFn

from slr.multiTile.wqMultiTile import MultiTileWqParallelDijkstraLCP
from slr.singleTile.dijkstra import SingleTileSerialDijkstraLCP
from slr.singleTile.parallelDijkstra import SingleTileParallelDijkstraLCP
from slr.tools.tileSplitter import splitTile

# .............................................................................
class TestCostDistance(object):
   """
   @summary: This class tests multiple classes to make sure that everything 
                produces the same result
   """

   # ............................
   def setup(self):
      """
      @summary: Set up the test
      """
      self.baseDir = self._getTemporaryDirectory(randint(0, 100000))
      print "Base directory for test:", self.baseDir

      self.surfaceFn = os.path.join(self.baseDir, "inSurface.asc")
      self.serialCostFn = os.path.join(self.baseDir, "serialCost.asc")
      self.parCostFn = os.path.join(self.baseDir, "parCost.asc")
      self.mtSurfaceDir = os.path.join(self.baseDir, "mtSurface")
      self.mtCostDir = os.path.join(self.baseDir, "mtCost")
      self.mtOutDir = os.path.join(self.baseDir, "mtOutput")
      self.mtCmpDir = os.path.join(self.baseDir, "mtCmpDir")
      self.mtSummaryFn = os.path.join(self.baseDir, "mtSummary.txt")
      
      # Make multitile directories
      os.mkdir(self.mtSurfaceDir)
      os.mkdir(self.mtCostDir)
      os.mkdir(self.mtOutDir)
      os.mkdir(self.mtCmpDir)
      
   # ............................
   def teardown(self):
      """
      @summary: Tear down the test
      """
      # Delete directory
      try:
         pass
         #shutil.rmtree(self.baseDir)
      except:
         pass
   
   # ............................
   def _getTemporaryDirectory(self, r):
      """
      @summary: Creates a temporary directory for testing
      """
      n = md5(str(r)).hexdigest()
      d = os.path.join(tempfile.gettempdir(), n)
      os.mkdir(d)
      return d
   
   # ............................
   def test_all_even(self):
      """
      @summary: This test creates a surface and runs each implementation of the
                   algorithm and checks that the results are the same
      """
      SURFACE_WIDTH = 1000
      SURFACE_HEIGHT = 1000
      STEP_SIZE = .2
      CELL_SIZE = .005
      TILE_SIZE = 1.0
      NUM_CONES = 100
      NUM_ELLIPSOIDS = 100
      MAX_HEIGHT = 50
      MAX_RADIUS = 200
      NUM_WORKERS = 1
      
      # Generate surface
      sg = SurfaceGenerator(SURFACE_HEIGHT, SURFACE_WIDTH, 0.0, 0.0, CELL_SIZE, 
                            defVal=0)
      sg.addRandom(numCones=NUM_CONES, numEllipsoids=NUM_ELLIPSOIDS, 
                   maxHeight=MAX_HEIGHT, maxRad=MAX_RADIUS)
      
      # Make sure that we have at least one source cell
      if np.min(sg.grid) > 0:
         sg.grid[0,0] = 0
      
      # Write out grid
      sg.writeGrid(self.surfaceFn)
      
      inputGrid = np.loadtxt(self.surfaceFn, dtype=int, comments='', skiprows=6)
      
      # Run Dijkstra
      serialInstance = SingleTileSerialDijkstraLCP(self.surfaceFn, 
                                                   self.serialCostFn, 
                                                   seaLevelRiseCostFn)
      serialInstance.findSourceCells()
      serialInstance.calculate()
   
      # Verify grid
      serialAry = np.loadtxt(self.serialCostFn, dtype=int, comments='', skiprows=6)
      self._verifyGrid(serialAry, inputGrid)
   
      # Run parallel Dijkstra
      parInstance = SingleTileParallelDijkstraLCP(self.surfaceFn,
                                                  self.parCostFn,
                                                  seaLevelRiseCostFn)
      parInstance.setMaxWorkers(16)
      parInstance.setStepSize(.20)
      parInstance.findSourceCells()
      parInstance.calculate()
      
      # Compare serial and parallel single tile
      parAry = np.loadtxt(self.parCostFn, dtype=int, comments='', skiprows=6)
      
      # Verify parallel run
      self._verifyGrid(parAry, inputGrid)
      
      #print len(np.where(serialAry != parAry))
      assert np.array_equal(serialAry, parAry)
      
      # Split surface
      splitTile(self.surfaceFn, TILE_SIZE, self.mtSurfaceDir)
      
      # Split cost surface
      splitTile(self.parCostFn, TILE_SIZE, self.mtCmpDir)
      
      # Run multi-tile
      mtInstance = MultiTileWqParallelDijkstraLCP(self.mtSurfaceDir,
                                                  self.mtCostDir,
                                                  self.mtOutDir,
                                                  TILE_SIZE,
                                                  STEP_SIZE,
                                                  summaryFn=self.mtSummaryFn)
   
      # Run
      print "Starting workers"
      mtInstance.startWorkers(NUM_WORKERS)
      mtInstance.calculate()
      
      # Only on success
      print "Stopping workers"
      mtInstance.stopWorkers()
      
      # Compare output directories
      assert self._checkOutputs(self.mtCostDir, self.mtCmpDir)
      
   # ............................
   def _checkOutputs(self, dir1, dir2):
      """
      @summary: Check that the outputs are what we expect from the compare 
                   directory
      @return: Boolean indicating if all of the files match
      """
      # Need to check that the files match
      assert len(glob.glob(os.path.join(dir1, "*.asc"))) == len(glob.glob(os.path.join(dir2, "*.asc")))
      
      # Need to check that the contents match
      for fn in glob.iglob(os.path.join(dir1, "*.asc")):
         n = os.path.basename(fn)
         print "Checking:", n
         a1 = np.loadtxt(fn, comments='', skiprows=6, dtype=int)
         a2 = np.loadtxt(os.path.join(dir2, n), comments='', skiprows=6, dtype=int)
         if not np.array_equal(a1, a2):
            #print n, "not equal"
            #print a1.tolist()
            #print
            #print a2.tolist()
            print np.where(a1 != a2)
            ys, xs = np.where(a1 != a2)
            print a1[ys[0],xs[0]], a2[ys[0],xs[0]]
            return False
      return True
   
   # ............................
   def _verifyGrid(self, costGrid, inGrid):
      # Check that cost and input grids have same shape
      assert costGrid.shape == inGrid.shape
      
      maxy, maxx = costGrid.shape
      
      for x in xrange(maxx):
         for y in xrange(maxy):
            cmpCells = []
            if y > 0:
               cmpCells.append(costGrid[y-1,x])
            if y < maxy-1:
               cmpCells.append(costGrid[y+1,x])
            if x > 0:
               cmpCells.append(costGrid[y,x-1])
            if x < maxx-1:
               cmpCells.append(costGrid[y,x+1])
            
            t1 = False
            if inGrid[y,x] == 0:
               t1 = costGrid[y,x] == 0
               if not t1:
                  print "(%s,%s) should have been source cell" % (x, y)
               assert t1
            
            # Only if it wasn't a source cell
            if not t1:
               tVal = max(inGrid[y,x], min(cmpCells))
               t2 = costGrid[y,x] == tVal
               if not t2:
                  print "(%s,%s) should have been %s, instead it is %s" % (x, y, tVal, costGrid[y,x])
               assert t2
