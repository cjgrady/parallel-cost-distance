"""
@summary: Test the Work Queue multitile module
@author: CJ Grady
@version: 1.0
@status: alpha
"""
import glob
from hashlib import md5
import numpy as np
import os
from random import randint
import shutil
import tempfile

from slr.multiTile.wqMultiTile import MultiTileWqParallelDijkstraLCP

from tests.helpers.testConstants import (ONE_DEGREE_TILE_DIR, 
                           ONE_DEGREE_TILE_COSTS_DIR, SIX_DEGREE_TILE_DIR, 
                           SIX_DEGREE_TILE_COSTS_DIR, SMALLER_TILE_DIR, 
                           SMALLER_TILE_COSTS_DIR, ONE_TILE_DIR,
                           ONE_TILE_COSTS_DIR, TWO_TILE_DIR, TWO_TILE_COSTS_DIR,
                           FOUR_TILE_DIR, FOUR_TILE_COSTS_DIR)
                           

NUM_WORKERS = 1 # Travis core limit

# .............................................................................
class TestWqParallelLCP(object):
   """
   @summary: This class tests the Work Queue Parallel Dijkstra implementation
   """
   # ............................
   def setup(self):
      """
      @summary: Set up the test
      """
      self.cDir = self._getTemporaryDirectory(randint(0, 10000))
      self.oDir = self._getTemporaryDirectory(randint(0, 10000))
      # We just want a name, so let it be deleted
      tf = tempfile.NamedTemporaryFile(delete=True)
      self.summaryFn = tf.name
      tf.close()
   
   # ............................
   def teardown(self):
      """
      @summary: Tear down the test
      """
      # Delete cost directory
      try:
         shutil.rmtree(self.cDir)
      except:
         pass
      
      # Delete output directory
      try:
         shutil.rmtree(self.oDir)
      except:
         pass
      
      # Delete summary file
      try:
         shutil.rmtree(self.summaryFn)
      except:
         pass
   
   # ............................
   def _checkOutputs(self, cmpDir):
      """
      @summary: Check that the outputs are what we expect from the compare 
                   directory
      @param cmpDir: This is a test data directory created by splitting the 
                        cost surface generated with the single surface mode
      @return: Boolean indicating if all of the files match
      """
      # Need to check that the files match
      assert len(glob.glob(os.path.join(self.cDir, "*.asc"))) == len(glob.glob(os.path.join(cmpDir, "*.asc")))
      
      # Need to check that the contents match
      for fn in glob.iglob(os.path.join(self.cDir, "*.asc")):
         n = os.path.basename(fn)
         print "Checking:", n
         a1 = np.loadtxt(fn, comments='', skiprows=6, dtype=int)
         a2 = np.loadtxt(os.path.join(cmpDir, n), comments='', skiprows=6, dtype=int)
         if not np.array_equiv(a1, a2):
            return False
      return True
         
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
   #def test_one_degree(self):
   #   """
   #   @summary: Test that the outputs are what we expect when using tile sizes
   #                that are evenly created from test surface
   #   """
   #   # Create instance
   #   myInstance = MultiTileWqParallelDijkstraLCP(ONE_DEGREE_TILE_DIR, 
   #                        self.cDir, self.oDir, 1.0, .2, 
   #                        summaryFn=self.summaryFn)
   #   # Run
   #   print "Starting workers"
   #   myInstance.startWorkers(NUM_WORKERS)
   #   try:
   #      myInstance.calculate()
   #   except Exception, e:
   #      print "stop workers (error)"
   #      myInstance.stopWorkers()
   #      raise e
   #   
   #   # Only on success
   #   print "Stopping workers"
   #   myInstance.stopWorkers()
   #   print "Done"
   #   
   #   # Compare outputs
   #   assert self._checkOutputs(ONE_DEGREE_TILE_COSTS_DIR)
   
   # ............................
   #def test_six_degree(self):
   #   """
   #   @summary: Test that the outputs are what we expect when using tile sizes
   #                that are not evenly created from test surface
   #   """
   #   # Create instance
   #   myInstance = MultiTileWqParallelDijkstraLCP(SIX_DEGREE_TILE_DIR, 
   #                        self.cDir, self.oDir, 6.0, .2, 
   #                        summaryFn=self.summaryFn)
   #   # Run
   #   print "Starting workers"
   #   myInstance.startWorkers(NUM_WORKERS)
   #   try:
   #      myInstance.calculate()
   #   except Exception, e:
   #      print "stop workers (error)"
   #      myInstance.stopWorkers()
   #      raise e
   #   
   #   # Compare outputs
   #   assert self._checkOutputs(SIX_DEGREE_TILE_COSTS_DIR)
   
   # ............................
   #def test_smaller(self):
   #   """
   #   @summary: Test that the outputs are what we expect when using tile sizes
   #                that are not evenly created from test surface
   #   """
   #   # Create instance
   #   myInstance = MultiTileWqParallelDijkstraLCP(SMALLER_TILE_DIR, 
   #                        self.cDir, self.oDir, 10.0, .2, 
   #                        summaryFn=self.summaryFn)
   #   # Run
   #   print "Starting workers"
   #   myInstance.startWorkers(NUM_WORKERS)
   #   try:
   #      myInstance.calculate()
   #   except Exception, e:
   #      print "stop workers (error)"
   #      myInstance.stopWorkers()
   #      raise e
   #   
   #   # Compare outputs
   #   assert self._checkOutputs(SMALLER_TILE_COSTS_DIR)
   
   # ............................
   def test_one_tile(self):
      """
      @summary: Test that the outputs are what we expect when using one tile
      """
      # Create instance
      myInstance = MultiTileWqParallelDijkstraLCP(ONE_TILE_DIR, 
                           self.cDir, self.oDir, 1.0, .2, 
                           summaryFn=self.summaryFn)
      # Run
      print "Starting workers"
      myInstance.startWorkers(NUM_WORKERS)
      try:
         myInstance.calculate()
      except Exception, e:
         print "stop workers (error)"
         myInstance.stopWorkers()
         raise e
      
      # Compare outputs
      assert self._checkOutputs(ONE_TILE_COSTS_DIR)
   
   # ............................
   #def test_two_tiles(self):
   #   """
   #   @summary: Test that the outputs are what we expect when using two tiles
   #   """
   #   # Create instance
   #   myInstance = MultiTileWqParallelDijkstraLCP(TWO_TILE_DIR, 
   #                        self.cDir, self.oDir, 1.0, .2, 
   #                        summaryFn=self.summaryFn)
   #   # Run
   #   print "Starting workers"
   #   myInstance.startWorkers(NUM_WORKERS)
   #   try:
   #      myInstance.calculate()
   #   except Exception, e:
   #      print "stop workers (error)"
   #      myInstance.stopWorkers()
   #      raise e
      
   #   # Compare outputs
   #   assert self._checkOutputs(TWO_TILE_COSTS_DIR)
   
   # ............................
   #def test_four_tiles(self):
   #   """
   #   @summary: Test that the outputs are what we expect when using four tiles
   #   """
   #   # Create instance
   #   myInstance = MultiTileWqParallelDijkstraLCP(FOUR_TILE_DIR, 
   #                        self.cDir, self.oDir, 1.0, .2, 
   #                        summaryFn=self.summaryFn)
   #   # Run
   #   print "Starting workers"
   #   myInstance.startWorkers(NUM_WORKERS)
   #   try:
   #      myInstance.calculate()
   #   except Exception, e:
   #      print "stop workers (error)"
   #      myInstance.stopWorkers()
   #      raise e
      
   #   # Compare outputs
   #   assert self._checkOutputs(FOUR_TILE_COSTS_DIR)
   
