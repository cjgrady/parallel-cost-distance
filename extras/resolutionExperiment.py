"""
@summary: This module performs an experiment over different tile and step resolutions
"""
import argparse
import glob
from hashlib import md5
import os
from random import randint
import shutil
import tempfile
import time

from slr.common.costFunctions import seaLevelRiseCostFn
from slr.multiTile.wqMultiTile import MultiTileWqParallelDijkstraLCP


STEP_SIZES = [1.0, 0.5, 0.34, 0.25, 0.1]

TWO_DEGREE = "two"
ONE_DEGREE = "one"
HALF_DEGREE = "half"

TILE_SIZES = [(.5, HALF_DEGREE), (1.0, ONE_DEGREE), (2.0, TWO_DEGREE)]

# .............................................................................
def _getTemporaryDirectory(baseDir):
   """
   @summary: Creates a temporary directory for testing
   """
   r = randint(0, 1000000)
   n = md5(str(r)).hexdigest()
   d = os.path.join(baseDir, n)
   while os.path.exists(d):
      r = randint(0, 1000000)
      n = md5(str(r)).hexdigest()
      d = os.path.join(baseDir, n)
   os.mkdir(d)
   return d

# .............................................................................
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   
   parser.add_argument("inDir", help="Input directory")
   parser.add_argument("statsFn", help="A file to write statistics to")
   parser.add_argument("statsDir", help="Where to write individual stats")
   parser.add_argument("tempDir", help="A location to write temporary files")
   
   args = parser.parse_args()
   
   print "Running"
   
   # Write headers?
   headers = ["Region", "tile_size", "step_size", "running_time", 
              "cells_changed", "tiles_computed"]

   writeHeaders = not os.path.exists(args.statsFn)

   with open(args.statsFn, 'a') as outF:
      
      # Write headers if new file
      if writeHeaders:
         outF.write("%s\n" % ','.join(headers))
      
      totalNum = len(glob.glob(os.path.join(args.inDir, '*'))) * len(TILE_SIZES) * len(STEP_SIZES)
      num = 0
      
      for testDir in glob.glob(os.path.join(args.inDir, '*')):
         print "Test directory:", testDir
         region = os.path.basename(testDir)
         
         for ts, tileDir in TILE_SIZES:
            
            for ss in STEP_SIZES:
               num += 1
               print "Number:", num, "of", totalNum
               statsFn = os.path.join(args.statsDir, 
                                    "stat-{0}-{1}-{2}.stats".format(region, ts, ss))
               print statsFn, os.path.exists(statsFn)
               
               if not os.path.exists(statsFn):
                  tileInDir = os.path.join(testDir, tileDir)
                  tileCostDir = _getTemporaryDirectory(args.tempDir)
                  tileOutDir = _getTemporaryDirectory(args.tempDir)
                  
                  # Probably don't need stats file
                  
                  atime = time.time()
                  lcp = MultiTileWqParallelDijkstraLCP(tileInDir, tileCostDir,
                                                          tileOutDir, ts, ss,
                                                          summaryFn=statsFn)
                  lcp.calculate()
                  btime = time.time()
                  
                  dtime = btime - atime
                  
                  outF.write("{0},{1},{2},{3},{4},{5}\n".format(region, ts, ss, 
                                                        dtime, lcp.cc, lcp.tc))
      
                  lcp = None
      
                  # Clean up directories created
                  #try:
                  #   shutil.rmtree(tileCostDir)
                  #   shutil.rmtree(tileOutDir)
                  #except Exception, e:
                  #   print str(e)
                  