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
from slr.singleTile.parallelDijkstra import SingleTileParallelDijkstraLCP

STEP_SIZES = [1.0, 0.5, 0.34, 0.25, 0.1]

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
   headers = ["Region", "step_size", "running_time"]

   writeHeaders = not os.path.exists(args.statsFn)

   with open(args.statsFn, 'a') as outF:
      
      # Write headers if new file
      if writeHeaders:
         outF.write("%s\n" % ','.join(headers))
      
      totalNum = len(glob.glob(os.path.join(args.inDir, '*.asc'))) * len(STEP_SIZES)
      num = 0
      
      for inputAscii in glob.glob(os.path.join(args.inDir, '*.asc')):
         print "Input grid:", inputAscii
         
         for ss in STEP_SIZES:
            num += 1
            print "Number:", num, "of", totalNum
            statsFn = os.path.join(args.statsDir, 
                                 "stat-{0}-{1}.stats".format(inputAscii, ss))
            print statsFn, os.path.exists(statsFn)
            
            if not os.path.exists(statsFn):
               tileCostDir = _getTemporaryDirectory(args.tempDir)
               costFn = os.path.join(tileCostDir, 'cost.asc')
               
               # Probably don't need stats file
               
               atime = time.time()
               lcp = SingleTileParallelDijkstraLCP(inputAscii, costFn, seaLevelRiseCostFn)
               lcp.calculate()
               btime = time.time()
               
               dtime = btime - atime
               
               outF.write("{0},{1},{2}\n".format(inputAscii, ss, dtime))
   
               lcp = None
   
               # Clean up directories created
               #try:
               #   shutil.rmtree(tileCostDir)
               #   shutil.rmtree(tileOutDir)
               #except Exception, e:
               #   print str(e)
               