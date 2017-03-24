"""
@summary: This is an experiment to collect statistics that may be used to find
             an "optimum" configuration of the tool for the *best performance
@author: CJ Grady
"""
import argparse
import glob
from hashlib import md5
import os
from random import randint
import shutil
import time

from slr.multiTile.wqMultiTile2 import MultiTileWqParallelDijkstraLCP

TILE_SIZES = [(0.5, 'half'), (1.0, 'one'), (2.0, 'two')]
PADDINGS = [1, 3, 5, 10]

# What are the configurations we will test
# What metrics will be used for testing

# .............................................................................
def createTemporaryDirectory(baseDir):
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
if __name__ == '__main__':
   
   parser = argparse.ArgumentParser()
   
   parser.add_argument('dataDir', help='Directory with region directories')
   parser.add_argument('statsFile', help='Where to write statistics')
   parser.add_argument('workDir', help='Where to do work')
   parser.add_argument('numWorkers', help='Just added to stats')

   args = parser.parse_args()
   
   if not os.path.exists(args.statsFile):
      writeHeaders = True
   else:
      writeHeaders = False
      
   with open(args.statsFile, 'a') as outF:
      if writeHeaders:
         outF.write("directory, tile size, number of tiles, number of tiles computed, cells changed, total time, padding, number of workers\n")

      regionDirs = glob.glob(os.path.join(args.dataDir, '*'))
      for region in regionDirs:
         for ts, d in TILE_SIZES:
            tileDir = os.path.join(region, d)
            for p in PADDINGS:
               costDir = createTemporaryDirectory(args.workDir)
               outDir = createTemporaryDirectory(args.workDir)
               m = MultiTileWqParallelDijkstraLCP(tileDir, costDir, outDir, ts, p)
               aTime = time.time()
               m.calculate()
               bTime = time.time()
               metrics = m.getMetrics()
         
               try:
                  shutil.rmtree(costDir)
                  shutil.rmtree(outDir)
               except Exception, e:
                  print str(e)
         
               tiles = set([])
               cc = 0
               for i in metrics:
                  tiles.add("{}-{}".format(i[0], i[1]))
                  cc += int(i[2])
               
               outF.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(tileDir, ts, 
                           len(tiles), len(metrics), cc, bTime - aTime, p, 
                           args.numWorkers))
   
