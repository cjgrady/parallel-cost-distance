"""
@summary: This benchmarking script is used to track the number of tiles computed
             and how many cells are changed per tile computation
"""

import argparse
import glob
from hashlib import md5
import os
from random import randint

from slr.multiTile.wqMultiTile import MultiTileWqParallelDijkstraLCP

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
   
   parser.add_argument('benchmarkFile', type=str, 
                       help="File location to write benchmarks")
   
   parser.add_argument('testDirGlob', type=str, 
                       help='Glob string for test directories')
   
   parser.add_argument('workDir', type=str)
   parser.add_argument('tileSize', type=float)
   parser.add_argument('stepSize', type=float)
   
   args = parser.parse_args()

   with open(args.benchmarkFile, 'w') as bmF:
      bmF.write("dirname, numTiles, numTilesCalculated, totalCellsComputed\n")
   
      for d in glob.glob(args.testDirGlob):
         costDir = createTemporaryDirectory(args.workDir)
         outDir = createTemporaryDirectory(args.workDir)
         m = MultiTileWqParallelDijkstraLCP(d, costDir, outDir, args.tileSize, args.stepSize)
         m.calculate()
         metrics = m.getMetrics()
         
         tiles = set([])
         cc = 0
         for i in metrics:
            tiles.add("{}-{}".format(i[0], i[1]))
            cc += int(i[2])
         
         bmF.write("{}, {}, {}, {}\n".format(d, len(tiles), len(metrics), cc))
   