"""
@summary: This module generates several runs of the single tile parallel 
             Dijkstra tool.  The purpose of this experiment is to generate 
             statistics that may be used to determine the optimal configuration.
"""
import argparse
import glob
import numpy as np
import os
import time

from extras.moransI import moransI
from slr.common.costFunctions import seaLevelRiseCostFn
from slr.singleTile.parallelDijkstra import SingleTileParallelDijkstraLCP

# Test constants
STEP_SIZES = [1.0, 0.5, .34, .25, .1]

# .............................................................................
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   
   parser.add_argument("surfacesDir", help="Directory containing surfaces to test")
   parser.add_argument("outputDir", help="Directory to write outputs")
   parser.add_argument("statsFn", help="File location to write the stats to")
   
   args = parser.parse_args()
   
   searchPath = os.path.join(args.surfacesDir, "*.asc")

   headers = ["filename", "rows", "columns", "stepSize", "running_time", "min", 
              "max", "mean", "stddev", "variance", "moransI", "cellsChanged", 
              "regionsComputed", "cellsAboveElev"]
   
   writeHeaders = not os.path.exists(args.statsFn)

   with open(args.statsFn, 'a') as outF:
      
      # Write headers if new file
      if writeHeaders:
         outF.write("%s\n" % ','.join(headers))
      
      for ss in STEP_SIZES:
         for fn in glob.glob(searchPath):
            print "Step size:", ss
            print "Filename:", fn
            
            tileStatsFn = os.path.join(args.outputDir, '%s-%s.csv' % (ss, os.path.basename(fn)))
            if not os.path.exists(tileStatsFn):
               try:
                  costFn = 'cost.asc'
                  atime = time.time()
                  tile = SingleTileParallelDijkstraLCP(fn, costFn, seaLevelRiseCostFn)
                  tile.setMaxWorkers(50)
                  tile.findSourceCells()
                  tile.setStepSize(ss)
                  tile.calculate()
                  btime = time.time()
                  tile.writeStats(os.path.join(args.outputDir, '%s-%s.csv' % (ss, os.path.basename(fn))))
                  
                  dtime = btime - atime
            
                  mtx = np.loadtxt(fn, dtype=int, skiprows=6)
                  rows, cols = mtx.shape
                  
                  outF.write("%s," % fn) # File name
                  outF.write("%s," % rows) # Rows
                  outF.write("%s," % cols) # Columns
                  outF.write("%s," % ss) # Step size
                  outF.write("%s," % dtime) # Running time
                  outF.write("%s," % mtx.min()) # Minimum
                  outF.write("%s," % mtx.max()) # Maximum
                  outF.write("%s," % mtx.mean()) # Mean
                  outF.write("%s," % mtx.std()) # Standard deviation
                  outF.write("%s," % mtx.var()) # Variance
                  outF.write("%s," % moransI(mtx.tolist())) # Moran's I
                  outF.write("%s," % tile.cellsChanged) # Cells changed
                  outF.write("%s," % len(tile.stats)) # Regions computed
                  outF.write("%s\n" % len(np.where(tile.cMtx>mtx)[0])) # Cells above elevation
                  tile.stats = []
                  tile.cMtx = None
                  tile = None
                  mtx = None
               except Exception, e:
                  print str(e)
               
               try:
                  os.remove(costFn)
               except:
                  pass
