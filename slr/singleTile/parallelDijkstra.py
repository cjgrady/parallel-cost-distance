"""
@summary: This module contains a parallel implementation of Dijkstra's
             algorithm for graph traversal
@author: CJ Grady
@status: alpha
"""
import argparse
import concurrent.futures
import heapq
#import logging
#from logging.handlers import RotatingFileHandler
import numpy as np
import os
import time
import traceback

from slr.common.costFunctions import seaLevelRiseCostFn
from slr.singleTile.base import SingleTileLCP

# .............................................................................
class SingleTileParallelDijkstraLCP(SingleTileLCP):
   """
   @summary: This is the serial Dijkstra implementation
   """
   step = 25
   maxWorkers = 8
   chunks = []
   cellsChanged = 0

   def addDebug(self, txt):
      self.extras.append(txt)
      
   # ..........................
   def setStepSize(self, step):
      """
      @summary: Set the step size for parallelization within tile
      @param step: Step size as percentage of tile size (0.0, 1.0].  If greater
                      than 1, use as the step size
      """
      if step > 1.0:
         self.step = step
      else:
         self.step = int(self.cMtx.shape[0] * step)
      print "Set step size to:", self.step
      print "Cost matrix shape:", self.cMtx.shape
      
   # ..........................
   def setMaxWorkers(self, maxWorkers):
      """
      @summary: Set the maximum number of worker threads
      @param maxWorkers: Maximum number of threads to use
      """
      self.maxWorkers = maxWorkers
      
   # ..........................
   def _calculate(self):
      """
      @summary: Calculate the least cost path for all cells in the grid to a 
                   source cell
      @note: This method should be implemented in subclasses
      """
      # Get original edges
      try:
         self.origLeft = np.copy(self.cMtx[:,0])
         self.origRight = np.copy(self.cMtx[:, -1])
         self.origTop = np.copy(self.cMtx[0,:])
         self.origBottom = np.copy(self.cMtx[-1, :])
      except:
         raise Exception, self.cMtx.shape

      # Split into chunks
      # Track running chunks
      # Only one worker per chunk
      # When done
      #  Reassemble
      
      
      # Find chunks with source cells
      startChunks = {}
      for x,y in self.sourceCells:
         # Get min x, miny, maxx, maxy
         xSteps = int(x / self.step)
         ySteps = int(y / self.step)
         minX = xSteps * self.step
         minY = ySteps * self.step
         maxX = min(((xSteps+1)*self.step - 1), self.cMtx.shape[1]-1)
         maxY = min(((ySteps+1)*self.step - 1), self.cMtx.shape[0]-1)
                    
         # Make a key
         k = "%s - %s - %s - %s" % (minX, minY, maxX, maxY)
         
         if not startChunks.has_key(k):
            startChunks[k] = {'bbox' : (minX, minY, maxX, maxY),
                              'sourceCells' : []}
         # Add to dictionary with source cell
         startChunks[k]['sourceCells'].append((x, y))
      
      #print "Start chunks:", startChunks
      
      for k in startChunks.keys():
         minX, minY, maxX, maxY = startChunks[k]['bbox']
         self.chunks.append((minX, minY, startChunks[k]['sourceCells']))
      ts = []
      
      with concurrent.futures.ThreadPoolExecutor(max_workers=self.maxWorkers) as executor:
         ccc = True
         while ccc:
            if len(self.chunks) > 0:
               chunk = self.chunks.pop(0)
               t =  executor.submit(self._dijkstraChunk, chunk)
               ts.append(t)
            ggg = all([t.done() for t in ts])
            ccc = len(self.chunks) > 0 or not ggg
      
   # ..........................
   def _dijkstraChunk(self, chunk):
      #minx, miny, maxx, maxy, sourceCells = chunk
      minx, miny, sourceCells = chunk
      print "Working on chunk:", chunk
      
      maxx = min(minx+self.step, self.inMtx.shape[1])
      maxy = min(miny+self.step, self.inMtx.shape[0])
      
      name = '%s-%s-%s-%s' % (minx, miny, maxx, maxy)
      hq = []
      
      leftCells = []
      rightCells = []
      topCells = []
      bottomCells = []

      # ........................
      def addCell(x, y, cost):
         """
         @summary: Add a cell to the heap if appropriate
         """
         if int(self.cMtx[y,x]) == int(self.noDataValue) or self.cMtx[y,x] > cost:
            heapq.heappush(hq, (cost, x, y))
   
      # ........................
      def addNeighbors(x, y, cost):
         cellCost = self.inMtx[y,x]
         if int(cellCost) != int(self.noDataValue):
            if x - 1 >= minx:
               addCell(x-1, y, self.costFn(cost, cellCost, self.inMtx[y,x-1], self.cellSize))
            if x + 1 < maxx:
               addCell(x+1, y, self.costFn(cost, cellCost, self.inMtx[y,x+1], self.cellSize))
            if y-1 >= miny:
               addCell(x, y-1, self.costFn(cost, cellCost, self.inMtx[y-1,x], self.cellSize))
            if y+1 < maxy:
               addCell(x, y+1, self.costFn(cost, cellCost, self.inMtx[y+1,x], self.cellSize))

      # Check to see if source cells inundate anything
      for x, y in sourceCells:
         cmpx = x
         cmpy = y
         if x < minx:
            cmpx = minx
         if x >= maxx:
            cmpx = maxx
         if y < miny:
            cmpy = miny
         if y >= maxy:
            cmpy = maxy
         
         #TODO: This should use the cost function
         c = max(self.cMtx[y,x], self.inMtx[cmpy,cmpx], 0)
         
         if cmpx != x or cmpy != y:
            
            # Update cost if:
            #   - cost is no data
            #   - cost is greater than c
            #   - cost is less than 0
            #if int(self.cMtx[cmpy][cmpx]) == int(self.noDataValue) or \
            #    int(self.cMtx[cmpy][cmpx]) > c or \
            #    int(self.cMtx[cmpy][cmpx]) < 0:
            
            if int(self.cMtx[cmpy,cmpx]) == int(self.noDataValue) or \
               (int(self.cMtx[cmpy,cmpx]) > c and self.cMtx[cmpy,cmpx] >= 0):
               self.cMtx[cmpy,cmpx] = c
               
               # TODO: Evaluate if we should do this at edges
               self.cellsChanged += 1
               addNeighbors(cmpx, cmpy, c)
            else:
               #log.debug("Not adding: %s, %s, %s, %s, %s, %s" % (self.cMtx[cmpy][cmpx], self.noDataValue, self.cMtx[cmpy][cmpx], c, x, y))
               pass
         else:
            addNeighbors(x, y, c)
            
      # .........................
      # Dijkstra
      while len(hq) > 0:
         cost, x, y = heapq.heappop(hq)
         #log.debug("Popped %s, %s, %s" % (cost, x, y))
         #res.append("Popped %s, %s, %s" % (cost, x, y))

         # Update cost if:
         #   - cost is no data
         #   - cost is greater than c
         #   - cost is less than 0
         #if int(self.cMtx[cmpy][cmpx]) == int(self.noDataValue) or \
         #       int(self.cMtx[cmpy][cmpx]) > cost or \
         #       int(self.cMtx[cmpy][cmpx]) < 0:
         if int(self.cMtx[y,x]) == int(self.noDataValue) or (cost < int(self.cMtx[y,x]) and self.cMtx[y,x] >= 0):
            self.cMtx[y,x] = cost
            self.cellsChanged += 1
            #log.debug("Setting cost in matrix for (%s, %s) = %s ... %s" % (x, y, cost, self.cMtx[y][x]))
            addNeighbors(x, y, cost)
            # Should we spread?
            #res.append("x, y, min x and y, max x and y: (%s, %s) (%s, %s), (%s, %s)" % (x, y, minx, miny, maxx, maxy))
            if x == minx:
               #res.append("Adding a left cell")
               leftCells.append((x, y))
            if x == maxx-1:
               #res.append("Adding a right cell")
               rightCells.append((x, y))
            if y == maxy-1:
               #res.append("Adding a bottom cell")
               bottomCells.append((x, y))
            if y == miny:
               #res.append("Adding a top cell")
               topCells.append((x, y))
      
      # Spread
      # Assume that chunks start on left and top edges and partials may be at
      #    bottom and right
      
      if len(leftCells) > 0:
         if minx > 0:
            self.chunks.append((minx-self.step, miny, leftCells))
      if len(rightCells) > 0:
         if maxx < self.cMtx.shape[1]-1:
            self.chunks.append((minx+self.step, miny, rightCells))
      if len(topCells) > 0:
         if miny > 0:
            self.chunks.append((minx, miny-self.step, topCells))
      if len(bottomCells) > 0:
         if maxy < self.cMtx.shape[0]-1:
            self.chunks.append((minx, miny+self.step, bottomCells))
      #log.debug("Number of chunks: %s" % len(self.chunks))
      #log.debug("Shape: %s, %s" % self.cMtx.shape)
      print "Done with chunk:", chunk
   
   # .............................
   def writeChangedVectors(self, outDir, taskId='unknown', ts=1.0, dTime=0.0):
      self.newLeft = self.cMtx[:,0]
      self.newRight = self.cMtx[:,-1]
      self.newTop = self.cMtx[0,:]
      self.newBottom = self.cMtx[-1, :]
      
      
      l = t = r = b = False
      # Save left vector if changed
      
      
      # Look for problems where input data is not no data but output grid is
      
      # TODO: Couldn't I just check for inequality?
      
      if self.cellsChanged > 0:
         #if len(np.where(self.origLeft > self.newLeft) | (self.origLeft+1 >= self.noDataValue))[0]) > 0:
         #if len(np.where((np.round(self.origLeft, 1) -.1 > np.round(self.newLeft, 1)) | (self.origLeft+1 >= self.noDataValue))[0]) > 0:
         #if len(np.where((self.origLeft > self.newLeft) | (self.origLeft+1 >= self.noDataValue))[0]) > 0:
         if len(np.where(self.origLeft != self.newLeft)[0]) > 0:
            fn = os.path.join(outDir, '%s-toLeft.npy' % taskId)
            np.save(fn, self.newLeft)
            l = True
         
         #if len(np.where((np.round(self.origTop, 1) -.1 > np.round(self.newTop, 1)) | (self.origTop+1 >= self.noDataValue))[0]) > 0:
         #if len(np.where((self.origTop > self.newTop) | (self.origTop+1 >= self.noDataValue))[0]) > 0:
         if len(np.where(self.origTop != self.newTop)[0]) > 0:
            fn = os.path.join(outDir, '%s-toTop.npy' % taskId)
            np.save(fn, self.newTop)
            t = True
         
         #if len(np.where((np.round(self.origRight, 1) -.1 > np.round(self.newRight, 1)) | (self.origRight+1 >= self.noDataValue))[0]) > 0:
         #if len(np.where((self.origRight > self.newRight) | (self.origRight+1 >= self.noDataValue))[0]) > 0:
         if len(np.where(self.origRight != self.newRight)[0]) > 0:
            fn = os.path.join(outDir, '%s-toRight.npy' % taskId)
            np.save(fn, self.newRight)
            r = True
         
         #if len(np.where((np.round(self.origBottom, 1) -.1 > np.round(self.newBottom, 1)) | (self.origBottom >= self.noDataValue))[0]) > 0:
         #if len(np.where((self.origBottom > self.newBottom) | (self.origBottom >= self.noDataValue))[0]) > 0:
         if len(np.where(self.origBottom != self.newBottom)[0]) > 0:
            fn = os.path.join(outDir, '%s-toBottom.npy' % taskId)
            np.save(fn, self.newBottom)
            b = True
            
         # Write out cost grid as it stands
         np.savetxt(os.path.join(outDir, '%s-grid.asc' % taskId), self.cMtx,
                    header=self.headers, fmt="%i", comments='')
                    
      with open(os.path.join(outDir, '%s-summary.txt' % taskId), 'w') as outF:
         outF.write("%s\n" % self.minLong)# Min x
         outF.write("%s\n" % self.minLat) # Min y
         try:
            newMaxLong = self.minLong + ts
         except Exception, e:
            newMaxLong = str(e)
         outF.write("%s\n" % newMaxLong) # Max x
         outF.write("%s\n" % (self.minLat + ts)) # Maxy y
         outF.write("%s\n" % l) # Left modified
         outF.write("%s\n" % t) # Top modified
         outF.write("%s\n" % r) # Right modified
         outF.write("%s\n" % b) # Bottom modified
         outF.write("%s\n" % self.cellsChanged)
         outF.write("%s\n" % dTime)
         outF.write("%s\n" % ','.join(["(%s, %s)" % (x, y) for x, y in self.sourceCells]))
         outF.write("%s\n" % '\n'.join(self.extras))
      
# .............................................................................
if __name__ == "__main__": # pragma: no cover
   
   aTime = time.time()
   
   parser = argparse.ArgumentParser()
   
   parser.add_argument("dem")
   parser.add_argument("costSurface")
   parser.add_argument('-g', help="Generate edge vectors for modified cells")
   parser.add_argument('-t', '--taskId', help="Use this task id for outputs")
   parser.add_argument('-v', '--vect', help="Use this vector for source cells", nargs="*")
   parser.add_argument('-s', '--fromSide', type=int, help="Source vector is from this side 0: left, 1: top, 2: right, 3: bottom", nargs="*")
   parser.add_argument('-o', help="File to write outputs")
   parser.add_argument('-w', type=int, help="Maximum number of worker threads")
   parser.add_argument('--step', type=float, help="The step size to use")
   parser.add_argument('--ts', type=float)
   parser.add_argument('-e', type=str, help="Log errors to this file location")

   args = parser.parse_args()
   print args
   
   try:
      tile = SingleTileParallelDijkstraLCP(args.dem, args.costSurface, seaLevelRiseCostFn)
   
      if args.fromSide is None or args.vect is None or len(args.fromSide) == 0 or len(args.vect) == 0:
         tile.findSourceCells()
      else:
         tile.addDebug(str(args.vect))
         tile.addDebug(str(args.fromSide))
         for sVect, fromDir in zip(args.vect, args.fromSide):
            tile.addDebug("Source vector: %s" % str(sVect))
            tile.addDebug("From dir: %s" % str(fromDir))
            sourceVector = np.load(sVect)
            tile.addSourceVector(sourceVector, fromDir)
      
      if args.w is not None:
         tile.setMaxWorkers(args.w)
      else:
         tile.setMaxWorkers(50)
      
      print "Set step size?"   
      if args.step is not None:
         print "Setting step size to:", args.step
         tile.setStepSize(args.step)
      else:
         tile.setStepSize(.15)
   
   
      tile.calculate()
      
      if args.taskId is not None:
         taskId = args.taskId
      else:
         taskId = 'unknownTask'
      
      bTime = time.time()
      dTime = bTime - aTime
      if args.g is not None:
         outDir = args.o
         tile.writeChangedVectors(outDir, taskId, ts=args.ts, dTime=dTime)
   except Exception, e:
      tb = traceback.format_exc()
      if args.e is not None:
         with open(args.e, 'w') as outF:
            outF.write(str(e))
            outF.write(tb)
         
            
