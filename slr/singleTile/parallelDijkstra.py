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
from concurrent.futures import process

#import uuid
#import logging
#logging.basicConfig(filename='%s.log' % str(uuid.uuid4().hex), loglevel=logging.)


TASK_WAIT_FOR_LOCK_TIME = .01
COST_KEY = "cost"
INPUT_KEY = "input"
LOCK_WAIT_TIME = .01
FROM_LEFT_KEY = "fromLeft"
FROM_RIGHT_KEY = "fromRight"
FROM_TOP_KEY = "fromTop"
FROM_BOTTOM_KEY = "fromBottom"
WAIT_TIME = .1

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
      #myName = str(uuid.uuid4().hex)
      #logger = logging.getLogger(myName)
      #hdlr = logging.FileHandler('/tmp/logs/%s.log' % myName)
      #formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
      #hdlr.setFormatter(formatter)
      #logger.addHandler(hdlr) 
      #logger.setLevel(logging.DEBUG)
      
      # ..........................
      def _getKey(minx, miny):
         return "{minx}-{miny}".format(minx=minx, miny=miny)
      
      #logger.debug("start of calculate")
      yLen, xLen = self.cMtx.shape
      #logger.debug("y len: %s, xlen: %s"  % (yLen, xLen))
      
      # Get original edges
      try:
         self.origLeft = np.copy(self.cMtx[:,0])
         self.origRight = np.copy(self.cMtx[:, -1])
         self.origTop = np.copy(self.cMtx[0,:])
         self.origBottom = np.copy(self.cMtx[-1, :])
      except:
         raise Exception, self.cMtx.shape

      # Initialize management objects
      resultsQueue = []
      resultsLock = False
      chunks = {}
      processingChunks = {}
      
      #logger.debug("Split into chunks")
      # Split into chunks
      for y in xrange(0, yLen, self.step):
         for x in xrange(0, xLen, self.step):
            key = _getKey(x, y)
            chunks[key] = {
             INPUT_KEY : self.inMtx[y:y+self.step,x:x+self.step],
             COST_KEY : self.cMtx[y:y+self.step,x:x+self.step]
            }
      
      #logger.debug("Chunks:")
      #logger.debug(str(chunks.keys()))
      
      # Initialize executor
      with concurrent.futures.ThreadPoolExecutor(max_workers=self.maxWorkers) as executor:
         
         # task callback function
         def taskCallback(task):
            """
            @summary: Callback after task completion
            @param task: The completed task
            """
            if task.cancelled():
               #TODO: Handle cancelled
               raise Exception, "Task cancelled"
            elif task.done():
               error = task.exception()
               if error:
                  # TODO: Handle error
                  raise Exception, "Error (%s, %s): %s, %s" % (task.minx, task.miny, error, str(dir(error)))
               else:
                  result = task.result()
                  # Get minx, miny from task
                  minx = task.minx
                  miny = task.miny
                  key = _getKey(minx, miny)
            
                  # Get cost surface and inundated edges
                  costSurface, left, right, top, bottom = result
                  
                  # Update cost surface
                  chunks[key][COST_KEY] = costSurface
                  
                  # Check for lock (wait until open)
                  if resultsLock:
                     time.sleep(TASK_WAIT_FOR_LOCK_TIME)
                  # Append to results queue
                  resultsQueue.append((minx, miny, left, right, top, bottom))
      
         # Process source cells
         sourceChunks = {}
         # Loop through source cells
         for x,y in self.sourceCells:
            # Find which chunk it belongs in
            # Use integer division and multiplication to find chunk
            minx = self.step * (x / self.step)
            miny = self.step * (y / self.step)
            modx = x-minx
            mody = y-miny
            #print x, y, minx, miny, modx, mody
            # Get key
            key = _getKey(minx, miny)
            # Add or append source cell to chunk
            if sourceChunks.has_key(key):
               sourceChunks[key]['sources'].append((modx, mody))
            else:
               sourceChunks[key] = {
                  'sources' : [(modx, mody)],
                  'minx' : minx,
                  'miny' : miny
               }

         #logger.debug("Source chunks:")
         #logger.debug(str(sourceChunks.keys()))
         
         # Submit source chunks
         for key in sourceChunks.keys():
            t = executor.submit(self._dijkstraChunk, (chunks[key][INPUT_KEY],
                  chunks[key][COST_KEY], None, None, None, None, sourceChunks[key]['sources']))
            t.minx = sourceChunks[key]['minx']
            t.miny = sourceChunks[key]['miny']
            t.add_done_callback(taskCallback)
            processingChunks[key] = {
               FROM_LEFT_KEY : None,
               FROM_RIGHT_KEY : None,
               FROM_TOP_KEY : None,
               FROM_BOTTOM_KEY : None
            }
            
         #logger.debug("Processing chunks:")
         #logger.debug(str(processingChunks.keys()))
         
         # Loop until done
         cont = len(processingChunks.keys()) > 0
         while cont:
            # Look for results
            #logger.debug("Len results queue: %s" % len(resultsQueue))
            #logger.debug(str(resultsQueue))
            
            if len(resultsQueue) > 0:
               # Lock
               resultsLock = True
               # Wait for any results that passed lock and need to set
               time.sleep(LOCK_WAIT_TIME)
               
               # TODO: Do we need to lock if use this method?
               # Process results queue
               while len(resultsQueue) > 0:
                  minx, miny, left, right, top, bottom = resultsQueue.pop(0)
                  key = _getKey(minx, miny)
                  
                  # Remove from processing queue
                  d = processingChunks.pop(key)
                  #logger.debug("Popped: %s" % key)
                  
                  # Resubmit if waiting
                  if d[FROM_LEFT_KEY] is not None or d[FROM_RIGHT_KEY] is not None or \
                       d[FROM_TOP_KEY] is not None or d[FROM_BOTTOM_KEY] is not None:
                     # Put back into processing
                     processingChunks[key] = {
                        FROM_LEFT_KEY: None,
                        FROM_RIGHT_KEY: None,
                        FROM_TOP_KEY: None,
                        FROM_BOTTOM_KEY: None
                     }
                     
                     # Make chunk
                     chunk = (chunks[key][INPUT_KEY], chunks[key][COST_KEY], d[FROM_LEFT_KEY],
                                d[FROM_RIGHT_KEY], d[FROM_TOP_KEY], d[FROM_BOTTOM_KEY], None)
                     # Submit
                     #logger.debug("Resubmitting: %s" % key)
                     t = executor.submit(self._dijkstraChunk, chunk)
                     t.minx = minx
                     t.miny = miny
                     # Add callback
                     t.add_done_callback(taskCallback)
                  
                  # Propagate calculations
                  # Left
                  if left is not None and minx - self.step >= 0:
                     leftKey = _getKey(minx-self.step, miny)
                     edge = chunks[key][COST_KEY][:,0]
                     if processingChunks.has_key(leftKey):
                        processingChunks[leftKey][FROM_RIGHT_KEY] = edge
                     else:
                        # Submit left chunk
                        leftChunk = (chunks[leftKey][INPUT_KEY], chunks[leftKey][COST_KEY],
                                      None, edge, None, None, None)
                        #logger.debug("Submitting: %s" % leftKey)
                        lt = executor.submit(self._dijkstraChunk, leftChunk)
                        lt.minx = minx-self.step
                        lt.miny = miny
                        lt.add_done_callback(taskCallback)
                        processingChunks[leftKey] = {
                           FROM_LEFT_KEY: None,
                           FROM_RIGHT_KEY: None,
                           FROM_TOP_KEY: None,
                           FROM_BOTTOM_KEY: None
                        }
                        
                  # Right
                  if right is not None and minx + self.step < xLen:
                     rightKey = _getKey(minx+self.step, miny)
                     edge = chunks[key][COST_KEY][:,-1]
                     if processingChunks.has_key(rightKey):
                        processingChunks[rightKey][FROM_LEFT_KEY] = edge
                     else:
                        # Submit right chunk
                        rightChunk = (chunks[rightKey][INPUT_KEY], chunks[rightKey][COST_KEY],
                                       edge, None, None, None, None)
                        #logger.debug("Submitting: %s" % rightKey)
                        rt = executor.submit(self._dijkstraChunk, rightChunk)
                        rt.minx = minx+self.step
                        rt.miny = miny
                        rt.add_done_callback(taskCallback)
                        processingChunks[rightKey] = {
                           FROM_LEFT_KEY: None,
                           FROM_RIGHT_KEY: None,
                           FROM_TOP_KEY: None,
                           FROM_BOTTOM_KEY: None
                        }
                        
                  # Top
                  if top is not None and miny - self.step >= 0:
                     topKey = _getKey(minx, miny-self.step)
                     edge = chunks[key][COST_KEY][0,:]
                     if processingChunks.has_key(topKey):
                        processingChunks[topKey][FROM_BOTTOM_KEY] = edge
                     else:
                        # Submit top chunk
                        topChunk = (chunks[topKey][INPUT_KEY], chunks[topKey][COST_KEY],
                                       None, None, None, edge, None)
                        #logger.debug("Submitting: %s" % topKey)
                        tt = executor.submit(self._dijkstraChunk, topChunk)
                        tt.minx = minx
                        tt.miny = miny - self.step
                        tt.add_done_callback(taskCallback)
                        processingChunks[topKey] = {
                           FROM_LEFT_KEY: None,
                           FROM_RIGHT_KEY: None,
                           FROM_TOP_KEY: None,
                           FROM_BOTTOM_KEY: None
                        }
                        
                  # Bottom
                  if bottom is not None and miny + self.step < yLen:
                     bottomKey = _getKey(minx, miny+self.step)
                     edge = chunks[key][COST_KEY][-1,:]
                     if processingChunks.has_key(bottomKey):
                        processingChunks[bottomKey][FROM_TOP_KEY] = edge
                     else:
                        # Submit bottom chunk
                        bottomChunk = (chunks[bottomKey][INPUT_KEY], chunks[bottomKey][COST_KEY],
                                         None, None, edge, None, None)
                        #logger.debug("Submitting: %s" % bottomKey)
                        bt = executor.submit(self._dijkstraChunk, bottomChunk)
                        bt.minx = minx
                        bt.miny = miny+self.step
                        bt.add_done_callback(taskCallback)
                        processingChunks[bottomKey] = {
                           FROM_LEFT_KEY: None,
                           FROM_RIGHT_KEY: None,
                           FROM_TOP_KEY: None,
                           FROM_BOTTOM_KEY: None
                        }
                  
               # Should we continue?
               cont = len(processingChunks.keys()) > 0
               #logger.debug("Should we continue? %s" % str(cont))
               #print len(processingChunks.keys())
               #print processingChunks.keys()

               # Unlock
               resultsLock = False
            #logger.debug("Sleeping")
            time.sleep(WAIT_TIME)
            #logger.debug("Awake")
      
      # Resassemble
      #print "Reassemble"
      #logger.debug("Reassembling")
      for y in xrange(0, yLen, self.step):
         for x in xrange(0, xLen, self.step):
            key = _getKey(x, y)
            self.cMtx[y:y+self.step, x:x+self.step] = chunks[key][COST_KEY]
            #print key
            #print self.cMtx[y:y+self.step, x:x+self.step]
      #logger.debug("Done")

   # ..........................
   def _dijkstraChunk(self, chunk):
      try:
         inSurface, costSurface, leftVector, rightVector, topVector, \
            bottomVector, sourceCells = chunk
   
         maxy, maxx = costSurface.shape
         
         # Make sure source cells is a list
         if sourceCells is None:
            sourceCells = []
         
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
            if int(costSurface[y,x]) == int(self.noDataValue) or costSurface[y,x] > cost:
               heapq.heappush(hq, (cost, x, y))
      
         # ........................
         def addNeighbors(x, y, cost):
            cellCost = inSurface[y,x]
            if int(cellCost) != int(self.noDataValue):
               if x - 1 >= 0:
                  addCell(x-1, y, self.costFn(cost, cellCost, inSurface[y,x-1], self.cellSize))
               if x + 1 < maxx:
                  addCell(x+1, y, self.costFn(cost, cellCost, inSurface[y,x+1], self.cellSize))
               if y - 1 >= 0:
                  addCell(x, y-1, self.costFn(cost, cellCost, inSurface[y-1,x], self.cellSize))
               if y + 1 < maxy:
                  addCell(x, y+1, self.costFn(cost, cellCost, inSurface[y+1,x], self.cellSize))
   
         # Process source vectors
         # Left
         if leftVector is not None:
            for y in xrange(maxy):
               if costSurface[y,0] < 0:
                  c = max(inSurface[y,0], leftVector[y])
               else:
                  c = max(inSurface[y,0], min(leftVector[y], costSurface[y,0]))
               if c < costSurface[y,0] or costSurface[y,0] < 0:
                  costSurface[y,0] = c
                  sourceCells.append((0, y))
                  if y == 0:
                     topCells.append((0, y))
                  elif y == maxy - 1:
                     bottomCells.append((0, y))
         # Right
         if rightVector is not None:
            for y in xrange(maxy):
               if costSurface[y,-1] < 0:
                  c = max(inSurface[y,-1], rightVector[y])
               else:
                  c = max(inSurface[y,-1], min(rightVector[y], costSurface[y,-1]))
               if c < costSurface[y,-1] or costSurface[y,-1] < 0:
                  costSurface[y,-1] = c
                  sourceCells.append((maxx-1, y))
                  if y == 0:
                     topCells.append((maxx-1, y))
                  elif y == maxy-1:
                     bottomCells.append((maxx-1, y))
         # Top
         if topVector is not None:
            for x in xrange(maxx):
               if costSurface[0,x] < 0:
                  c = max(inSurface[0,x], topVector[x])
               else:
                  c = max(inSurface[0,x], min(topVector[x], costSurface[0,x]))
               if c < costSurface[0,x] or costSurface[0,x] < 0:
                  costSurface[0,x] = c
                  sourceCells.append((x, 0))
                  if x == 0:
                     leftCells.append((x,0))
                  elif x == maxx-1:
                     rightCells.append((x,0))
         # Bottom
         if bottomVector is not None:
            for x in xrange(maxx):
               if costSurface[-1,x] < 0:
                  c = max(inSurface[-1,x], bottomVector[x])
               else:
                  c = max(inSurface[-1,x], min(bottomVector[x], costSurface[-1,x]))
               if c < costSurface[-1,x] or costSurface[-1,x] < 0:
                  costSurface[-1,x] = c
                  sourceCells.append((x, maxy-1))
                  if x == 0:
                     leftCells.append((x, maxy-1))
                  elif x == maxx-1:
                     rightCells.append((x, maxy-1))
                     
         
         # Check to see if source cells inundate anything
         for x, y in sourceCells:
            
            #TODO: This should use the cost function
            c = max(costSurface[y,x], inSurface[y,x], 0)
            addNeighbors(x, y, c)
               
         # .........................
         # Dijkstra
         while len(hq) > 0:
            cost, x, y = heapq.heappop(hq)
   
            # Update cost if:
            #   - cost is no data
            #   - cost is greater than c
            #   - cost is less than 0
            if int(costSurface[y,x]) == int(self.noDataValue) or \
                (cost < int(costSurface[y,x]) and costSurface[y,x] >= 0):
               costSurface[y,x] = cost
               self.cellsChanged += 1
               #log.debug("Setting cost in matrix for (%s, %s) = %s ... %s" % (x, y, cost, self.cMtx[y][x]))
               addNeighbors(x, y, cost)
               # Should we spread?
               #res.append("x, y, min x and y, max x and y: (%s, %s) (%s, %s), (%s, %s)" % (x, y, minx, miny, maxx, maxy))
               if x == 0:
                  #res.append("Adding a left cell")
                  leftCells.append((x, y))
               if x == maxx-1:
                  #res.append("Adding a right cell")
                  rightCells.append((x, y))
               if y == maxy-1:
                  #res.append("Adding a bottom cell")
                  bottomCells.append((x, y))
               if y == 0:
                  #res.append("Adding a top cell")
                  topCells.append((x, y))
         
         # Spread
         # Assume that chunks start on left and top edges and partials may be at
         #    bottom and right
         leftVect = rightVect = topVect = bottomVect = None
         if len(leftCells) > 0:
            leftVect = costSurface[:,0]
         if len(rightCells) > 0:
            rightVect = costSurface[:,-1]
         if len(topCells) > 0:
            topVect = costSurface[0,:]
         if len(bottomCells) > 0:
            bottomVect = costSurface[-1,:]
   
         return costSurface, leftVect, rightVect, topVect, bottomVect
      except Exception, e:
         msg = traceback.format_exc()
         raise Exception, str(msg)
      
   # .............................
   def writeChangedVectors(self, outDir, taskId='unknown', ts=1.0, dTime=0.0): 
      self.newLeft = self.cMtx[:,0]
      self.newRight = self.cMtx[:,-1]
      self.newTop = self.cMtx[0,:]
      self.newBottom = self.cMtx[-1, :]
      
      
      l = t = r = b = False
      # Save left vector if changed
      
      
      # Look for problems where input data is not no data but output grid is
      
      if self.cellsChanged > 0:
         if len(np.where(self.origLeft != self.newLeft)[0]) > 0:
            fn = os.path.join(outDir, '%s-toLeft.npy' % taskId)
            np.save(fn, self.newLeft)
            l = True
         
         if len(np.where(self.origTop != self.newTop)[0]) > 0:
            fn = os.path.join(outDir, '%s-toTop.npy' % taskId)
            np.save(fn, self.newTop)
            t = True
         
         if len(np.where(self.origRight != self.newRight)[0]) > 0:
            fn = os.path.join(outDir, '%s-toRight.npy' % taskId)
            np.save(fn, self.newRight)
            r = True
         
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
   import logging
   logging.basicConfig()
   
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
         print tile.sourceCells
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
         
            
