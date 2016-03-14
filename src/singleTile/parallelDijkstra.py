"""
@summary: This module contains a parallel implementation of Dijkstra's
             algorithm for graph traversal
@author: CJ Grady
@status: alpha
"""
import concurrent.futures
import heapq
import logging
from logging.handlers import RotatingFileHandler
import os
from time import sleep

from singleTile.base import SingleTileLCP

LOG_PATH = '/home/cjgrady/logs/'

# .............................................................................
class SingleTileParallellDijkstraLCP(SingleTileLCP):
   """
   @summary: This is the serial Dijkstra implementation
   """
   step = 25
   maxWorkers = 8
   chunks = []
   
   # ..........................
   def setStepSize(self, step):
      self.step = step
      
   # ..........................
   def setMaxWorkers(self, maxWorkers):
      self.maxWorkers = maxWorkers
      
   # ..........................
   def _calculate(self):
      """
      @summary: Calculate the least cost path for all cells in the grid to a 
                   source cell
      @note: This method should be implemented in subclasses
      """
      # Find chunks with source cells
      startChunks = {}
      for x,y in self.sourceCells:
         # Get min x, miny, maxx, maxy
         xSteps = int(x / self.step)
         ySteps = int(y / self.step)
         minX = xSteps * self.step
         minY = ySteps * self.step
         maxX = min(((xSteps+1)*self.step - 1), self.cMtx.shape[1])
         maxY = min(((ySteps+1)*self.step - 1), self.cMtx.shape[0])
                    
         # Make a key
         k = "%s - %s - %s - %s" % (minX, minY, maxX, maxY)
         
         if not startChunks.has_key(k):
            startChunks[k] = {'bbox' : (minX, minY, maxX, maxY),
                              'sourceCells' : []}
         # Add to dictionary with source cell
         startChunks[k]['sourceCells'].append((x, y))
      
      print startChunks.keys()
      for k in startChunks.keys():
         minX, minY, maxX, maxY = startChunks[k]['bbox']
         self.chunks.append((minX, minY, maxX, maxY, startChunks[k]['sourceCells']))
      
      ts = []
      
      with concurrent.futures.ThreadPoolExecutor(max_workers=self.maxWorkers) as executor:
         ccc = True
         while ccc:
         #while len(self.chunks) > 0 and not all([t.done() for t in ts]):
            if len(self.chunks) > 0:
               chunk = self.chunks.pop(0)
               t =  executor.submit(self.dijkstraChunk, chunk)
               ts.append(t)
            ggg = all([t.done() for t in ts])
            ccc = len(self.chunks) != 0 or not ggg
      print "Done"      
      sleep(10)
      
   # ..........................
   def dijkstraChunk(self, chunk):#minx, miny, maxx, maxy, sourceCells):
      minx, miny, maxx, maxy, sourceCells = chunk
      
      name = '%s-%s-%s-%s' % (minx, miny, maxx, maxy)
      log = logging.getLogger(name)
      log.setLevel(logging.DEBUG)
      #logFormat = '%(asctime)-15s %(message)s'
      #logging.basicConfig(filename=os.path.join(LOG_PATH, '%s.log' % name, format=logFormat, level=logging.DEBUG)
      
      hndl = RotatingFileHandler(os.path.join(LOG_PATH, '%s.log' % name))
      #hndl.setFormatter(logFormat)
      hndl.setLevel(logging.DEBUG)
      log.addHandler(hndl)
      
      log.error("In Dijkstra chunk")
      log.debug("Minx: %s, miny: %s, maxx: %s, maxy: %s" % (minx, miny, maxx, maxy))
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
         log.debug("Adding cell: %s %s %s" % (x, y, cost))
         if int(self.cMtx[y][x]) == int(self.noDataValue) or self.cMtx[y][x] > cost:
            log.debug("Adding to heap")
            heapq.heappush(hq, (cost, x, y))
   
      # ........................
      def addNeighbors(x, y, cost):
         cellCost = self.inMtx[y][x]
         if cellCost != self.noDataValue:
            if x - 1 >= minx:
               addCell(x-1, y, self.costFn(cost, cellCost, self.inMtx[y][x-1], self.cellSize))
            if x + 1 <= maxx:
               addCell(x+1, y, self.costFn(cost, cellCost, self.inMtx[y][x+1], self.cellSize))
            if y-1 >= miny:
               addCell(x, y-1, self.costFn(cost, cellCost, self.inMtx[y-1][x], self.cellSize))
            if y+1 <= maxy:
               addCell(x, y+1, self.costFn(cost, cellCost, self.inMtx[y+1][x], self.cellSize))

      # Check to see if source cells inundate anything
      for x, y in sourceCells:
         cmpx = x
         cmpy = y
         if x < minx:
            cmpx = minx
         if x > maxx:
            cmpx = maxx
         if y < miny:
            cmpy = miny
         if y > maxy:
            cmpy = maxy
         
         c = max(self.cMtx[y][x], self.inMtx[cmpy][cmpx])

         if cmpx != x or cmpy != y:
            if int(self.cMtx[cmpy][cmpx]) == int(self.noDataValue) or self.cMtx[cmpy][cmpx] > c:
               self.cMtx[cmpy][cmpx] = c
               addNeighbors(cmpx, cmpy, c)
            else:
               log.debug("Not adding: %s, %s, %s, %s, %s, %s" % (self.cMtx[cmpy][cmpx], self.noDataValue, self.cMtx[cmpy][cmpx], c, x, y))
         else:
            addNeighbors(x, y, c)
            
      # .........................
      # Dijkstra
      
      while len(hq) > 0:
         cost, x, y = heapq.heappop(hq)
         log.debug("Popped %s, %s, %s" % (cost, x, y))
         #res.append("Popped %s, %s, %s" % (cost, x, y))
         if int(self.cMtx[y][x]) == int(self.noDataValue) or cost < self.cMtx[y][x]:
            self.cMtx[y][x] = cost
            log.debug("Setting cost in matrix for (%s, %s) = %s ... %s" % (x, y, cost, self.cMtx[y][x]))
            addNeighbors(x, y, cost)
            # Should we spread?
            #res.append("x, y, min x and y, max x and y: (%s, %s) (%s, %s), (%s, %s)" % (x, y, minx, miny, maxx, maxy))
            if x == minx:
               #res.append("Adding a left cell")
               leftCells.append((x, y))
            if x == maxx:
               #res.append("Adding a right cell")
               rightCells.append((x, y))
            if y == maxy:
               #res.append("Adding a bottom cell")
               bottomCells.append((x, y))
            if y == miny:
               #res.append("Adding a top cell")
               topCells.append((x, y))
         else:
            log.debug("Cost >= exist: (%s, %s) for (%s, %s)" % (cost, self.cMtx[y][x], x, y))
      
      # Spread
      # Assume that chunks start on left and top edges and partials may be at
      #    bottom and right
      
      # TODO: Handle moving left or up from right or bottom edges
      
      if len(leftCells) > 0:
         if minx > 0:
            self.chunks.append((minx-self.step, miny, minx-1, maxy, leftCells))
      if len(rightCells) > 0:
         if maxx < self.cMtx.shape[1]:
            newMaxX = min(self.cMtx.shape[1]-1, maxx + self.step)
            self.chunks.append((minx+self.step, miny, newMaxX, maxy, rightCells))
      if len(topCells) > 0:
         if miny > 0:
            self.chunks.append((minx, miny-self.step, maxx, miny-1, topCells))
      if len(bottomCells) > 0:
         if maxy < self.cMtx.shape[0]:
            newMaxY = min(self.cMtx.shape[0]-1, maxy + self.step)
            self.chunks.append((minx, miny+self.step, maxx, newMaxY, bottomCells))
      log.debug("Number of chunks: %s" % len(self.chunks))
      log.debug("Shape: %s, %s" % self.cMtx.shape)

            