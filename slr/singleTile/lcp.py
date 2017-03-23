"""
@summary: This module contains a class to perform single tile least cost path
             calculations for a larger, multi-tile, study area
@author: CJ Grady
@version: 2.0
@status: alpha
@license: gpl2
"""
import argparse
import heapq
import numpy as np
import os
import re

from slr.common.costFunctions import seaLevelRiseCostFn

# .............................................................................
class SingleTileLCP(object):
   """
   @summary: This class performs least cost path computations on a single tile
                but is meant to be part of a multi-tile experiment
   """
   # ..........................
   def __init__(self, inputFilename, costFilename, costFn, padding=1):
      
      self.padding = padding
      self.inFn = inputFilename
      self.cFn = costFilename
      self.costFn = costFn
      self.sourceCells = []
      #TODO: Handle when no no data value is provided / better default
      self.noDataValue = -9999
      self.cellsChanged = 0
      
      self._initialize()
   
   # ..........................
   def calculate(self):
      if not self.sourceCells: # No source cells
         self._findSourceCells()
      
      # Get comparison vectors
      self.leftSource = np.copy(self.cMtx[self.padding:-self.padding,self.cmpL])
      self.rightSource = np.copy(self.cMtx[self.padding:-self.padding,self.cmpR])
      self.topSource = np.copy(self.cMtx[self.cmpT,self.padding:-self.padding])
      self.bottomSource = np.copy(self.cMtx[self.cmpB,self.padding:-self.padding])
      
      self._calculate()
      self._writeOutputs()
   
   # ..........................
   def writeChangedDirections(self, workDir, taskId):
      """
      @summary: If an edge has changed, write the matrix needed for adjacent
                   calculations
      """
      inputBaseName = "input_{}_{}".format(self.minLong, self.minLat)
      topInFn = topOutFn = bottomInFn = bottomOutFn = leftInFn = leftOutFn = rightInFn = rightOutFn = None
      
      # Top
      if len(np.where(self.cMtx[
            self.cmpT,self.padding:-self.padding] != self.topSource)[0]) > 0:
         topInFn = os.path.join(workDir, "{}-top.npy".format(inputBaseName))
         topOutFn = os.path.join(workDir, "{}-toTop.npy".format(taskId))
         
         # If the top input matrix has not been written, write it
         if not os.path.exists(topInFn):
            np.save(topInFn, self.inMtx[self.padding:2 * self.padding,
                                        self.padding:-self.padding])
         
         # Save the output padding
         np.save(topOutFn, self.cMtx[self.padding:2*self.padding,
                                     self.padding:-self.padding])
         
      # Bottom
      if len(np.where(self.cMtx[
          self.cmpB,self.padding:-self.padding] != self.bottomSource)[0]) > 0:
         bottomInFn = os.path.join(workDir, "{}-bottom.npy".format(inputBaseName))
         bottomOutFn = os.path.join(workDir, "{}-toBottom.npy".format(taskId))
         
         # If the bottom input matrix has not been written, write it
         if not os.path.exists(bottomInFn):
            np.save(bottomInFn, self.inMtx[-2*self.padding:-self.padding,
                                           self.padding:-self.padding])
         # Save the output padding matrix
         np.save(bottomOutFn, self.cMtx[-2*self.padding:-self.padding,
                                        self.padding:-self.padding])
      
      # Left
      if len(np.where(self.cMtx[
            self.padding:-self.padding,self.cmpL] != self.leftSource)[0]) > 0:
         leftInFn = os.path.join(workDir, "{}-left.npy".format(inputBaseName))
         leftOutFn = os.path.join(workDir, "{}-toLeft.npy".format(taskId))
         
         # If the left input matrix has not been written, write it
         if not os.path.exists(leftInFn):
            np.save(leftInFn, self.inMtx[self.padding:-self.padding,
                                         self.padding:2*self.padding])
         # Save the output padding matrix
         np.save(leftOutFn, self.cMtx[self.padding:-self.padding,
                                      self.padding:2*self.padding])

      # Right
      if len(np.where(self.cMtx[
            self.padding:-self.padding,self.cmpR] != self.rightSource)[0]) > 0:
         rightInFn = os.path.join(workDir, "{}-right.npy".format(inputBaseName))
         rightOutFn = os.path.join(workDir, "{}-toRight.npy".format(taskId))
         
         # If the right input matrix has not been written, write it
         if not os.path.exists(rightInFn):
            np.save(rightInFn, self.inMtx[self.padding:-self.padding,
                                          -2*self.padding:-self.padding])
         # Save the output padding matrix
         np.save(rightOutFn, self.cMtx[self.padding:-self.padding,
                                       -2*self.padding:-self.padding])

      self.outputFiles = [leftInFn, leftOutFn, topInFn, topOutFn,
                          rightInFn, rightOutFn, bottomInFn, bottomOutFn]

   # ..........................
   def _calculate(self):
      """
      @summary: Calculate the least cost path for all cells in the grid to a 
                   source cell
      @note: This method should be implemented in subclasses
      """
      hq = []
      
      # ........................
      def addCell(x, y, cost):
         """
         @summary: Add a cell to the heap if appropriate
         @param x: The x coordinate of the cell
         @param y: The y coordinate of the cell
         @param cost: The new cost to test for the cell
         """
         if int(self.cMtx[y,x]) == int(self.noDataValue) or self.cMtx[y,x] > cost:
            heapq.heappush(hq, (cost, x, y))
   
      # ........................
      def addNeighbors(x, y, cost):
         """
         @summary: Add neighbors of the cell to the heap
         @param x: The x value of the cell
         @param y: The y value of the cell
         @param cost: The cost of the cell
         """
         cellCost = self.inMtx[y,x]
         if cellCost != self.noDataValue:
            if x - 1 >= 0:
               addCell(x-1, y, self.costFn(cost, cellCost, self.inMtx[y][x-1], self.cellSize))
            if x + 1 < len(self.inMtx[y]):
               addCell(x+1, y, self.costFn(cost, cellCost, self.inMtx[y][x+1], self.cellSize))
            if y-1 >= 0:
               addCell(x, y-1, self.costFn(cost, cellCost, self.inMtx[y-1][x], self.cellSize))
            if y+1 < len(self.inMtx):
               addCell(x, y+1, self.costFn(cost, cellCost, self.inMtx[y+1][x], self.cellSize))

      # .........................
      # Dijkstra
      
      for x, y in self.sourceCells:
         if int(self.cMtx[y,x]) == int(self.noDataValue):
            c = self.inMtx[y,x]
         else:
            c = self.cMtx[y,x]
         addNeighbors(x, y, c)
            
      while len(hq) > 0:
         cost, x, y = heapq.heappop(hq)
         
         if self.cMtx[y,x] == self.noDataValue or cost < self.cMtx[y,x]:
            self.cMtx[y,x] = cost
            self.cellsChanged += 1
            addNeighbors(x, y, cost)

   # ..........................
   def _initialize(self):
      if os.path.exists(self.inFn):
         numHeaders = 0
         hs = []
         with open(self.inFn) as f:
            for line in f:
               if line.lower().startswith('ncols'):
                  ncols = int(re.split(r' +', line.replace('\t', ' '))[1])
                  hs.append(line)
                  numHeaders += 1
               elif line.lower().startswith('nrows'):
                  nrows = int(re.split(r' +', line.replace('\t', ' '))[1])
                  numHeaders += 1
                  hs.append(line)
               elif line.lower().startswith('xllcorner'):
                  #TODO: Round?
                  self.minLong = float(re.split(r' +', line.replace('\t', ' '))[1])
                  numHeaders += 1
                  hs.append(line)
               elif line.lower().startswith('yllcorner'):
                  #TODO: Round?
                  self.minLat = float(re.split(r' +', line.replace('\t', ' '))[1])
                  numHeaders += 1
                  hs.append(line)
               elif line.lower().startswith('cellsize'):
                  self.cellSize = float(re.split(r' +', line.replace('\t', ' '))[1])
                  dx = dy = self.cellSize
                  numHeaders += 1
                  hs.append(line)
               elif line.lower().startswith('dx'):
                  dx = float(re.split(r' +', line.replace('\t', ' '))[1])
                  numHeaders += 1
                  hs.append(line)
               elif line.lower().startswith('dy'):
                  dy = float(re.split(r' +', line.replace('\t', ' '))[1])
                  numHeaders += 1
                  hs.append(line)
               elif line.lower().startswith('nodata_value'):
                  self.noData = float(re.split(r' +', line.replace('\t', ' '))[1])
                  numHeaders += 1
                  hs.append(line)
               elif line.lower().startswith('xllce') or line.lower().startswith('yllce'):
                  #TODO: This will probably fail, need to be able to get lower left corner
                  numHeaders += 1
                  hs.append(line)
               else:
                  break
         
         self.headers = ''.join(hs)
         
         tmpIn = np.loadtxt(self.inFn, skiprows=numHeaders, dtype=int)
         
         tmpShape0, tmpShape1 = tmpIn.shape
         
         # Create a padded matrix
         self.inMtx = np.ones(
            (tmpShape0 + 2*self.padding, tmpShape1 + 2*self.padding), 
            dtype=int) * self.noDataValue
         
         print self.inMtx.shape, tmpIn.shape
         # Set the meat of the matrix to the input matrix
         self.inMtx[self.padding:-self.padding,
                    self.padding:-self.padding] = tmpIn
         #self.inMtx = np.loadtxt(self.inFn, skiprows=numHeaders, dtype=int)
      else:
         raise IOError, "Input grid does not exist: %s" % self.inFn

      # Initialize padded cost matrix
      # Create a cost matrix that is the size of the padded input matrix
      self.cMtx = np.ones(shape=self.inMtx.shape, dtype=int) * self.noDataValue

      # If we have a cost matrix, load it into the middle
      if os.path.exists(self.cFn):
         tmpCmtx = np.loadtxt(self.cFn, skiprows=6, dtype=int)
         self.cMtx[self.padding:-self.padding,
                   self.padding:-self.padding] = tmpCmtx
                   
      self.cmpL = self.padding
      self.cmpR = self.cMtx.shape[1] - (self.padding + 1)
      self.cmpT = self.padding
      self.cmpB = self.cMtx.shape[0] - (self.padding + 1)
   
   # ..........................
   def _findSourceCells(self):
      """
      @summary: Find source cells in the input grid
      """
      #TODO: Make this smarter
      tmp = np.vstack(
         np.where(self.inMtx <= 0)[::-1]).T
      try:
         xMax = self.cMtx.shape[1] - self.padding
         yMax = self.cMtx.shape[0] - self.padding
         yMin = xMin = self.padding
         for x, y in tmp:
            if x >= xMin and y >= yMin and x < xMax and y < yMax:
               self.cMtx[y,x] = 0
               self.cellsChanged += 1
               self.sourceCells.append((x,y))
      except Exception, e:
         raise Exception, "{0}-{1}-{2}".format(str(self.sourceCells), 
                                               str(e), str(tmp.shape))

   # ..........................
   def _writeOutputs(self):
      """
      @summary: Write output files (cost surface)
      """
      tmpCout = self.cMtx[self.padding:-self.padding, self.padding:-self.padding]
      np.savetxt(self.cFn, tmpCout, header=self.headers, fmt="%i", comments='')#fmt="%1.2f")

   # ..........................
   def addLeftSourceMatrix(self, mtx, inMtx):
      """
      @summary: Adds a source matrix to the left side
      @param mtx: The cost source matrix
      @param inMtx: The input source matrix
      """
      # Check to see if matrix has the correct shape
      testShape = self.cMtx[self.padding:-self.padding,
                            0:self.padding].shape
      if mtx.shape != testShape:
         mtx = self._stretchSquishMatrix(mtx, testShape[1], testShape[0])
         inMtx = self._stretchSquishMatrix(inMtx, testShape[1], testShape[0])

      self.inMtx[self.padding:-self.padding, 0:self.padding] = inMtx
      self.cMtx[self.padding:-self.padding, 0:self.padding] = mtx
      
      # Add source cells
      for i in xrange(self.padding, self.cMtx.shape[0] - (self.padding + 1)):
         if self.cMtx[i, self.cmpL-1] > self.noDataValue:
            self.sourceCells.append((self.cmpL-1, i))
      
      # Move self.cmpL
      self.cmpL -= 1
   
   # ..........................
   def addRightSourceMatrix(self, mtx, inMtx):
      """
      @summary: Adds a source matrix to the right side
      @param mtx: The cost source matrix
      @param inMtx: The input source matrix
      """
      # Check to see if matrix has the correct shape
      testShape = self.cMtx[self.padding:-self.padding,
                            -self.padding:].shape
      if mtx.shape != testShape:
         mtx = self._stretchSquishMatrix(mtx, testShape[1], testShape[0])
         inMtx = self._stretchSquishMatrix(inMtx, testShape[1], testShape[0])

      self.inMtx[self.padding:-self.padding, -self.padding:] = inMtx
      self.cMtx[self.padding:-self.padding, -self.padding:] = mtx
      
      # Add source cells
      for i in xrange(self.padding, self.cMtx.shape[0] - (self.padding + 1)):
         if self.cMtx[i, self.cmpR + 1] > self.noDataValue:
            self.sourceCells.append((self.cmpR + 1, i))
      
      # Move self.cmpR
      self.cmpR += 1
   
   # ..........................
   def addTopSourceMatrix(self, mtx, inMtx):
      """
      @summary: Adds a source matrix to the top side
      @param mtx: The cost source matrix
      @param inMtx: The input source matrix
      """
      # Check to see if matrix has the correct shape
      testShape = self.cMtx[:self.padding,
         self.padding:-self.padding].shape
         
      if mtx.shape != testShape:
         mtx = self._stretchSquishMatrix(mtx, testShape[1], testShape[0])
         inMtx = self._stretchSquishMatrix(inMtx, testShape[1], testShape[0])

      self.inMtx[:self.padding, self.padding:-self.padding] = inMtx
      self.cMtx[:self.padding, self.padding:-self.padding] = mtx
      
      # Add source cells
      for i in xrange(self.padding, self.cMtx.shape[1] - (self.padding + 1)):
         if self.cMtx[self.cmpT-1, i] > self.noDataValue:
            self.sourceCells.append((i, self.cmpT - 1))
      
      # Move self.cmpT
      self.cmpT -= 1
   
   # ..........................
   def addBottomSourceMatrix(self, mtx, inMtx):
      """
      @summary: Adds a source matrix to the bottom side
      @param mtx: The cost source matrix
      @param inMtx: The input source matrix
      """
      # Check to see if matrix has the correct shape
      testShape = self.cMtx[-self.padding:,
         self.padding:-self.padding].shape
         
      if mtx.shape != testShape:
         mtx = self._stretchSquishMatrix(mtx, testShape[1], testShape[0])
         inMtx = self._stretchSquishMatrix(inMtx, testShape[1], testShape[0])

      self.inMtx[-self.padding:, self.padding:-self.padding] = inMtx
      self.cMtx[-self.padding:, self.padding:-self.padding] = mtx
      
      # Add source cells
      for i in xrange(self.padding, self.cMtx.shape[1] - (self.padding + 1)):
         if self.cMtx[self.cmpB + 1, i] > self.noDataValue:
            self.sourceCells.append((i, self.cmpB + 1))
      
      # Move self.cmpB
      self.cmpB += 1
   
   # ..........................
   def _stretchSquishMatrix(self, mtx, toX, toY):
      origY, origX = mtx.shape
      
      newMtx = np.ones((toY, toX), dtype=int)
      for y in xrange(toY):
         for x in xrange(toX):
            newMtx[y,x] = mtx[int(1.0 * origY * y / toY),
                              int(1.0 * origX * x / toX)]
      return newMtx
   
   # ..........................
   def writeSummary(self, sfn):
      with open(sfn, 'w') as outF:
         outF.write("{}\n".format(self.minLong))
         outF.write("{}\n".format(self.minLat))
         for i in self.outputFiles:
            outF.write("{}\n".format(i))
         outF.write("{}\n".format(self.cellsChanged))


"""
stats / etc
"""
# .............................................................................
if __name__ == "__main__": # pragma: no cover
   
   parser = argparse.ArgumentParser()
   
   parser.add_argument("dem")
   parser.add_argument("costSurface")
   #parser.add_argument('-g', help="Generate edge vectors for modified cells")
   parser.add_argument('-t', '--taskId', help="Use this task id for outputs")
   parser.add_argument('-w', '--workDir', help="Work directory")
   parser.add_argument('-p', '--padding', type=int, help="Number of padding cells")
   
   parser.add_argument('-st', help='Source matrix from the top')
   parser.add_argument('-sb', help='Source matrix from the bottom')
   parser.add_argument('-sl', help='Source matrix from the left')
   parser.add_argument('-sr', help='Source matrix from the right')

   parser.add_argument('-it', help='Input matrix from the top')
   parser.add_argument('-ib', help='Input matrix from the bottom')
   parser.add_argument('-il', help='Input matrix from the left')
   parser.add_argument('-ir', help='Input matrix from the right')
   
   parser.add_argument('-s', '--summaryFn', help="Where to write summary information")
   
   #parser.add_argument('-o', help="File to write outputs")
   #parser.add_argument('--ts', type=float)
   #parser.add_argument('-e', type=str, help="Log errors to this file location")
   #parser.add_argument('-b', type=str, help="File to store benchmarks")

   args = parser.parse_args()
   print args
   
   if args.padding is not None:
      padding = args.padding
   else:
      padding = 1
   
   
   tile = SingleTileLCP(args.dem, args.costSurface, seaLevelRiseCostFn, 
                        padding=padding)
   if args.st is not None and args.it is not None:
      tile.addTopSourceMatrix(np.load(args.st), np.load(args.it))
   if args.sb is not None and args.ib is not None:
      tile.addBottomSourceMatrix(np.load(args.sb), np.load(args.ib))
   if args.sl is not None and args.il is not None:
      tile.addLeftSourceMatrix(np.load(args.sl), np.load(args.il))
   if args.sr is not None and args.ir is not None:
      tile.addRightSourceMatrix(np.load(args.sr), np.load(args.ir))
 
   tile.calculate()
   tile.writeChangedDirections(args.workDir, args.taskId)
   
   if args.summaryFn is not None:
      tile.writeSummary(args.summaryFn)

   