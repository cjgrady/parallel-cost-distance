"""
@summary: This module contains the base class for least cost path calculations
             for a single tile
@author: CJ Grady
@status: alpha
"""
import numpy
import os

# .............................................................................
class SingleTileLCP(object):
   """
   @summary: This is the base class for single tile least cost path 
                calculations.  This class defines the interface for the 
                subclasses to use.  
   @note: Instances of this class should use the calculate method to calculate 
             the least cost path for every cell in the tile to reach one of the 
             source cells.  
   @note: Source cells should be set before calculate is used.  Do this by 
             either using the findSourceCells or addSourceVector method.
   """
   # ..........................
   def __init__(self, inputFilename, costFilename, costFn):
      """
      @summary: Constructor
      @param inputFilename: The file path to the input cost surface grid
      @param costFilename: The file path to the calculated cost surface grid
      @param costFn: The cost function to be used to calculate the cost to 
                        reach the destination cell.  The function signature
                        should be (source value, destination value, distance)
      """
      self.inFn = inputFilename
      self.cFn = costFilename
      self.costFn = costFn
      self.sourceCells = []
      
      self.initialize()
   
   # ..........................
   def addSourceVector(self, vect, originSide):
      """
      @summary: Add source cells from a vector of cells along one side of the
                   grid
      @param vect: The vector of cells
      @param originSide: The side that this vector is adjacent to (0: left, 1: 
                            top, 2: right, 3: bottom)
      """
      if originSide == 0:
         srcIdxs = numpy.where(vect < self.cMtx[0])[0]
         self.cMtx[0] = numpy.minimum(vect, self.cMtx[0])
         self.sourceCells = numpy.array([(0, x) for x in srcIdxs])
      elif originSide == 1:
         srcIdxs = numpy.where(vect < self.cMtx[:,0])[0]
         self.cMtx[:,0] = numpy.minimum(vect, self.cMtx[:,0])
         self.sourceCells = numpy.array([(x, 0) for x in srcIdxs])
      elif originSide == 2:
         x = len(self.cMtx)
         srcIdxs = numpy.where(vect < self.cMtx[x])[0]
         self.cMtx[x] = numpy.minimum(vect, self.cMtx[x])
         self.sourceCells = numpy.array([(x, y) for y in srcIdxs])
      else:
         y = len(self.cMtx[0])
         srcIdxs = numpy.where(vect < self.cMtx[:,y])[0]
         self.cMtx[:,y] = numpy.minimum(vect, self.cMtx[:,y])
         self.sourceCells = numpy.array([(x, y) for x in srcIdxs])

   # ..........................
   def calculate(self):
      """
      @summary: This function calculates the least cost path from every cell in
                   the tile to a source cell
      """
      self._calculate()
      self.writeOutputs()

   # ..........................
   def findSourceCells(self, sourceValue=None):
      """
      @summary: Find source cells in the input grid
      @param sourceValue: Source cells are grid cells with this value
      """
      #self.sourceCells = numpy.vstack(numpy.where(self.inMtx == self.noDataValue)).T
      self.sourceCells = numpy.vstack(numpy.where(self.inMtx <= -100)[::-1]).T
      for x, y in self.sourceCells:
         print x, y, self.inMtx[y][x]
         self.cMtx[y][x] = self.inMtx[y,x]
   
   # ..........................
   def initialize(self):
      """
      @summary: Initialize the object (read / prep files)
      @todo: Handle headers with other than 6 rows (dx / dy resolution)
      @todo: Pull out header information that may be needed
      """
      if os.path.exists(self.inFn):
         self.inMtx = numpy.loadtxt(self.inFn, skiprows=6)
         # Get headers
         with open(self.inFn) as f:
            headers = [next(f) for x in range(6)]
         self.headers = ''.join(headers).strip()
         self.noDataValue = int(self.headers.split('NODATA_VALUE')[1].strip())
         #TODO: Read from input
         self.cellSize = 10
      else:
         raise Exception, "Input grid does not exist"

      if os.path.exists(self.cFn):
         self.cMtx = numpy.loadtxt(self.cFn, skiprows=6)
      else:
         self.cMtx = numpy.ones(shape=self.inMtx.shape) * self.noDataValue

   # ..........................
   def writeOutputs(self):
      """
      @summary: Write output files (cost surface)
      """
      numpy.savetxt(self.cFn, self.cMtx, header=self.headers, fmt="%i", comments='')#fmt="%1.2f")

   # ..........................
   def _calculate(self):
      """
      @summary: Calculate the least cost path for all cells in the grid to a 
                   source cell
      @note: This method should be implemented in subclasses
      """
      pass
   