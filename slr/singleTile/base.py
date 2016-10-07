"""
@summary: This module contains the base class for least cost path calculations
             for a single tile
@author: CJ Grady
@status: alpha
"""
import numpy
import os
import re

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
      #TODO: Handle when no no data value is provided / better default
      self.noDataValue = -9999
      
      self._initialize()
   
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
         inVect = self.inMtx[0,:]
         costVect = self.cMtx[0,:]
         cmpVect = self._squishStretchVector(vect, len(inVect))
         getSourceCoords = lambda j: (0, j)
      elif originSide == 1:
         inVect = self.inMtx[:,0]
         costVect = self.cMtx[:,0]
         cmpVect = self._squishStretchVector(vect, len(inVect))
         getSourceCoords = lambda j: (j, 0)
      elif originSide == 2:
         inVect = self.inMtx[self.cMtx.shape[0]-1,:]
         costVect = self.cMtx[self.cMtx.shape[0]-1,:]
         cmpVect = self._squishStretchVector(vect, len(inVect))
         getSourceCoords = lambda j: (self.cMtx.shape[0]-1, j)
      elif originSide == 3:
         inVect = self.inMtx[:,self.cMtx.shape[1] -1]
         costVect = self.cMtx[:,self.cMtx.shape[1] -1]
         cmpVect = self._squishStretchVector(vect, len(inVect))
         getSourceCoords = lambda j: (j, self.cMtx.shape[1] - 1)
      else:
         raise Exception, "Cannot add source vector for side: %s" % originSide 

      # For each element in the array
      for i in xrange(len(cmpVect)):
         # Get cost to inundate
         # TODO: This should use cost function
         c = max(inVect[i], cmpVect[i])
         if int(costVect[i]) == int(self.noDataValue) or costVect[i] > c:
            costVect[i] = c
            self.sourceCells.append(getSourceCoords(i))


   # ..........................
   def calculate(self):
      """
      @summary: This function calculates the least cost path from every cell in
                   the tile to a source cell
      """
      self._calculate()
      self._writeOutputs()

   # ..........................
   def findSourceCells(self, sourceValue=None):
      """
      @summary: Find source cells in the input grid
      @param sourceValue: Source cells are grid cells with this value
      """
      #TODO: Make this smarter
      self.sourceCells = numpy.vstack(numpy.where(self.inMtx <= 0)[::-1]).T
      for x, y in self.sourceCells:
         self.cMtx[y][x] = 0#self.inMtx[y,x]
   
   # ..........................
   def _initialize(self):
      """
      @summary: Initialize the object (read / prep files)
      """
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
         
         self.inMtx = numpy.loadtxt(self.inFn, skiprows=numHeaders)
         
      else:
         raise IOError, "Input grid does not exist: %s" % self.inFn

      if os.path.exists(self.cFn):
         self.cMtx = numpy.loadtxt(self.cFn, skiprows=6)
      else:
         self.cMtx = numpy.ones(shape=self.inMtx.shape) * self.noDataValue

   # ..........................
   def _writeOutputs(self):
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
      raise NotImplementedError, "_calculate method is not implemented"
   
   # ..........................
   def _squishStretchVector(self, vect, toSize):
      """
      @summary: Shrink or stretch a vector so that it matches the one it will 
                   be compared to
      """
      vLen = len(vect)
      newV = []
      
      for i in xrange(toSize):
         idx = int(1.0 * vLen * i / toSize)
         newV.append(vect[idx])
      return numpy.array(newV)
   
