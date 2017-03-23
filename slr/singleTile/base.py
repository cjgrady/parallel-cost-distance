"""
@summary: This module contains the base class for least cost path calculations
             for a single tile
@author: CJ Grady
@version: 1.0
@status: release
@license: gpl2
"""
import numpy
import os
import re
import time

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
   def __init__(self, inputFilename, costFilename, costFn, padding):
      """
      @summary: Constructor
      @param inputFilename: The file path to the input cost surface grid
      @param costFilename: The file path to the calculated cost surface grid
      @param costFn: The cost function to be used to calculate the cost to 
                        reach the destination cell.  The function signature
                        should be (source value, destination value, distance)
      """
      self.padding = padding
      self.inFn = inputFilename
      self.cFn = costFilename
      self.costFn = costFn
      self.leftSource = None
      self.rightSource = None
      self.topSource = None
      self.bottomSource = None
      self.sourceCells = []
      #TODO: Handle when no no data value is provided / better default
      self.noDataValue = -9999
      
      self._initialize()
      self.extras = []
   
   # ..........................
   def addLeftSourceMatrix(self, mtx):
      # Stretch / squish matrix
      # Set cost values
      pass
   
   # ..........................
   def addRightSourceMatrix(self, mtx):
      pass
   
   # ..........................
   def addTopSourceMatrix(self, mtx):
      pass
   
   # ..........................
   def addBottomSourceMatrix(self, mtx):
      pass
   
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
         inVect = self.inMtx[:,0]
         costVect = self.cMtx[:,0]
         cmpVect = self._squishStretchVector(vect, len(inVect))
         self.leftSource = numpy.copy(cmpVect)
         getSourceCoords = lambda j: (0, j)
         #self.extras.append("Origin side: 0")
         #self.extras.append("In vector [:,0] - %s" % str(inVect.tolist()))
         #self.extras.append("Cost vector [:,0] - %s" % str(costVect.tolist()))
         #self.extras.append("Cmp vector - %s" % str(cmpVect.tolist()))
      elif originSide == 1:
         inVect = self.inMtx[0,:]
         costVect = self.cMtx[0,:]
         cmpVect = self._squishStretchVector(vect, len(inVect))
         self.topSource = numpy.copy(cmpVect)
         getSourceCoords = lambda j: (j, 0)
         #self.extras.append("Origin side: 1")
         #self.extras.append("In vector [0,:] - %s" % str(inVect.tolist()))
         #self.extras.append("Cost vector [0,:] - %s" % str(costVect.tolist()))
         #self.extras.append("Cmp vector - %s" % str(cmpVect.tolist()))
      elif originSide == 2:
         inVect = self.inMtx[:,-1]
         costVect = self.cMtx[:,-1]
         cmpVect = self._squishStretchVector(vect, len(inVect))
         self.rightSource = numpy.copy(cmpVect)
         getSourceCoords = lambda j: (self.cMtx.shape[1]-1, j)
         #self.extras.append("Origin side: 2")
         #self.extras.append("In vector [:,-1] - %s" % str(inVect.tolist()))
         #self.extras.append("Cost vector [:,-1] - %s" % str(costVect.tolist()))
         #self.extras.append("Cmp vector - %s" % str(cmpVect.tolist()))
      elif originSide == 3:
         inVect = self.inMtx[-1,:]
         costVect = self.cMtx[-1,:]
         cmpVect = self._squishStretchVector(vect, len(inVect))
         self.bottomSource = numpy.copy(cmpVect)
         getSourceCoords = lambda j: (j, self.cMtx.shape[0] - 1)
         #self.extras.append("Origin side: 3")
         #self.extras.append("In vector [-1,:] - %s" % str(inVect.tolist()))
         #self.extras.append("Cost vector [-1,:] - %s" % str(costVect.tolist()))
         #self.extras.append("Cmp vector - %s" % str(cmpVect.tolist()))
      else:
         raise Exception, "Cannot add source vector for side: %s" % originSide 

      # For each element in the array
      for i in xrange(len(cmpVect)):
         # Get cost to inundate
         # TODO: This should use cost function
         # TODO: Evaluate this and if we are doing things correctly
         c = max(inVect[i], cmpVect[i])
         if int(costVect[i]) == int(self.noDataValue) or costVect[i] > c:
            # Experimental change
            x,y = getSourceCoords(i)
            self.cMtx[y,x] = c
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
      tmp = numpy.vstack(numpy.where(self.inMtx <= 0)[::-1]).T
      if tmp.shape[0] > 0 and tmp.shape[1] == 2:
         self.sourceCells = tmp
      try:
         print self.sourceCells
         for x, y in self.sourceCells:
            self.cMtx[y][x] = 0#self.inMtx[y,x]
      except Exception, e:
         raise Exception, "{0}-{1}-{2}".format(str(self.sourceCells), str(e), str(tmp.shape))
   
   # ..........................
   def _initialize(self):
      """
      @summary: Initialize the object (read / prep files)
      """
      if os.path.exists(self.inFn):
         numHeaders = 0
         hs = []
         t1 = time.time()
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
         
         tmpIn = numpy.loadtxt(self.inFn, skiprows=numHeaders, dtype=int)
         
         tmpShape0, tmpShape1 = tmpIn.shape
         
         # Create a padded matrix
         self.inMtx = numpy.ones(
            (tmpShape0 + 2*self.padding, tmpShape1 + 2*self.padding), 
            dtype=int) * self.noDataValue
         
         # Set the meat of the matrix to the input matrix
         self.inMtx[self.padding:-self.padding,
                    self.padding:-self.padding] = tmpIn
         #self.inMtx = numpy.loadtxt(self.inFn, skiprows=numHeaders, dtype=int)
         t2 = time.time()
         self.rc = t2 - t1
      else:
         raise IOError, "Input grid does not exist: %s" % self.inFn

      # Initialize padded cost matrix
      # Create a cost matrix that is the size of the padded input matrix
      self.cMtx = numpy.ones(shape=self.inMtx.shape, dtype=int) * self.noDataValue

      # If we have a cost matrix, load it into the middle
      if os.path.exists(self.cFn):
         tmpCmtx = numpy.loadtxt(self.cFn, skiprows=6, dtype=int)
         self.cMtx[self.padding:-self.padding,
                   self.padding:-self.padding] = tmpCmtx

   # ..........................
   def _writeOutputs(self):
      """
      @summary: Write output files (cost surface)
      """
      t1 = time.time()
      tmpCout = self.cMtx[self.padding:-self.padding, self.padding:-self.padding]
      numpy.savetxt(self.cFn, tmpCout, header=self.headers, fmt="%i", comments='')#fmt="%1.2f")
      t2 = time.time()
      self.wc = t2 - t1

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
   
