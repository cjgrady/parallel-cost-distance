"""
@summary: This script will split a large ASCII file into smaller chunks
@author: CJ Grady
@version: 1.0
@status: release
@license: gpl2
"""
import argparse
import numpy as np
import os
import re

# .............................................................................
def _writeRaster(grid, xll, yll, cellsize, noData, outDir, ts, maxX, maxY):
   """
   @summary: Write a raster tile
   @param grid: The data grid
   @param xll: The x value at the lower left corner of the grid
   @param yll: The y value at the lower left corner of the grid
   @param cellsize: The size of each cell
   @param noData: The no data value for the grid
   @todo: Consider determining ts from cell size and shape
   @todo: Take dx and dy as option instead of only cellsize
   @todo: Consider moving rounding to write raster only
   """
   nrows, ncols = grid.shape
   fn = os.path.join(outDir, 'grid%s-%s-%s-%s.asc' % (xll, yll, 
                                       min(maxX, xll+ts), min(maxY, yll+ts)))
   headers = """\
ncols   {0}
nrows   {1}
xllcorner   {2}
yllcorner   {3}
cellsize   {4}
NODATA_value   {5}
""".format(ncols, nrows, xll, yll, cellsize, noData)
   np.savetxt(fn, grid, comments='', header=headers, fmt="%i")

# .............................................................................
def splitTile(fn, ts, outDir, xOffset=0, yOffset=0, debug=False):
   """
   @summary: This function will split a raster into tiles of size ts.  Raster
                must be in EPSG: 4326 projection
   @param fn: The path to the ASCII grid to split
   @param ts: The desired size of the resulting tiles (in degrees)
   @param outDir: The directory to write the tiles to
   @param xOffset: (optional) Offset the tiles this many cells in the X direction
   @param yOffset: (optional) Offset the tiles this many cells in the Y direction
   """
   xc = yc = None
   # Get headers
   numHeaders = 0
   with open(fn) as inF:
      for line in inF:
         if line.lower().startswith('ncols'):
            ncols = int(re.split(r' +', line.replace('\t', ' '))[1])
            numHeaders += 1
         elif line.lower().startswith('nrows'):
            nrows = int(re.split(r' +', line.replace('\t', ' '))[1])
            numHeaders += 1
         elif line.lower().startswith('xllcorner'):
            #TODO: Round?
            xll = float(re.split(r' +', line.replace('\t', ' '))[1])
            numHeaders += 1
         elif line.lower().startswith('yllcorner'):
            #TODO: Round?
            yll = float(re.split(r' +', line.replace('\t', ' '))[1])
            numHeaders += 1
         elif line.lower().startswith('xllcenter'):
            #TODO: Round?
            xc = float(re.split(r' +', line.replace('\t', ' '))[1])
            numHeaders += 1
         elif line.lower().startswith('yllcenter'):
            #TODO: Round?
            yc = float(re.split(r' +', line.replace('\t', ' '))[1])
            numHeaders += 1
         elif line.lower().startswith('cellsize'):
            cellsize = float(re.split(r' +', line.replace('\t', ' '))[1])
            dx = dy = cellsize
            numHeaders += 1
         elif line.lower().startswith('dx'): # pragma: no cover
            dx = float(re.split(r' +', line.replace('\t', ' '))[1])
            numHeaders += 1
         elif line.lower().startswith('dy'): # pragma: no cover
            dy = float(re.split(r' +', line.replace('\t', ' '))[1])
            numHeaders += 1
         elif line.lower().startswith('nodata_value'):
            noData = float(re.split(r' +', line.replace('\t', ' '))[1])
            numHeaders += 1
         elif line.lower().startswith('xllce') or line.lower().startswith('yllce'): # pragma: no cover
            #TODO: This will probably fail, need to be able to get lower left corner
            numHeaders += 1
         else: 
            #print line[:40]
            break
   
   # Convert xc and yc to xll and yll if present
   if xc is not None:
      xll = round(xc - dx*(ncols / 2), 2)
   if yc is not None:
      yll = round(yc - dy*(nrows / 2), 2)
   
   # Get x and y cells per tile
   xCells = int(ts / dx)
   yCells = int(ts / dy)
   
   maxX = round(xll + ncols*dx, 2)
   maxY = round(yll + nrows*dy, 2)
   
   if debug: # pragma: no cover
      print "Num cols:", ncols
      print "Num rows:", nrows
      print "X lower left corner:", xll
      print "Y lower left corner:", yll
      print "X per tile:", xCells
      print "Y per tile:", yCells
      print "No data value:", noData

   grid = np.loadtxt(fn, skiprows=numHeaders)
   
   
   # Get the number of steps in each direction
   def getNumSteps(num, step, offset):
      mod = 1 if (num-offset) % step else 0
      return mod + (num-offset) / step

   numXsteps = getNumSteps(ncols, xCells, xOffset)
   numYsteps = getNumSteps(nrows, yCells, yOffset)

   if debug: # pragma: no cover
      print "Number of x steps:", numXsteps, "ncols:", ncols, "xCells:", xCells, "xOffset:", xOffset
      print "Number of y steps:", numYsteps, "nrows:", nrows, "yCells:", yCells, "yOffset:", yOffset
   
   # Split into tiles
   for i in xrange(numXsteps):
      for j in xrange(numYsteps):
         if debug: # pragma: no cover
            print "Writing grid", i, j
         fromY = max(0, nrows - ((j+1) * yCells))
         toY = nrows - (j * yCells)
         fromX = i * xCells
         toX = min(ncols, (i+1) * xCells)
         if debug:
            print "[", fromY, ":", toY, ", ", fromX, ":", toX, "]"
         g = grid[fromY:toY, fromX:toX]
         myXll = round(xll + i * xCells * cellsize, 2)
         myYll = round(yll + j * yCells * cellsize, 2)
         _writeRaster(g, myXll, myYll, cellsize, noData, outDir, ts, maxX, maxY)
   
   
   
# .............................................................................
if __name__ == "__main__": # pragma: no cover
   
   parser = argparse.ArgumentParser()
   parser.add_argument('fn', type=str, help="Path to the raster file to split")
   parser.add_argument('tileSize', type=float, 
                       help="What size tiles to split the raster into")
   parser.add_argument('outDir', type=str, 
                       help="Directory where the tiles should be written")
   
   parser.add_argument('-x', '--xOffset', type=int, 
                     help="Offset the grid this many cells in the x direction")
   parser.add_argument('-y', '--yOffset', type=int, 
                     help="Offset the grid this many cells in the y direction")
   parser.add_argument('-d', '--debug', type=int, choices=[0,1], 
                       help="Enable debugging (1 for yes, 0 for no)")
   
   args = parser.parse_args()
   
   xOffset = 0
   yOffset = 0
   debug = False
   
   if args.xOffset is not None:
      xOffset = args.xOffset
      
   if args.yOffset is not None:
      yOffset = args.yOffset
   
   if args.debug is not None:
      debug = bool(args.debug)
      
   splitTile(args.fn, args.tileSize, args.outDir, xOffset=xOffset, 
             yOffset=yOffset, debug=debug)
   
