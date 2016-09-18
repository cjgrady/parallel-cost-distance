"""
@summary: This script will split a large ASCII file into smaller chunks
@author: CJ Grady
"""

import argparse
import numpy as np
import os
import re

XCELLS = 1200
YCELLS = 1200
OUTPUT_DIR = "/home/cjgrady/thesis/flTest/"

def writeRaster(grid, ncols, nrows, xll, yll, cellsize, noData, latLL, longLL, outDir, ts):
   """
   @summary: Write a raster
   """
   fn = os.path.join(outDir, 'grid%s-%s-%s-%s.asc' % (longLL, latLL, 
                                                      longLL+ts, latLL+ts))
   print "Writing", fn
   headers = """\
ncols   {0}
nrows   {1}
xllcorner   {2}
yllcorner   {3}
cellsize   {4}
NODATA_value   {5}
""".format(ncols, nrows, xll, yll, cellsize, noData)
   np.savetxt(fn, grid, comments='', header=headers)
   

def readInputData(fn, xOffset=0, yOffset=0, lat=0, lon=0, xll=0, yll=0, 
                  xCells=XCELLS, yCells=YCELLS, outDir=OUTPUT_DIR, ts=1.0):
   """
   @summary: Read the input data
   """
   # Get headers
   with open(fn) as inF:
      for line in inF:
         if line.lower().startswith('ncols'):
            ncols = int(re.split(r' +', line.replace('\t', ' '))[1])
            print "Num cols:", ncols
         elif line.lower().startswith('nrows'):
            nrows = int(re.split(r' +', line.replace('\t', ' '))[1])
            print "Num rows:", nrows
         elif line.lower().startswith('xllcorner'):
            #xll = float(re.split(r' +', line.replace('\t', ' '))[1])
            pass
         elif line.lower().startswith('yllcorner'):
            #yll = float(re.split(r' +', line.replace('\t', ' '))[1])
            pass
         elif line.lower().startswith('cellsize'):
            cellsize = float(re.split(r' +', line.replace('\t', ' '))[1])
            print "Cell size:", cellsize
         #elif line.lower().startswith('dx'):
         #   cellsizeLine = line
         #elif line.lower().startswith('dy'):
         #   pass
         elif line.lower().startswith('nodata_value'):
            noData = float(re.split(r' +', line.replace('\t', ' '))[1])
            print "No data:", noData
         elif line.lower().startswith('xllce') or line.lower().startswith('yllce'):
            pass
         else:
            print line[:40]
            break
   
   grid = np.loadtxt(fn, skiprows=6)
   
   numXsteps = len(range(xOffset, ncols, xCells))
   numYsteps = len(range(yOffset, nrows, yCells))
   print range(xOffset, ncols - xCells, xCells)
   print range(yOffset, nrows - yCells, yCells)
   # Loop through and write files
   
   # Start at bottom left
   
   #for i in xrange(xOffset, ncols-XCELLS, XCELLS):
   for i in xrange(numXsteps):
      for j in xrange(numYsteps):
      #for j in xrange(yOffset, nrows-YCELLS, YCELLS):
         print "Writing grid", i, j
         fromY = nrows - ((j+1) * yCells)
         toY = nrows - (j * yCells)
         fromX = i * xCells
         toX = (i+1) * xCells
         print "[", fromY, ":", toY, ", ", fromX, ":", toX, "]"
         #g = grid[j:j+YCELLS,i:i+XCELLS]
         g = grid[fromY:toY, fromX:toX]
         myLong = lon + (i * ts)
         myLat = lat + (j * ts)
         print "Lat:", lat, "j:", j, "my lat:", myLat
         print "Long:", lon, "i:", i, "my long:", myLong
         print myLong, ",", myLat
         myXll = xll + i * xCells * cellsize
         myYll = yll + j * yCells * cellsize
         #myXll = xll + (numXs - i) * XCELLS * cellsize
         #myYll = yll + (numYs - j) * YCELLS * cellsize
         #myYll = yll + j * YCELLS * cellsize
         writeRaster(g, xCells, yCells, myXll, myYll, cellsize, noData, myLat, myLong, outDir, ts)
   
# .............................................................................
if __name__ == "__main__":
   # Split into 1 degree by 1 degree chunks
   # First, verify that tiles line up
#    parser = argparse.ArgumentParser()
#    parser.add_argument('-x', '--xOffset', type=int, 
#                      help="Offset the grid this many cells in the x direction")
#    parser.add_argument('-y', '--yOffset', type=int, 
#                      help="Offset the grid this many cells in the y direction")
#    parser.add_argument('-lat', '--startingLatitude', type=int,
#                   help="Integer latitude of the top left corner of the raster")
#    parser.add_argument('-long', '--startingLongitude', type=int,
#                   help="Integer longitude of the top left corner of the raster")
   
   # Volume 1
   #fn = '/home/cjgrady/thesis/ne_atl_crm_v1.asc'
   #readInputData(fn, xOffset=1, yOffset=1, lat=39, lon=-78, xll=-78, yll=39)

   # Volume 2
   fn = '/home/cjgrady/thesis/se_atl_crm_v1.asc'
   #readInputData(fn, xOffset=1, yOffset=1, lat=30, lon=-83, xll=-83, yll=30,
   #              xCells=int(.5*XCELLS), yCells=int(.5*YCELLS), outDir='/home/cjgrady/thesis/.5degree/se/', ts=0.5)
   readInputData(fn, xOffset=1, yOffset=1, lat=30, lon=-83, xll=-83, yll=30,
                 xCells=XCELLS, yCells=YCELLS, outDir='/home/cjgrady/thesis/1degree/se/', ts=1.0)
   readInputData(fn, xOffset=1, yOffset=1, lat=30, lon=-83, xll=-83, yll=30,
                 xCells=2*XCELLS, yCells=2*YCELLS, outDir='/home/cjgrady/thesis/2degree/se/', ts=2.0)
   
   # Volume 3
   #fn = '/home/cjgrady/thesis/fl_east_gom_crm_v1.asc'
   #readInputData(fn, xOffset=1, yOffset=1, lat=24, lon=-88, xll=-88, yll=24,
   #              xCells=int(.5*XCELLS), yCells=int(.5*YCELLS), outDir='/home/cjgrady/thesis/.5degree/fl/', ts=0.5)
   
   # Volume 4
   #fn = '/home/cjgrady/thesis/central_gom_crm_v1.asc'
   #readInputData(fn, xOffset=1, yOffset=1, lat=24, lon=-95, xll=-95, yll=24)
   
   # Volume 5
   #fn = '/home/cjgrady/thesis/western_gom_crm_v1.asc'
   #readInputData(fn, xOffset=1, yOffset=1, lat=25, lon=-100, xll=-100, yll=25)
   
   # Volume 6 - Already 1 degree tiles
   
   # Volume 7
   #fn = '/home/cjgrady/thesis/central_pacific_crm_v1.asc'
   #readInputData(fn, xOffset=1, yOffset=1, lat=36, lon=-127, xll=-127, yll=36)
   
   # Volume 8
   #fn = '/home/cjgrady/thesis/nw_pacific_crm_v1.asc'
   #readInputData(fn, xOffset=1, yOffset=1, lat=43, lon=-127, xll=-127, yll=43)
   
   # Volume 9
   fn = '/home/cjgrady/thesis/puerto_rico_crm_v1.asc'
   readInputData(fn, xOffset=1, yOffset=1, lat=17, lon=-68, xll=-68, yll=17,
                 xCells=int(.5*XCELLS), yCells=int(.5*YCELLS), outDir='/home/cjgrady/thesis/.5degree/pr/', ts=0.5)
   readInputData(fn, xOffset=1, yOffset=1, lat=17, lon=-68, xll=-68, yll=17,
                 xCells=XCELLS, yCells=YCELLS, outDir='/home/cjgrady/thesis/1degree/pr/', ts=1.0)
   readInputData(fn, xOffset=1, yOffset=1, lat=17, lon=-68, xll=-68, yll=17,
                 xCells=2*XCELLS, yCells=2*YCELLS, outDir='/home/cjgrady/thesis/2degree/pr/', ts=2.0)
   
   # Volume 10
   #fn = '/home/cjgrady/thesis/hawaii_crm_v1.asc'
   #readInputData(fn, xOffset=1, yOffset=1, lat=18, lon=-161, xll=-161, yll=18)
   
   