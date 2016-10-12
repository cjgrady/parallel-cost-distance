"""
@summary: This module tests the tile splitter tool
@author: CJ Grady
"""
from hashlib import md5
import numpy as np
import os
from random import randint
import re
from tempfile import NamedTemporaryFile, gettempdir

from slr.tools.tileSplitter import splitTile

# .............................................................................
TEST_SURFACE = [
 [11,  2, 14, 12,  5, 20,  6, 11, 11, 15, 11, 16, 10,  8, 16, 20,  6,  1, 17, 19], 
 [19, 11,  9,  1, 16, 16, 18,  3, 10,  9, 15,  0,  1, 18, 20,  7, 15,  5, 10,  9], 
 [10,  0,  7, 15,  7, 16, 12, 10, 10,  4, 15,  2, 14, 17, 14,  4,  1,  3, 18, 20], 
 [ 8,  0, 15, 13,  4,  3,  7,  0,  6,  6,  6,  4,  5, 20,  9, 12,  5,  6,  1, 19], 
 [18, 16, 14, 20,  2,  4,  5, 16, 10,  6,  7,  6,  2,  7,  4, 18,  0,  6,  0,  9], 
 [ 8,  2,  6, 16, 19,  5, 12,  3,  5,  0,  2, 10,  7, 20, 14, 16,  9,  9, 19, 15], 
 [ 4, 11, 16, 13,  8,  7, 13,  2, 13,  0,  4,  4,  0, 16, 11,  4, 20, 15, 16,  3], 
 [ 2,  2,  0, 10, 18,  7, 20,  5, 10, 15,  5,  5,  8, 19, 16,  9,  3, 15,  7, 19], 
 [19, 15,  8, 13,  6,  6, 12, 20,  4, 15, 12, 17, 16,  7,  1, 17, 16, 19,  9, 11], 
 [ 9, 19,  3,  5,  7, 18, 17,  3, 12, 13,  5, 13, 20, 12,  2, 20, 13,  1,  4,  5], 
 [ 7,  4, 11, 11,  2, 13, 20, 17,  2, 16, 10,  6,  3, 13, 10,  1, 13,  1, 10, 17], 
 [18,  8,  7,  1, 15,  3, 14, 17,  3,  7,  9,  3,  1,  1, 13,  7,  4,  5,  2,  7], 
 [16, 13,  1,  8, 13, 15,  3,  6,  2,  6, 13,  9, 15,  9,  4, 17, 12,  5,  5, 18], 
 [ 4,  2, 14,  4,  5,  0, 16,  9,  6, 12, 20, 15,  6, 13, 13,  8,  7, 15, 17, 13], 
 [ 5, 12,  1,  4, 18,  7,  9, 11, 19,  0,  0,  4,  5, 12, 11,  5, 18,  0, 16, 18], 
 [10, 15, 20,  5,  7, 14,  9,  0, 15,  0,  8, 10, 18, 15, 16, 18,  1,  1,  9, 12], 
 [ 8, 12, 14, 20, 18, 16,  9,  3,  2,  1, 11,  1, 13,  2, 11, 20,  7, 11,  5, 19], 
 [ 5, 12, 14,  2,  9,  7, 16, 17,  0, 14,  5, 10,  3,  1,  9, 15, 20, 17, 16, 16], 
 [ 0,  3,  3,  5, 11, 20, 16, 15, 15, 16,  9, 10,  4, 16, 12, 19,  3, 18, 20, 15], 
 [16,  5,  6,  5, 16, 18, 10, 13, 11, 11, 17, 13, 16, 10, 11, 16, 16,  9, 11,  3]
]
# .............................................................................
def _getTemporaryDirectory(r):
   """
   @summary: Creates a temporary directory for testing
   """
   n = md5(str(r)).hexdigest()
   d = os.path.join(gettempdir(), n)
   os.mkdir(d)
   return d

# .............................................................................
def _readHeaders(fn):
   """
   @summary: Helper function to read tile headers
   """
   with open(fn) as inF:
      for line in inF:
         if line.lower().startswith('ncols'):
            ncols = int(re.split(r' +', line.replace('\t', ' '))[1])
         elif line.lower().startswith('nrows'):
            nrows = int(re.split(r' +', line.replace('\t', ' '))[1])
         elif line.lower().startswith('xllcorner'):
            xll = float(re.split(r' +', line.replace('\t', ' '))[1])
         elif line.lower().startswith('yllcorner'):
            yll = float(re.split(r' +', line.replace('\t', ' '))[1])
         elif line.lower().startswith('cellsize'):
            cellsize = float(re.split(r' +', line.replace('\t', ' '))[1])
         elif line.lower().startswith('nodata_value'):
            noData = float(re.split(r' +', line.replace('\t', ' '))[1])
         else:
            break
   return ncols, nrows, xll, yll, cellsize, noData

# .............................................................................
def test_split_even_tiles():
   """
   @summary: This test checks that a surface can be split into even tiles
   """
   # Write input file - 10 degree by 10 degree
   tf = NamedTemporaryFile(delete=False)
   tfn = tf.name
   tf.write('ncols   {0}\n'.format(str(len(TEST_SURFACE[0]))))
   tf.write('nrows   {0}\n'.format(str(len(TEST_SURFACE))))
   tf.write("xllcorner   0\n")
   tf.write("yllcorner   0\n")
   tf.write("cellsize   0.5\n")
   tf.write("NODATA_value   -999\n")
   for line in TEST_SURFACE:
      tf.write("{0}\n".format(' '.join([str(i) for i in line])))
   tf.close()

   # Create directory
   outDir = _getTemporaryDirectory(randint(0, 1000))
   
   # Run tile splitter
   splitTile(tfn, 5.0, outDir)
   
   # Look at results
   # Should have (0,0,5,5),(5,0,10,5),(0,5,5,10),(5,5,10,10)
   t1Fn = os.path.join(outDir, "grid0.0-0.0-5.0-5.0.asc")
   t2Fn = os.path.join(outDir, "grid5.0-0.0-10.0-5.0.asc")
   t3Fn = os.path.join(outDir, "grid0.0-5.0-5.0-10.0.asc")
   t4Fn = os.path.join(outDir, "grid5.0-5.0-10.0-10.0.asc")
   
   assert os.path.exists(t1Fn)
   assert os.path.exists(t2Fn)
   assert os.path.exists(t3Fn)
   assert os.path.exists(t4Fn)
   
   # Check headers of each
   # Check content of each
   
   # Check (0,0,5,5)
   #  -- Check headers
   ncols, nrows, xll, yll, cs, nd = _readHeaders(t1Fn)
   assert ncols == 10
   assert nrows == 10
   assert xll == 0.0
   assert yll == 0.0
   assert cs == 0.5
   assert nd == -999
   
   # -- Check content
   mtx1 = np.loadtxt(t1Fn, dtype=int, comments='', skiprows=6)
   cmp1 = np.array([line[:10] for line in TEST_SURFACE[10:]], dtype=int)
   assert np.array_equal(mtx1, cmp1)
   
   # Check (5,0,10,5)
   #  -- Check headers
   ncols, nrows, xll, yll, cs, nd = _readHeaders(t2Fn)
   assert ncols == 10
   assert nrows == 10
   assert xll == 5.0
   assert yll == 0.0
   assert cs == 0.5
   assert nd == -999
   
   # -- Check content
   mtx2 = np.loadtxt(t2Fn, dtype=int, comments='', skiprows=6)
   cmp2 = np.array([line[10:] for line in TEST_SURFACE[10:]], dtype=int)
   assert np.array_equal(mtx2, cmp2)
   
   # Check (0,5,5,10)
   #  -- Check headers
   ncols, nrows, xll, yll, cs, nd = _readHeaders(t3Fn)
   assert ncols == 10
   assert nrows == 10
   assert xll == 0.0
   assert yll == 5.0
   assert cs == 0.5
   assert nd == -999
   
   # -- Check content
   mtx3 = np.loadtxt(t3Fn, dtype=int, comments='', skiprows=6)
   cmp3 = np.array([line[:10] for line in TEST_SURFACE[:10]], dtype=int)
   assert np.array_equal(mtx3, cmp3)
   
   # Check (5,5,10,10)
   #  -- Check headers
   ncols, nrows, xll, yll, cs, nd = _readHeaders(t4Fn)
   assert ncols == 10
   assert nrows == 10
   assert xll == 5.0
   assert yll == 5.0
   assert cs == 0.5
   assert nd == -999
   
   # -- Check content
   mtx4 = np.loadtxt(t4Fn, dtype=int, comments='', skiprows=6)
   cmp4 = np.array([line[10:] for line in TEST_SURFACE[:10]], dtype=int)
   assert np.array_equal(mtx4, cmp4)
   
# .............................................................................
def test_split_uneven_tiles():
   """
   @summary: This test checks that a surface can be split into uneven tiles
   """
   # Write input file - 10 degree by 10 degree
   tf = NamedTemporaryFile(delete=False)
   tfn = tf.name
   tf.write('ncols   {0}\n'.format(str(len(TEST_SURFACE[0]))))
   tf.write('nrows   {0}\n'.format(str(len(TEST_SURFACE))))
   tf.write("xllcorner   0\n")
   tf.write("yllcorner   0\n")
   tf.write("cellsize   0.5\n")
   tf.write("NODATA_value   -999\n")
   for line in TEST_SURFACE:
      tf.write("{0}\n".format(' '.join([str(i) for i in line])))
   tf.close()

   # Create directory
   outDir = _getTemporaryDirectory(randint(0, 1000))
   
   # Run tile splitter
   splitTile(tfn, 7.5, outDir)
   
   # Look at results
   # Should have (0,0,7.5,7.5),(7.5,0,10,7.5),(0,7.5,7.5,10),(7.5,7.5,10,10)
   t1Fn = os.path.join(outDir, "grid0.0-0.0-7.5-7.5.asc")
   t2Fn = os.path.join(outDir, "grid7.5-0.0-10.0-7.5.asc")
   t3Fn = os.path.join(outDir, "grid0.0-7.5-7.5-10.0.asc")
   t4Fn = os.path.join(outDir, "grid7.5-7.5-10.0-10.0.asc")
   
   assert os.path.exists(t1Fn)
   assert os.path.exists(t2Fn)
   assert os.path.exists(t3Fn)
   assert os.path.exists(t4Fn)
   
   # Check headers of each
   # Check content of each
   
   # Check (0,0,7.5,7.5)
   #  -- Check headers
   ncols, nrows, xll, yll, cs, nd = _readHeaders(t1Fn)
   assert ncols == 15
   assert nrows == 15
   assert xll == 0.0
   assert yll == 0.0
   assert cs == 0.5
   assert nd == -999
   
   # -- Check content
   mtx1 = np.loadtxt(t1Fn, dtype=int, comments='', skiprows=6)
   cmp1 = np.array([line[:15] for line in TEST_SURFACE[5:]], dtype=int)
   assert np.array_equal(mtx1, cmp1)
   
   # Check (7.5,0,10,7.5)
   #  -- Check headers
   ncols, nrows, xll, yll, cs, nd = _readHeaders(t2Fn)
   assert ncols == 5
   assert nrows == 15
   assert xll == 7.5
   assert yll == 0.0
   assert cs == 0.5
   assert nd == -999
   
   # -- Check content
   mtx2 = np.loadtxt(t2Fn, dtype=int, comments='', skiprows=6)
   cmp2 = np.array([line[15:] for line in TEST_SURFACE[5:]], dtype=int)
   assert np.array_equal(mtx2, cmp2)
   
   # Check (0,7.5,7.5,10)
   #  -- Check headers
   ncols, nrows, xll, yll, cs, nd = _readHeaders(t3Fn)
   assert ncols == 15
   assert nrows == 5
   assert xll == 0.0
   assert yll == 7.5
   assert cs == 0.5
   assert nd == -999
   
   # -- Check content
   mtx3 = np.loadtxt(t3Fn, dtype=int, comments='', skiprows=6)
   cmp3 = np.array([line[:15] for line in TEST_SURFACE[:5]], dtype=int)
   assert np.array_equal(mtx3, cmp3)
   
   # Check (7.5,7.5,10,10)
   #  -- Check headers
   ncols, nrows, xll, yll, cs, nd = _readHeaders(t4Fn)
   assert ncols == 5
   assert nrows == 5
   assert xll == 7.5
   assert yll == 7.5
   assert cs == 0.5
   assert nd == -999
   
   # -- Check content
   mtx4 = np.loadtxt(t4Fn, dtype=int, comments='', skiprows=6)
   cmp4 = np.array([line[15:] for line in TEST_SURFACE[:5]], dtype=int)
   assert np.array_equal(mtx4, cmp4)
   
