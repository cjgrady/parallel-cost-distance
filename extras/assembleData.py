"""
@summary: This script assembles the existing surfaces for an experiment
"""
import glob
import os
import shutil

# Full paths to directories containing data
DATA_DIR = "/home/cjgrady/ngdc/"
TWO_DEGREE_PATH = os.path.join(DATA_DIR, "twoDegree")
ONE_DEGREE_PATH = os.path.join(DATA_DIR, "oneDegree")
HALF_DEGREE_PATH = os.path.join(DATA_DIR, "halfDegree")

# Relative sub directory names
TWO_DEGREE = "two"
ONE_DEGREE = "one"
HALF_DEGREE = "half"

# Output directories
OUTPUT_DIR = "/home/cjgrady/exp5outputs"


# Four degree blocks (corners)
BLOCKS = [
   (-165.0, 15.5), (-163.0, 15.5), (-161.0, 15.5), (-130.5, 31.5), 
   (-130.5, 33.5), (-130.5, 35.5), (-130.0, 40.0), (-130.0, 42.0), 
   (-128.5, 31.5), (-128.5, 33.5), (-128.5, 35.5), (-128.0, 40.0), 
   (-128.0, 42.0), (-103.5, 21.5), (-103.5, 23.5), (-101.5, 21.5), 
   (-101.5, 23.5), ( -99.5, 20.0), ( -99.5, 22.0), ( -99.5, 24.0), 
   ( -97.5, 20.0), ( -97.5, 22.0), ( -97.5, 24.0), ( -95.5, 20.0), 
   ( -95.5, 22.0), ( -95.5, 24.0), ( -93.0, 20.0), ( -93.0, 22.0), 
   ( -93.0, 24.0), ( -91.0, 20.0), ( -91.0, 22.0), ( -91.0, 24.0), 
   ( -90.5, 24.5), ( -90.5, 26.5), ( -90.5, 28.5), ( -90.5, 30.5), 
   ( -89.0, 20.0), ( -89.0, 22.0), ( -89.0, 24.0), ( -88.5, 24.5), 
   ( -88.5, 26.5), ( -88.5, 28.5), ( -88.5, 30.5), ( -87.0, 20.0), 
   ( -87.0, 22.0), ( -87.0, 24.0), ( -86.5, 24.5), ( -86.5, 26.5), 
   ( -86.5, 28.5), ( -86.5, 30.5), ( -84.5, 24.5), ( -84.5, 26.5), 
   ( -84.5, 28.5), ( -84.5, 30.5), ( -84.5, 35.5), ( -84.5, 37.5), 
   ( -82.5, 24.5), ( -82.5, 26.5), ( -82.5, 28.5), ( -82.5, 30.5), 
   ( -82.5, 35.5), ( -82.5, 37.5), ( -80.5, 24.5), ( -80.5, 26.5), 
   ( -80.5, 28.5), ( -80.5, 30.5), ( -80.5, 35.5), ( -80.5, 37.5), 
   ( -78.5, 35.5), ( -78.5, 37.5), ( -76.5, 35.5), ( -76.5, 37.5)
]

# Four degree blocks
for lon, lat in BLOCKS:
   myDir = os.path.join(OUTPUT_DIR, '4blocks', "{0}_-{1}".format(lon, lat))
   
   # Make directory
   if not os.path.exists(myDir):
      os.mkdir(myDir)

   # 2 degree
   twoDdir = os.path.join(myDir, TWO_DEGREE)
   if not os.path.exists(twoDdir):
      os.mkdir(twoDdir)
   
   for x in range(0, 4, 2):
      for y in range(0, 4, 2):
         startLon = lon + x
         startLat = lat + y
         fn = "grid{0}-{1}-{2}-{3}.asc".format(startLon, startLat, startLon+2, startLat+2)
         pth = os.path.join(TWO_DEGREE_PATH, fn)
         shutil.copy(pth, os.path.join(twoDdir, fn))

   # 1 degree
   oneDdir = os.path.join(myDir, ONE_DEGREE)
   if not os.path.exists(oneDdir):
      os.mkdir(oneDdir)
   
   for x in range(0, 4):
      for y in range(0, 4):
         startLon = lon + x
         startLat = lat + y
         fn = "grid{0}-{1}-{2}-{3}.asc".format(startLon, startLat, startLon+1, startLat+1)
         pth = os.path.join(ONE_DEGREE_PATH, fn)
         print pth, os.path.exists(pth)
         shutil.copy(pth, os.path.join(oneDdir, fn))

   # Half degree
   halfDdir = os.path.join(myDir, HALF_DEGREE)
   if not os.path.exists(halfDdir):
      os.mkdir(halfDdir)
   
   x = 0
   while x < 4:
      y = 0
      while y < 4:
         startLon = lon + x
         startLat = lat + y
         fn = "grid{0}-{1}-{2}-{3}.asc".format(startLon, startLat, startLon+.5, startLat+.5)
         pth = os.path.join(HALF_DEGREE_PATH, fn)
         print pth, os.path.exists(pth)
         shutil.copy(pth, os.path.join(halfDdir, fn))
         y += .5
      x += .5

   

