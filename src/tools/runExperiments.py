"""
@summary: Run a group of experiments
"""
import argparse
import os
from subprocess import Popen
from time import sleep

WORKER_CMD = ""

DIR_SLASH_TILE_SIZES = [
                        ('/home/cjgrady/thesis/.5degree/se/', .5, 'se.5', 'se5.out'),
                        ('/home/cjgrady/thesis/.5degree/fl/', .5, 'fl.5', 'fl5.out'),
                        ('/home/cjgrady/thesis/.5degree/pr/', .5, 'pr.5', 'pr5.out'),
                        ('/home/cjgrady/thesis/1degree/fl/', 1.0, 'fl1', 'fl1.out'),
                        ('/home/cjgrady/thesis/1degree/pr/', 1.0, 'pr1', 'pr1.out'),
                        ('/home/cjgrady/thesis/1degree/se/', 1.0, 'se1', 'se1.out'),
                        ('/home/cjgrady/thesis/2degree/fl/', 2.0, 'fl2', 'fl2.out'),
                        ('/home/cjgrady/thesis/2degree/pr/', 2.0, 'pr2', 'pr2.out'),
                        ('/home/cjgrady/thesis/2degree/se/', 2.0, 'se2', 'se2.out'),
                       ]
#STEP_SIZES = [.1]#.1, .2, .5]
STEP_SIZES = [1.0, .5, .2, .1]

COSTS_DIR = '/home/cjgrady/thesis/costs'
OUTPUTS_DIR = '/home/cjgrady/thesis/outputs'

# .............................................................................
def getCommand(pth, tileSize, stepSize, exp, outF):
   costsDir = os.path.join(COSTS_DIR, exp)
   outDir = os.path.join(OUTPUTS_DIR, exp)
   os.makedirs(costsDir)
   os.makedirs(outDir)
   return "{pyCmd} {pyMod} {inDir} {costDir} {outDir} {ts} {ss} {outFile}".format(pyCmd='python', 
               pyMod='/home/cjgrady/git/irksome-broccoli/src/multiTile/wqMultiTile.py',
               inDir=pth,
               costDir=costsDir,
               outDir=outDir,
               ss=stepSize,
               ts=tileSize,
               outFile=outF
               )

# .............................................................................
if __name__ == "__main__":
   #parser = argparse.ArgumentParser()
   #parser.add_argument('-w', '--workers', dest='workers', help="Number of workers", type=int)
   #args = parser.parse_args()
   
   #numWorkers = int(args.workers)
   
   # Start the workers
   #print "Starting", numWorkers, "workers"
   #for x in xrange(numWorkers):
   #   Popen(WORKER_CMD, shell=True)
   
   # Run the experiments
   for d, ts, exp, outF in DIR_SLASH_TILE_SIZES:
      for ss in STEP_SIZES:
         print "Starting %s, %s, %s" % (d, ts, ss)
         cmd = getCommand(d, ts, ss, '%s-%s' % (exp, ss), '%s-%s.out' % (exp, ss))
         p = Popen(cmd, shell=True)
         # Wait
         while p.poll() is None:
            sleep(1)
         print "Finished"
            