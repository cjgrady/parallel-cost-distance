"""
@summary: Run a group of experiments
"""
import argparse
from subprocess import Popen
from time import sleep

WORKER_CMD = ""

DIR_SLASH_TILE_SIZES = [
                        (),
                       ]
STEP_SIZES = [.005, .01, .05]

# .............................................................................
def getCommand(pth, tileSize, stepSize):
   return "python experiment.py %s %d %d" % (pth, tileSize, stepSize)

# .............................................................................
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('-w', '--workers', dest='workers', help="Number of workers", type=int)
   args = parser.parse_args()
   
   numWorkers = int(args.workers)
   
   # Start the workers
   print "Starting", numWorkers, "workers"
   for x in xrange(numWorkers):
      Popen(WORKER_CMD, shell=True)
   
   # Run the experiments
   for d, ts in DIR_SLASH_TILE_SIZES:
      for ss in STEP_SIZES:
         print "Starting %s, %s, %s" % (d, ts, ss)
         cmd = getCommand(d, ts, ss)
         p = Popen(cmd, shell=True)
         # Wait
         while p.poll() is None:
            sleep(1)
         print "Finished"
            