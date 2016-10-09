"""
@summary: Runs using Work Queue over multiple tiles
@author: CJ Grady
"""
import argparse
import glob
import os
import signal
import sys
import time
from work_queue import *

import slr.singleTile.parallelDijkstra

PYTHON_BIN = sys.executable
# Assume that work queue is in path
WORKER_BIN = "work_queue_worker"
#PATH = "export PYTHONPATH=/home/cjgrady/cctools/lib/python2.7/site-packages/:/home/cjgrady/git/irksome-broccoli/"
# PYTHONPATH to export for workers
WORKER_PYTHONPATH = ':'.join(sys.path)

# .............................................................................
def getParallelDijkstraModulePath():
   print slr.singleTile.parallelDijkstra.__file__
   return os.path.abspath(slr.singleTile.parallelDijkstra.__file__)

# .............................................................................
class MultiTileWqParallelDijkstraLCP(object):
   """
   @summary: Runs a parallel version of Dijkstra's algorithm over multiple 
                tiles using Work Queue
   """
   # ...........................
   def __init__(self, inDir, costDir, outDir, tileSize, stepSize, summaryFn=None):
      self.inDir = inDir
      self.cDir = costDir
      self.oDir = outDir
      self.tileSize = tileSize
      self.stepSize = stepSize
      self.summaryFn = summaryFn

   # ...........................
   def _getGridFilename(self, d, minx, miny, maxx, maxy):
      return os.path.join(d, "grid%s-%s-%s-%s.asc" % (minx, miny, maxx, maxy))
   
   # ...........................
   def _getConnectedTask(self, minx, miny, maxx, maxy, vects, fromSides, tag):
      inGrid = self._getGridFilename(self.inDir, minx, miny, maxx, maxy)
      
      if os.path.exists(inGrid):
         task = Task('')
         
         vectsSec = ' '.join(['-v %s' % v for v in vects])
         sidesSec = ' '.join(['-s %s' % s for s in fromSides])
         
         cmd = "{python} {pycmd} {inGrid} {costGrid} -g 1 -o {outputsPath} -w 50 -t {taskId} --step={ss} --ts={ts} {vectsSec} {sidesSec}".format(
               python=PYTHON_BIN,
               #pycmd='~/git/irksome-broccoli/src/singleTile/parallelDijkstra.py',
               pycmd=getParallelDijkstraModulePath(),
               inGrid=self._getGridFilename(self.inDir, minx, miny, maxx, maxy),
               costGrid=self._getGridFilename(self.cDir, minx, miny, maxx, maxy),
               outputsPath=self.oDir, ss=self.stepSize, ts=self.tileSize,
               taskId=tag, vectsSec=vectsSec, sidesSec=sidesSec)
         task.specify_command(cmd)
         task.specify_output_file(self._getSummaryFile(tag))
         task.specify_tag(str(tag))
         return task
      else:
         return None

   # ...........................
   def _getKey(self, minx, miny, maxx, maxy):
      return "{0},{1},{2},{3}".format(minx, miny, maxx, maxy)

   # ...........................
   def _getStartupTask(self, minx, miny, maxx, maxy, tag):
      task = Task('')
      cmd = "{python} {pycmd} {inGrid} {costGrid} -g 1 -o {outputsPath} -w 50 -t {taskId} --step={ss} --ts={ts} -e {e}".format(
            python=PYTHON_BIN,
            pycmd=getParallelDijkstraModulePath(),
            inGrid=self._getGridFilename(self.inDir, minx, miny, maxx, maxy),
            costGrid=self._getGridFilename(self.cDir, minx, miny, maxx, maxy),
            ss=self.stepSize, ts=self.tileSize, outputsPath=self.oDir, taskId=tag,
            e=os.path.join(self.oDir, '%s.error' % tag))
      task.specify_command(cmd)
      task.specify_output_file(self._getSummaryFile(tag))
      task.specify_tag(str(tag))
      return task

   # ...........................
   def _getSummaryFile(self, taskId):
      return os.path.join(self.oDir, "%s-summary.txt" % taskId)

   # ...........................
   def _getVectorFilename(self, taskId, d):
      dirPart = ['toLeft', 'toTop', 'toRight', 'toBottom'][d]
      return os.path.join(self.oDir, "%s-%s.npy" % (taskId, dirPart))
   
   # ...........................
   def _readOutputs(self, taskId):
      #TODO: I changed this from task.tag that could not exist
      cnt = open(self._getSummaryFile(taskId)).readlines()
      print cnt
      minx = float(cnt[0])
      miny = float(cnt[1])
      maxx = float(cnt[2])
      maxy = float(cnt[3])
      l = cnt[4].lower().strip() == 'true'
      t = cnt[5].lower().strip() == 'true'
      r = cnt[6].lower().strip() == 'true'
      b = cnt[7].lower().strip() == 'true'
      cc = int(cnt[8])
      
      return minx, miny, maxx, maxy, l, t, r, b, cc
   
   # ...........................
   def calculate(self):
      """
      @summary: Performs the calculation
      """
      aTime = time.time()
      currentTag = 1
      inputGrids = glob.glob(os.path.join(self.inDir, "*.asc"))
      
      port = WORK_QUEUE_DEFAULT_PORT
      print "Port:" , port
      
      rGrids = []
      waitingGrids = {}
      
      q = WorkQueue(port=port)
      
      for g in inputGrids:
         task = Task('')
         
         # TODO: Can we do something more elegant than this?
         # Need to figure out range of tiles
         # Replace '--' with '-!', this happens if we are negative
         # Also remove leading 'grid' and trailing '.asc'
         tmpG = os.path.basename(g).replace('--', '-!').replace('grid', '').replace('.asc', '')
         splitG = tmpG.split('-')
         print splitG
         # Replace inserted !s with -s for negative
         minx = splitG[0].replace('!', '-')
         miny = splitG[1].replace('!', '-')
         maxx = splitG[2].replace('!', '-')
         maxy = splitG[3].replace('!', '-')

         tag = currentTag
         currentTag += 1
         print minx, miny, maxx, maxy
         k = self._getKey(minx, miny, maxx, maxy)
         if not k in rGrids:
            task = self._getStartupTask(minx, miny, maxx, maxy, tag)
            #k = '%s,%s,%s,%s' % (minx, miny, maxx, maxy)
            rGrids.append(k)
            print "Added", k, "to running list", tag
         
            print "Submitting task:", task.tag
            q.submit(task)
   
      r = 0
      while not q.empty() and r < 1000:
         # Wait a maximum of 10 seconds for a task to come back.
         task = q.wait(1)
         r += 1
         if task:
            r = 0
            print "Task id:", task.id
            print "Task tag:", task.tag
            
            if os.path.exists(self._getSummaryFile(task.tag)):
               minx, miny, maxx, maxy, l, t, r, b, cc = self._readOutputs(task.tag)
               print "Changed", cc, "cells"
               
               k = self._getKey(minx, miny, maxx, maxy)
               print "Removing", k, "from running list", task.tag
               try:
                  rGrids.remove(k)
               except:
                  print rGrids
                  raise
            
               # Add any tasks that were waiting on this tile to finish
               #tKey = '%s,%s,%s,%s' % (minx, miny, maxx, maxy)
               if waitingGrids.has_key(k):
                  sides = waitingGrids.pop(k)
                  ss,vs = zip(*sides)
                  tag = currentTag
                  currentTag += 1
                  nTask = self._getConnectedTask(minx, miny, maxx, maxy, vs, ss, tag)
                  if nTask is not None:
                     rGrids.append(k)
                     print "Added", k, "to running list", tag
                     q.submit(nTask)
               
               # Add adjacent tiles as necessary
               if l:
                  vect = self._getVectorFilename(task.tag, 0)
                  
                  #myKey = '%s,%s,%s,%s' % (minx-self.tileSize, miny, maxx-self.tileSize, maxy)
                  myKey = self._getKey(minx-self.tileSize, miny, minx, maxy)
                  if myKey in rGrids:
                     if not waitingGrids.has_key(myKey):
                        waitingGrids[myKey] = []
                     waitingGrids[myKey].append((2, vect))
                     print "Waiting for:", myKey
                  else:
                     print "Submitting for:", myKey
                     tag = currentTag
                     currentTag += 1
                     nTask = self._getConnectedTask(minx-self.tileSize, miny, miny, maxy, [vect], [2], tag)
                     if nTask is not None:
                        rGrids.append(myKey)
                        print "Added", myKey, "to running list", tag
                        q.submit(nTask)
               if t:
                  vect = self._getVectorFilename(task.tag, 1)
                  
                  #myKey = '%s,%s,%s,%s' % (minx, miny+self.tileSize, maxx, maxy+self.tileSize)
                  myKey = self._getKey(minx, maxy, maxx, maxy+self.tileSize)
                  if myKey in rGrids:
                     if not waitingGrids.has_key(myKey):
                        waitingGrids[myKey] = []
                     waitingGrids[myKey].append((3, vect))
                     print "Waiting for:", myKey
                  else:
                     print "Submitting for:", myKey
                     tag = currentTag
                     currentTag += 1
                     nTask = self._getConnectedTask(minx, maxy, maxx, maxy+self.tileSize, [vect], [3], tag)
                     if nTask is not None:
                        rGrids.append(myKey)
                        print "Added", myKey, "to running list", tag
                        q.submit(nTask)
               if r:
                  vect = self._getVectorFilename(task.tag, 2)
   
                  #myKey = '%s,%s,%s,%s' % (minx+self.tileSize, miny, maxx+self.tileSize, maxy)
                  myKey = self._getKey(maxx, miny, maxx+self.tileSize, maxy)
                  if myKey in rGrids:
                     if not waitingGrids.has_key(myKey):
                        waitingGrids[myKey] = []
                     waitingGrids[myKey].append((0, vect))
                     print "Waiting for:", myKey
                  else:
                     print "Submitting for:", myKey
                     tag = currentTag
                     currentTag += 1
                     nTask = self._getConnectedTask(maxx, miny, maxx+self.tileSize, maxy, [vect], [0], tag)
                     if nTask is not None:
                        rGrids.append(myKey)
                        print "Added", myKey, "to running list", tag
                        q.submit(nTask)
               if b:
                  vect = self._getVectorFilename(task.tag, 3)
                  #myKey = '%s,%s,%s,%s' % (minx, miny-self.tileSize, maxx, maxy-self.tileSize)
                  myKey = self._getKey(minx, miny-self.tileSize, maxx, miny)
                  if myKey in rGrids:
                     if not waitingGrids.has_key(myKey):
                        waitingGrids[myKey] = []
                     waitingGrids[myKey].append((1, vect))
                     print "Waiting for:", myKey
                  else:
                     print "Submitting for:", myKey
                     tag = currentTag
                     currentTag += 1
                     nTask = self._getConnectedTask(minx, miny-self.tileSize, maxx, miny, [vect], [1], tag)
                     if nTask is not None:
                        rGrids.append(myKey)
                        print "Added", myKey, "to running list", tag
                        q.submit(nTask)
            else:
               print task.id
               print task.command
               print task.output
               print dir(task)
         else:
            pass
            #print q.__dict__
      bTime = time.time()
      
      print bTime - aTime
      # Write summary
      if self.summaryFn is not None:
         with open(self.summaryFn, 'w') as outF:
            if r >= 1000:
               outF.write("-1\n")
            outF.write('%s\n' % (bTime - aTime))
   
   def startWorkers(self, numWorkers):
      import subprocess
      
      self.workers = []
      for i in range(numWorkers):
         cmd = "{0}; {1} {2} {3}".format(WORKER_PYTHONPATH, WORKER_BIN, '127.0.0.1', WORK_QUEUE_DEFAULT_PORT)
         #cmd = "{0} {1} {2}".format(WORKER_BIN, '127.0.0.1', WORK_QUEUE_DEFAULT_PORT)
         self.workers.append(subprocess.Popen(cmd, shell=True))
   
   def stopWorkers(self):
      for w in self.workers:
         print "Sending kill signal"
         os.killpg(w.pid, signal.SIGTERM)
      
# .............................................................................
if __name__ == "__main__":

   # Read inputs
   parser = argparse.ArgumentParser()
   parser.add_argument('inputDir', type=str)
   parser.add_argument('cDir', type=str)
   parser.add_argument('oDir', type=str)
   parser.add_argument('tileSize', type=float)
   parser.add_argument('stepSize', type=float)
   parser.add_argument('outputFile', type=str)

   args = parser.parse_args()
   
   inDir = args.inputDir
   cDir = args.cDir
   oDir = args.oDir
   stepSize = args.stepSize
   ts = args.tileSize
   
   myInstance = MultiTileWqParallelDijkstraLCP(inDir, cDir, oDir, ts, stepSize, 
                                               summaryFn=args.outputFile)
   print "Starting workers"
   myInstance.startWorkers(2)
   myInstance.calculate()
   print "Stopping workers"
   myInstance.stopWorkers()
   print "Done"
