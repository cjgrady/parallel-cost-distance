"""
@summary: Runs using Work Queue over multiple tiles
@author: CJ Grady
"""
import argparse
import glob
import numpy as np
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
import slr
pth = os.path.abspath(os.path.join(os.path.dirname(slr.__file__), '..'))
WORKER_PYTHONPATH = "export PYTHONPATH={pypth}".format(pypth=pth)
#WORKER_PYTHONPATH = "export PYTHONPATH={0}".format(':'.join(sys.path))

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
      self.grids = {}

   # ...........................
   def _getGridFilename(self, d, minx, miny):
      return os.path.join(d, self.grids["{0},{1}".format(minx, miny)])
   
   # ...........................
   def _getConnectedTask(self, minx, miny, vects, fromSides, tag): # Removed max x and max y
      inGrid = self._getGridFilename(self.inDir, minx, miny)
      
      if os.path.exists(inGrid):
         task = Task('')
         
         vectsSec = ' '.join(['-v %s' % v for v in vects])
         sidesSec = ' '.join(['-s %s' % s for s in fromSides])
         
         print "Submitting task for grid:", minx, miny
         if os.path.exists(self._getGridFilename(self.cDir, minx, miny)):
            m = np.loadtxt(self._getGridFilename(self.cDir, minx, miny), comments='', skiprows=6, dtype=int)
         
         cmd = "{python} {pycmd} {inGrid} {costGrid} -g 1 -o {outputsPath} -w 50 -t {taskId} --step={ss} --ts={ts} {vectsSec} {sidesSec} -e {e}".format(
               python=PYTHON_BIN,
               #pycmd='~/git/irksome-broccoli/src/singleTile/parallelDijkstra.py',
               pycmd=getParallelDijkstraModulePath(),
               inGrid=self._getGridFilename(self.inDir, minx, miny),
               costGrid=self._getGridFilename(self.cDir, minx, miny),
               outputsPath=self.oDir, ss=self.stepSize, ts=self.tileSize,
               taskId=tag, vectsSec=vectsSec, sidesSec=sidesSec,
               e=os.path.join(self.oDir, '%s.error' % tag))
         print "Submitting:"
         print cmd
         task.specify_command(cmd)
         task.specify_output_file(self._getSummaryFile(tag))
         task.specify_tag(str(tag))
         
         return task
      else:
         return None

   # ...........................
   def _getKey(self, minx, miny):
      k = "{0},{1}".format(minx, miny)
      if self.grids.has_key(k):
         return self.grids[k]
      else:
         return None

   # ...........................
   def _getStartupTask(self, minx, miny, tag):
      task = Task('')
      cmd = "{python} {pycmd} {inGrid} {costGrid} -g 1 -o {outputsPath} -w 50 -t {taskId} --step={ss} --ts={ts} -e {e}".format(
            python=PYTHON_BIN,
            pycmd=getParallelDijkstraModulePath(),
            inGrid=self._getGridFilename(self.inDir, minx, miny),
            costGrid=self._getGridFilename(self.cDir, minx, miny),
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
         self.grids["{0},{1}".format(minx, miny)] = os.path.basename(g)
         
         tag = currentTag
         currentTag += 1
         k = self._getKey(minx, miny)
         if not k in rGrids:
            task = self._getStartupTask(minx, miny, tag)
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
            
            if os.path.exists(os.path.join(self.oDir, '%s.error' % task.tag)):
               print open(os.path.join(self.oDir, '%s.error' % task.tag)).read()
               
            if os.path.exists(self._getSummaryFile(task.tag)):
               minx, miny, maxx, maxy, l, t, r, b, cc = self._readOutputs(task.tag)
               print "Changed", cc, "cells"
               
               print "Calculated result"
               m = np.loadtxt(self._getGridFilename(self.cDir, minx, miny), comments='', skiprows=6, dtype=int)
               print m.tolist()
               
               k = self._getKey(minx, miny)
               print "Removing", k, "from running list", task.tag
               try:
                  rGrids.remove(k)
               except:
                  print rGrids
                  raise
            
               # Add any tasks that were waiting on this tile to finish
               if waitingGrids.has_key(k):
                  # Having issues with Travis so only working on one side at a time
                  if len(waitingGrids[k]) > 1:
                     l = len(waitingGrids[k])
                     tmp = waitingGrids[k].pop(0)
                     assert len(waitingGrids[k]) < l 
                     # Check that pop is modifying dictionary
                  else:
                     tmp = waitingGrids.pop(k)[0]
                  print tmp
                  ss, vs = tmp
                  #sides = waitingGrids.pop(k)
                  #print "Sides:", sides
                  #ss,vs = zip(*sides)
                  tag = currentTag
                  currentTag += 1
                  print minx, miny
                  print vs
                  print ss
                  nTask = self._getConnectedTask(minx, miny, [vs], [ss], tag)
                  if nTask is not None:
                     rGrids.append(k)
                     print "Added", k, "to running list", tag
                     q.submit(nTask)
               
               # Add adjacent tiles as necessary
               if l:
                  vect = self._getVectorFilename(task.tag, 0)
                  print "Left:", minx, miny
                  print np.load(vect).tolist()
                  
                  myKey = self._getKey(minx-self.tileSize, miny)
                  if myKey is not None:
                     if myKey in rGrids:
                        if not waitingGrids.has_key(myKey):
                           waitingGrids[myKey] = []
                        waitingGrids[myKey].append((2, vect))
                        print "Waiting for:", myKey
                     else:
                        print "Submitting for:", myKey
                        tag = currentTag
                        currentTag += 1
                        nTask = self._getConnectedTask(minx-self.tileSize, miny, [vect], [2], tag)
                        if nTask is not None:
                           rGrids.append(myKey)
                           print "Added", myKey, "to running list", tag
                           q.submit(nTask)
               if t:
                  vect = self._getVectorFilename(task.tag, 1)
                  print "Top:", minx, miny
                  print np.load(vect).tolist()
                  
                  myKey = self._getKey(minx, maxy)
                  if myKey is not None:
                     if myKey in rGrids:
                        if not waitingGrids.has_key(myKey):
                           waitingGrids[myKey] = []
                        waitingGrids[myKey].append((3, vect))
                        print "Waiting for:", myKey
                     else:
                        print "Submitting for:", myKey
                        tag = currentTag
                        currentTag += 1
                        nTask = self._getConnectedTask(minx, maxy, [vect], [3], tag)
                        if nTask is not None:
                           rGrids.append(myKey)
                           print "Added", myKey, "to running list", tag
                           q.submit(nTask)
               if r:
                  vect = self._getVectorFilename(task.tag, 2)
                  print "Right:", minx, miny
                  print np.load(vect).tolist()
   
                  myKey = self._getKey(maxx, miny)
                  if myKey is not None:
                     if myKey in rGrids:
                        if not waitingGrids.has_key(myKey):
                           waitingGrids[myKey] = []
                        waitingGrids[myKey].append((0, vect))
                        print "Waiting for:", myKey
                     else:
                        print "Submitting for:", myKey
                        tag = currentTag
                        currentTag += 1
                        nTask = self._getConnectedTask(maxx, miny, [vect], [0], tag)
                        if nTask is not None:
                           rGrids.append(myKey)
                           print "Added", myKey, "to running list", tag
                           q.submit(nTask)
               if b:
                  vect = self._getVectorFilename(task.tag, 3)
                  print "Bottom:", minx, miny
                  print np.load(vect).tolist()
                  myKey = self._getKey(minx, miny-self.tileSize)
                  if myKey is not None:
                     if myKey in rGrids:
                        if not waitingGrids.has_key(myKey):
                           waitingGrids[myKey] = []
                        waitingGrids[myKey].append((1, vect))
                        print "Waiting for:", myKey
                     else:
                        print "Submitting for:", myKey
                        tag = currentTag
                        currentTag += 1
                        nTask = self._getConnectedTask(minx, miny-self.tileSize, [vect], [1], tag)
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
         print cmd
         #cmd = "{0} {1} {2}".format(WORKER_BIN, '127.0.0.1', WORK_QUEUE_DEFAULT_PORT)
         self.workers.append(subprocess.Popen(cmd, shell=True))
   
   def stopWorkers(self):
      for w in self.workers:
         print "Sending kill signal"
         #os.killpg(w.pid, signal.SIGTERM)
         w.kill()
      
# .............................................................................
if __name__ == "__main__": # pragma: no cover

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
