"""
@summary: Runs using Work Queue over multiple tiles
@author: CJ Grady
@version: 1.0
@status: release
@license: gpl2
"""
import argparse
import glob
import numpy as np
import os
import signal
import sys
import time
from work_queue import *

#import slr.singleTile.parallelDijkstra
import slr.singleTile.lcp

PYTHON_BIN = sys.executable
# Assume that work queue is in path
WORKER_BIN = "work_queue_worker"
import slr
pth = os.path.abspath(os.path.join(os.path.dirname(slr.__file__), '..'))
WORKER_PYTHONPATH = "export PYTHONPATH={pypth}".format(pypth=pth)

# .............................................................................
def getParallelDijkstraModulePath():
   """
   @summary: Get the module path for the parallel Dijkstra code.  This is used
                for commands to Work Queue
   """
   #return os.path.abspath(slr.singleTile.parallelDijkstra.__file__)
   return os.path.abspath(slr.singleTile.lcp.__file__)

# .............................................................................
class MultiTileWqParallelDijkstraLCP(object):
   """
   @summary: Runs a parallel version of Dijkstra's algorithm over multiple 
                tiles using Work Queue
   """
   # ...........................
   def __init__(self, inDir, costDir, outDir, tileSize, padding, summaryFn=None):
      """
      @summary: Constructor
      """
      self.inDir = inDir
      self.cDir = costDir
      self.oDir = outDir
      self.tileSize = tileSize
      self.padding = padding
      self.summaryFn = summaryFn
      self.grids = {}
      self.cc = 0
      self.tc = 0
      self.metrics = []

   # ...........................
   def _getGridFilename(self, d, minx, miny):
      """
      @summary: Get the filename for a grid matching the geographic parameters
      @param d: The directory where the grid is located
      @param minx: The minimum X value for the grid
      @param miny: The minimum Y value for the grid
      """
      return os.path.join(d, self.grids["{0},{1}".format(minx, miny)])
   
   # ...........................
   def _getConnectedTask(self, minx, miny, tag, sideArgs):
      """
      @summary: Submit a task to WorkQueue
      @param minx: The minimum X for the region
      @param miny: The minimum Y for the region
      @param tag: A tag to associate with the task
      """
      inGrid = self._getGridFilename(self.inDir, minx, miny)
      
      if os.path.exists(inGrid):
         task = Task('')
         
         print "Submitting task for grid:", minx, miny
         #if os.path.exists(self._getGridFilename(self.cDir, minx, miny)):
         #   m = np.loadtxt(self._getGridFilename(self.cDir, minx, miny), comments='', skiprows=6, dtype=int)

         summaryFn = self._getSummaryFile(tag)
         
         cmd = "{python} {pycmd} '{inGrid}' {costGrid} -w {outputsPath} -t {taskId} -p {padding} {sidesSec} -s {summaryFn}".format(
               python=PYTHON_BIN,
               pycmd=getParallelDijkstraModulePath(),
               inGrid=self._getGridFilename(self.inDir, minx, miny),
               costGrid=self._getGridFilename(self.cDir, minx, miny),
               outputsPath=self.oDir, ts=self.tileSize,
               taskId=tag, sidesSec=sideArgs,
               padding=self.padding, summaryFn=summaryFn)

         print "Submitting:", cmd
         task.specify_command(cmd)
         task.specify_output_file(summaryFn)
         task.specify_tag(str(tag))
         
         return task
      else:
         return None

   # ...........................
   def _getKey(self, minx, miny):
      """
      @summary: Get a key given a minimum x and y
      @param minx: Minimum X value for a tile
      @param miny: Minimum Y value for a tile
      """
      k = "{0},{1}".format(minx, miny)
      print k
      if self.grids.has_key(k):
         return self.grids[k]
      else:
         print self.grids
         return None

   # ...........................
   def _getStartupTask(self, minx, miny, tag):
      """
      @summary: Get a startup task for a region (no source vectors)
      @param minx: Minimum X value for a tile
      @param miny: Minimum Y value for a tile
      @param tag: A tag to associate with the task
      """
      task = Task('')
      summaryFn = self._getSummaryFile(tag)
      cmd = "{python} {pycmd} {inGrid} {costGrid} -w {outputsPath} -t {taskId} -s {summaryFn}".format(
            python=PYTHON_BIN,
            pycmd=getParallelDijkstraModulePath(),
            inGrid=self._getGridFilename(self.inDir, minx, miny),
            costGrid=self._getGridFilename(self.cDir, minx, miny),
            outputsPath=self.oDir, taskId=tag, summaryFn=summaryFn)
      print cmd
      task.specify_command(cmd)
      task.specify_output_file(summaryFn)
      task.specify_tag(str(tag))
      return task

   # ...........................
   def _getSummaryFile(self, taskId):
      """
      @summary: Get a filename for a location to store summary information 
                   for this process
      """
      return os.path.join(self.oDir, "%s-summary.txt" % taskId)

   # ...........................
   def _readOutputs(self, taskId):
      """
      @summary: Read the outputs of a task
      @param taskId: The id of the task to get outputs for
      """
      
      def stringOrNone(line):
         t = line.strip()
         if t.lower() == 'none':
            return None
         else:
            return t
      
      fn = self._getSummaryFile(taskId)
      with open(fn) as sumF:
         cnt = sumF.readlines()
      #cnt = open(self._getSummaryFile(taskId)).readlines()
      minx = float(cnt[0])
      miny = float(cnt[1])

      inLeft = stringOrNone(cnt[2])
      outLeft = stringOrNone(cnt[3])
      inTop = stringOrNone(cnt[4])
      outTop = stringOrNone(cnt[5])
      inRight = stringOrNone(cnt[6])
      outRight = stringOrNone(cnt[7])
      inBottom = stringOrNone(cnt[8])
      outBottom = stringOrNone(cnt[9])
      
      cc = int(cnt[10])

      self.metrics.append([minx, miny, cc])
      return (minx, miny, inLeft, outLeft, inTop, outTop, inRight, outRight, 
              inBottom, outBottom, cc)
   
   # ...........................
   def calculate(self):
      """
      @summary: Performs the calculation
      """
      stats = []
      aTime = time.time()
      currentTag = 1
      inputGrids = glob.glob(os.path.join(self.inDir, "*.asc"))
      
      #cctools_debug_flags_set("all")
      #cctools_debug_config_file("/tmp/myWQ.log")
      port = WORK_QUEUE_DEFAULT_PORT
      print "Port:" , port
      
      rGrids = []
      waitingGrids = {}
      
      q = WorkQueue(port=port)
      #print "Monitoring"
      #q.enable_monitoring_full('/tmp')
      
      for g in inputGrids:
         task = Task('')
         
         # TODO: Can we do something more elegant than this?
         # Need to figure out range of tiles
         # Replace '--' with '-!', this happens if we are negative
         # Also remove leading 'grid' and trailing '.asc'
         tmp1 = os.path.basename(g).replace('grid-', 'grid!').split('grid')[1].split('.asc')[0]
         tmpG = tmp1.replace('--', '-!')
         splitG = tmpG.split('-')
         #print splitG
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
            #print "Task output:", task.output
            
            stats.append((task.id, task.tag))#, task.resources_measured.memory, 
            #              task.resources_measured.virtual_memory, 
            #              task.resources_measured.cpu_time))
            # 
            if os.path.exists(os.path.join(self.oDir, '%s.error' % task.tag)):
               print "Error:"
               print open(os.path.join(self.oDir, '%s.error' % task.tag)).read()
               
            if os.path.exists(self._getSummaryFile(task.tag)):
               minx, miny, il, ol, it, ot, ir, oR, ib, ob, cc = self._readOutputs(task.tag)
               #print minx, miny, maxx, maxy, l, t, r, b, cc
               self.cc += cc
               self.tc += 1
               
               k = self._getKey(minx, miny)
               print "Removing", k, "from running list", task.tag
               try:
                  rGrids.remove(k)
               except Exception, e:
                  print str(e)
                  print rGrids
                  raise e
            
               # Add any tasks that were waiting on this tile to finish
               if waitingGrids.has_key(k):
                  
                  sideArgs = waitingGrids.pop(k)
                  
                  tag = currentTag
                  currentTag += 1
                  
                  nTask = self._getConnectedTask(minx, miny, tag, sideArgs)
                                                 
                  if nTask is not None:
                     rGrids.append(k)
                     print "Added", k, "to running list", tag
                     q.submit(nTask)
               
               # Add adjacent tiles as necessary
               if ol is not None:
                  
                  myKey = self._getKey(minx-self.tileSize, miny)
                  if myKey is not None:
                     sideArgs = " -sl {} -il {}".format(ol, il)
                     if myKey in rGrids:
                        if not waitingGrids.has_key(myKey):
                           waitingGrids[myKey] = ""
                        waitingGrids[myKey] += sideArgs
                        print "Waiting for:", myKey
                     else:
                        print "Submitting for:", myKey
                        tag = currentTag
                        currentTag += 1
                        nTask = self._getConnectedTask(minx-self.tileSize, miny, tag, sideArgs)
                        if nTask is not None:
                           rGrids.append(myKey)
                           print "Added", myKey, "to running list", tag
                           q.submit(nTask)

               if ot is not None:
                  myKey = self._getKey(minx, miny+self.tileSize)
                  if myKey is not None:
                     sideArgs = " -st {} -it {}".format(ot, it)
                     if myKey in rGrids:
                        if not waitingGrids.has_key(myKey):
                           waitingGrids[myKey] = ""
                        waitingGrids[myKey] += sideArgs
                        print "Waiting for:", myKey
                     else:
                        print "Submitting for:", myKey
                        tag = currentTag
                        currentTag += 1
                        nTask = self._getConnectedTask(minx, miny+self.tileSize, tag, sideArgs)
                        if nTask is not None:
                           rGrids.append(myKey)
                           print "Added", myKey, "to running list", tag
                           q.submit(nTask)

               if oR is not None:
                  myKey = self._getKey(minx+self.tileSize, miny)
                  if myKey is not None:
                     sideArgs = " -sr {} -ir {}".format(oR, ir)
                     if myKey in rGrids:
                        if not waitingGrids.has_key(myKey):
                           waitingGrids[myKey] = ""
                        waitingGrids[myKey] += sideArgs
                        print "Waiting for:", myKey
                     else:
                        print "Submitting for:", myKey
                        tag = currentTag
                        currentTag += 1
                        nTask = self._getConnectedTask(minx+self.tileSize, miny, tag, sideArgs)
                        if nTask is not None:
                           rGrids.append(myKey)
                           print "Added", myKey, "to running list", tag
                           q.submit(nTask)

               if ob is not None:
                  myKey = self._getKey(minx, miny-self.tileSize)
                  if myKey is not None:
                     sideArgs = " -sb {} -ib {}".format(ob, ib)
                     if myKey in rGrids:
                        if not waitingGrids.has_key(myKey):
                           waitingGrids[myKey] = ""
                        waitingGrids[myKey] += sideArgs
                        print "Waiting for:", myKey
                     else:
                        print "Submitting for:", myKey
                        tag = currentTag
                        currentTag += 1
                        nTask = self._getConnectedTask(minx, miny-self.tileSize, tag, sideArgs)
                        if nTask is not None:
                           rGrids.append(myKey)
                           print "Added", myKey, "to running list", tag
                           q.submit(nTask)
            else:
               print task.id
               print task.command
               print task.output
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
            for s in stats:
               outF.write("%s\n" % ', '.join([str(i) for i in s]))
   
   # ...........................
   def getMetrics(self):
      """
      @summary: Retrieve computation metrics
      """
      return self.metrics
   
   # ...........................
   def startWorkers(self, numWorkers):
      """
      @summary: Start work queue workers to perform computations
      @param numWorkers: The number of workers to start
      """
      import subprocess
      
      self.workers = []
      for i in range(numWorkers):
         cmd = "{0}; {1} {2} {3}".format(WORKER_PYTHONPATH, WORKER_BIN, '127.0.0.1', WORK_QUEUE_DEFAULT_PORT)
         print cmd
         #cmd = "{0} {1} {2}".format(WORKER_BIN, '127.0.0.1', WORK_QUEUE_DEFAULT_PORT)
         self.workers.append(subprocess.Popen(cmd, shell=True))
   
   # ...........................
   def stopWorkers(self):
      """
      @summary: Stop running workers
      """
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
   parser.add_argument('padding', type=int)
   parser.add_argument('outputFile', type=str)

   args = parser.parse_args()
   
   inDir = args.inputDir
   cDir = args.cDir
   oDir = args.oDir
   padding = args.padding
   ts = args.tileSize
   
   myInstance = MultiTileWqParallelDijkstraLCP(inDir, cDir, oDir, ts, padding, 
                                               summaryFn=args.outputFile)
   print "Starting workers"
   myInstance.startWorkers(2)
   myInstance.calculate()
   print "Stopping workers"
   myInstance.stopWorkers()
   print "Done"
