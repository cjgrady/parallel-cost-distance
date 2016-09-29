"""
@summary: Runs using Work Queue over multiple tiles
"""
import argparse
import glob
import os
import time
from work_queue import *

OUTPUTS_PATH = "/home/cjgrady/thesis/outputs/"
#INPUT_PATH = "/home/cjgrady/thesis/degreeGrids/"
INPUT_PATH = "/home/cjgrady/thesis/flTest/"
COSTS_PATH = "/home/cjgrady/thesis/outputs/"

# .............................................................................
def getInputGridFilename(inDir, minx, miny, maxx, maxy):
   return os.path.join(inDir, "grid%s-%s-%s-%s.asc" % (minx, miny, maxx, maxy))

# .............................................................................
def getCostGridFilename(cDir, minx, miny, maxx, maxy):
   return os.path.join(cDir, "grid%s-%s-%s-%s.asc" % (minx, miny, maxx, maxy))

# .............................................................................
def getVectorFilename(oDir, taskId, d):
   dirPart = ['toLeft', 'toTop', 'toRight', 'toBottom'][d]
   
   return os.path.join(oDir, "%s-%s.npy" % (taskId, dirPart))
   
# .............................................................................
def getSummaryFile(oDir, taskId):
   return os.path.join(oDir, "%s-summary.txt" % taskId)

# .............................................................................
def readOutputs(oDir, taskId):
   cnt = open(getSummaryFile(oDir, task.tag)).readlines()
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
   
# .............................................................................
def getStartupTask(minx, miny, maxx, maxy, tag, inDir, cDir, oDir, stepSize, tileSize):
   task = Task('')
   cmd = "{python} {pycmd} {inGrid} {costGrid} -g 1 -o {outputsPath} -w 50 -t {taskId} --step={ss} --ts={ts}".format(
            python='python',
            pycmd='~/git/irksome-broccoli/src/singleTile/parallelDijkstra.py',
            inGrid=getInputGridFilename(inDir, minx, miny, maxx, maxy),
            costGrid=getCostGridFilename(cDir, minx, miny, maxx, maxy),
            ss=stepSize, ts=tileSize, outputsPath=oDir, taskId=tag)
   task.specify_command(cmd)
   task.specify_output_file(getSummaryFile(oDir, tag))
   task.specify_tag(str(tag))
   return task

# .............................................................................
def getConnectedTask(minx, miny, maxx, maxy, vects, fromSides, tag, inDir, cDir, oDir, stepSize, ts):
   inGrid = getInputGridFilename(inDir, minx, miny, maxx, maxy)
   
   if os.path.exists(inGrid):
      task = Task('')
      
      vectsSec = ' '.join(['-v %s' % v for v in vects])
      sidesSec = ' '.join(['-s %s' % s for s in fromSides])
      
      cmd = "{python} {pycmd} {inGrid} {costGrid} -g 1 -o {outputsPath} -w 50 -t {taskId} --step={ss} --ts={ts} {vectsSec} {sidesSec}".format(
            python='python',
            pycmd='~/git/irksome-broccoli/src/singleTile/parallelDijkstra.py',
            inGrid=getInputGridFilename(inDir, minx, miny, maxx, maxy),
            costGrid=getCostGridFilename(cDir, minx, miny, maxx, maxy),
            outputsPath=oDir, 
            ss=stepSize,
            ts=ts,
            taskId=tag,
            vectsSec=vectsSec,
            sidesSec=sidesSec)
      task.specify_command(cmd)
      task.specify_output_file(getSummaryFile(oDir, tag))
      task.specify_tag(str(tag))
      return task
   else:
      return None

def getKey(minx, miny, maxx, maxy):
   return "{0},{1},{2},{3}".format(minx, miny, maxx, maxy)

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
   
   print args.inputDir
   print args.tileSize
   print args.stepSize

   inDir = args.inputDir
   cDir = args.cDir
   oDir = args.oDir
   stepSize = args.stepSize
   ts = args.tileSize

   aTime = time.time()
   currentTag = 1
   inputGrids = glob.glob(os.path.join(inDir, "*.asc"))
   
   port = WORK_QUEUE_DEFAULT_PORT
   print "Port:" , port
   
   rGrids = []
   waitingGrids = {}
   
   q = WorkQueue(port=port)
   
   for g in inputGrids:
      task = Task('')
      minx = '-%s' % g.split('-')[1]
      miny = '%s' % g.split('-')[2]
      maxx = '-%s' % g.split('-')[4]
      #maxy = '%s' % g.split('-')[5].split('.')[0]
      maxy = '%s' % g.split('-')[5].strip('.asc')
      tag = currentTag
      currentTag += 1
      print minx, miny, maxx, maxy
      k = getKey(minx, miny, maxx, maxy)
      if not k in rGrids:
         task = getStartupTask(minx, miny, maxx, maxy, tag, inDir, cDir, oDir, stepSize, ts)
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
         
         if os.path.exists(getSummaryFile(oDir, task.tag)):
            minx, miny, maxx, maxy, l, t, r, b,cc = readOutputs(oDir, task.tag)
            print "Changed", cc, "cells"

            #TODO: Remove from running list
            #k = '%s,%s,%s,%s' % (minx, miny, maxx, maxy)
            k = getKey(minx, miny, maxx, maxy)
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
               nTask = getConnectedTask(minx, miny, maxx, maxy, vs, ss, tag, inDir, cDir, oDir, stepSize, ts)
               if nTask is not None:
                  rGrids.append(k)
                  print "Added", k, "to running list", tag
                  q.submit(nTask)
            
            # Add adjacent tiles as necessary
            if l:
               vect = getVectorFilename(oDir, task.tag, 0)
               
               #myKey = '%s,%s,%s,%s' % (minx-ts, miny, maxx-ts, maxy)
               myKey = getKey(minx-ts, miny, minx, maxy)
               if myKey in rGrids:
                  if not waitingGrids.has_key(myKey):
                     waitingGrids[myKey] = []
                  waitingGrids[myKey].append((2, vect))
                  print "Waiting for:", myKey
               else:
                  print "Submitting for:", myKey
                  tag = currentTag
                  currentTag += 1
                  nTask = getConnectedTask(minx-ts, miny, miny, maxy, [vect], [2], tag, inDir, cDir, oDir, stepSize, ts)
                  if nTask is not None:
                     rGrids.append(myKey)
                     print "Added", myKey, "to running list", tag
                     q.submit(nTask)
            if t:
               vect = getVectorFilename(oDir, task.tag, 1)
               
               #myKey = '%s,%s,%s,%s' % (minx, miny+ts, maxx, maxy+ts)
               myKey = getKey(minx, maxy, maxx, maxy+ts)
               if myKey in rGrids:
                  if not waitingGrids.has_key(myKey):
                     waitingGrids[myKey] = []
                  waitingGrids[myKey].append((3, vect))
                  print "Waiting for:", myKey
               else:
                  print "Submitting for:", myKey
                  tag = currentTag
                  currentTag += 1
                  nTask = getConnectedTask(minx, maxy, maxx, maxy+ts, [vect], [3], tag, inDir, cDir, oDir, stepSize, ts)
                  if nTask is not None:
                     rGrids.append(myKey)
                     print "Added", myKey, "to running list", tag
                     q.submit(nTask)
            if r:
               vect = getVectorFilename(oDir, task.tag, 2)

               #myKey = '%s,%s,%s,%s' % (minx+ts, miny, maxx+ts, maxy)
               myKey = getKey(maxx, miny, maxx+ts, maxy)
               if myKey in rGrids:
                  if not waitingGrids.has_key(myKey):
                     waitingGrids[myKey] = []
                  waitingGrids[myKey].append((0, vect))
                  print "Waiting for:", myKey
               else:
                  print "Submitting for:", myKey
                  tag = currentTag
                  currentTag += 1
                  nTask = getConnectedTask(maxx, miny, maxx+ts, maxy, [vect], [0], tag, inDir, cDir, oDir, stepSize, ts)
                  if nTask is not None:
                     rGrids.append(myKey)
                     print "Added", myKey, "to running list", tag
                     q.submit(nTask)
            if b:
               vect = getVectorFilename(oDir, task.tag, 3)
               #myKey = '%s,%s,%s,%s' % (minx, miny-ts, maxx, maxy-ts)
               myKey = getKey(minx, miny-ts, maxx, miny)
               if myKey in rGrids:
                  if not waitingGrids.has_key(myKey):
                     waitingGrids[myKey] = []
                  waitingGrids[myKey].append((1, vect))
                  print "Waiting for:", myKey
               else:
                  print "Submitting for:", myKey
                  tag = currentTag
                  currentTag += 1
                  nTask = getConnectedTask(minx, miny-ts, maxx, miny, [vect], [1], tag, inDir, cDir, oDir, stepSize, ts)
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
   with open(args.outputFile, 'w') as outF:
      if r >= 1000:
         outF.write("-1\n")
      outF.write('%s\n' % (bTime - aTime))
   
   