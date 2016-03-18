"""
@summary: Runs using Work Queue over multiple tiles
"""
import glob
import os
import time
from work_queue import *

OUTPUTS_PATH = "/home/cjgrady/thesis/outputs/"
#INPUT_PATH = "/home/cjgrady/thesis/degreeGrids/"
INPUT_PATH = "/home/cjgrady/thesis/flTest/"
COSTS_PATH = "/home/cjgrady/thesis/outputs/"

# .............................................................................
def getInputGridFilename(minx, miny, maxx, maxy):
   return os.path.join(INPUT_PATH, "grid%s-%s-%s-%s.asc" % (minx, miny, maxx, maxy))

# .............................................................................
def getCostGridFilename(minx, miny, maxx, maxy):
   return os.path.join(COSTS_PATH, "grid%s-%s-%s-%s.asc" % (minx, miny, maxx, maxy))

# .............................................................................
def getVectorFilename(taskId, d):
   dirPart = ['toLeft', 'toTop', 'toRight', 'toBottom'][d]
   
   return os.path.join(OUTPUTS_PATH, "%s-%s.npy" % (taskId, dirPart))
   
# .............................................................................
def getSummaryFile(taskId):
   return os.path.join(OUTPUTS_PATH, "%s-summary.txt" % taskId)

# .............................................................................
def readOutputs(taskId):
   cnt = open(getSummaryFile(task.tag)).readlines()
   minx = int(cnt[0])
   miny = int(cnt[1])
   maxx = int(cnt[2])
   maxy = int(cnt[3])
   l = cnt[4].lower().strip() == 'true'
   t = cnt[5].lower().strip() == 'true'
   r = cnt[6].lower().strip() == 'true'
   b = cnt[7].lower().strip() == 'true'
   cc = int(cnt[8])
   
   return minx, miny, maxx, maxy, l, t, r, b, cc
   
# .............................................................................
def getStartupTask(minx, miny, maxx, maxy, tag):
   task = Task('')
   cmd = "{python} {pycmd} {inGrid} {costGrid} -g 1 -o {outputsPath} -w 50 -t {taskId} --step=150".format(
            python='python',
            pycmd='~/git/irksome-broccoli/src/singleTile/parallelDijkstra.py',
            inGrid=getInputGridFilename(minx, miny, maxx, maxy),
            costGrid=getCostGridFilename(minx, miny, maxx, maxy),
            outputsPath=OUTPUTS_PATH, taskId=tag)
   task.specify_command(cmd)
   task.specify_output_file(getSummaryFile(tag))
   task.specify_tag(str(tag))
   return task

# .............................................................................
def getConnectedTask(minx, miny, maxx, maxy, vects, fromSides, tag):
   inGrid = getInputGridFilename(minx, miny, maxx, maxy)
   
   if os.path.exists(inGrid):
      task = Task('')
      
      vectsSec = ' '.join(['-v %s' % v for v in vects])
      sidesSec = ' '.join(['-s %s' % s for s in fromSides])
      
      cmd = "{python} {pycmd} {inGrid} {costGrid} -g 1 -o {outputsPath} -w 50 -t {taskId} --step=150 {vectsSec} {sidesSec}".format(
            python='python',
            pycmd='~/git/irksome-broccoli/src/singleTile/parallelDijkstra.py',
            inGrid=getInputGridFilename(minx, miny, maxx, maxy),
            costGrid=getCostGridFilename(minx, miny, maxx, maxy),
            outputsPath=OUTPUTS_PATH, 
            taskId=tag,
            vectsSec=vectsSec,
            sidesSec=sidesSec)
      task.specify_command(cmd)
      task.specify_output_file(getSummaryFile(tag))
      task.specify_tag(str(tag))
      return task
   else:
      return None


# .............................................................................
if __name__ == "__main__":

   aTime = time.time()
   currentTag = 1
   #inputGrids = [getInputGridFilename(-80, 33, -79, 34)]
   inputGrids = glob.glob(os.path.join(INPUT_PATH, "*.asc"))
   
   port = WORK_QUEUE_DEFAULT_PORT
   print "Port:" , port
   
   rGrids = []
   waitingGrids = {}
   
   q = WorkQueue(port)
   
   for g in inputGrids:
      task = Task('')
      minx = '-%s' % g.split('-')[1]
      miny = '%s' % g.split('-')[2]
      maxx = '-%s' % g.split('-')[4]
      maxy = '%s' % g.split('-')[5].split('.')[0]
      tag = currentTag
      currentTag += 1
      task = getStartupTask(minx, miny, maxx, maxy, tag)
      print minx, miny, maxx, maxy
      k = '%s,%s,%s,%s' % (minx, miny, maxx, maxy)
      rGrids.append(k)
      print "Added", k, "to running list", tag
      
      print "Submitting task:", task.tag
      q.submit(task)

   while not q.empty():
      # Wait a maximum of 10 seconds for a task to come back.
      task = q.wait(1)
      if task:
         print "Task id:", task.id
         print "Task tag:", task.tag
         
         if os.path.exists(getSummaryFile(task.tag)):
            minx, miny, maxx, maxy, l, t, r, b,cc = readOutputs(task.tag)
            print "Changed", cc, "cells"

            #TODO: Remove from running list
            k = '%s,%s,%s,%s' % (minx, miny, maxx, maxy)
            print "Removing", k, "from running list", task.tag
            try:
               rGrids.remove(k)
            except:
               print rGrids
               raise
         
            
            # Add any tasks that were waiting on this tile to finish
            tKey = '%s,%s,%s,%s' % (minx, miny, maxx, maxy)
            if waitingGrids.has_key(tKey):
               sides = waitingGrids.pop(tKey)
               ss,vs = zip(*sides)
               tag = currentTag
               currentTag += 1
               nTask = getConnectedTask(minx, miny, maxx, maxy, vs, ss, tag)
               if nTask is not None:
                  rGrids.append(tKey)
                  print "Added", tKey, "to running list", tag
                  q.submit(nTask)
            
            # Add adjacent tiles as necessary
            if l:
               vect = getVectorFilename(task.tag, 0)
               
               myKey = '%s,%s,%s,%s' % (minx-1, miny, maxx-1, maxy)
               if myKey in rGrids:
                  if not waitingGrids.has_key(myKey):
                     waitingGrids[myKey] = []
                  waitingGrids[myKey].append((2, vect))
                  print "Waiting for:", myKey
               else:
                  print "Submitting for:", myKey
                  tag = currentTag
                  currentTag += 1
                  nTask = getConnectedTask(minx-1, miny, maxx-1, maxy, [vect], [2], tag)
                  if nTask is not None:
                     rGrids.append(myKey)
                     print "Added", myKey, "to running list", tag
                     q.submit(nTask)
            if t:
               vect = getVectorFilename(task.tag, 1)
               
               myKey = '%s,%s,%s,%s' % (minx, miny+1, maxx, maxy+1)
               if myKey in rGrids:
                  if not waitingGrids.has_key(myKey):
                     waitingGrids[myKey] = []
                  waitingGrids[myKey].append((3, vect))
                  print "Waiting for:", myKey
               else:
                  print "Submitting for:", myKey
                  tag = currentTag
                  currentTag += 1
                  nTask = getConnectedTask(minx, miny+1, maxx, maxy+1, [vect], [3], tag)
                  if nTask is not None:
                     rGrids.append(myKey)
                     print "Added", myKey, "to running list", tag
                     q.submit(nTask)
            if r:
               vect = getVectorFilename(task.tag, 2)

               myKey = '%s,%s,%s,%s' % (minx+1, miny, maxx+1, maxy)
               if myKey in rGrids:
                  if not waitingGrids.has_key(myKey):
                     waitingGrids[myKey] = []
                  waitingGrids[myKey].append((0, vect))
                  print "Waiting for:", myKey
               else:
                  print "Submitting for:", myKey
                  tag = currentTag
                  currentTag += 1
                  nTask = getConnectedTask(minx+1, miny, maxx+1, maxy, [vect], [0], tag)
                  if nTask is not None:
                     rGrids.append(myKey)
                     print "Added", myKey, "to running list", tag
                     q.submit(nTask)
            if b:
               vect = getVectorFilename(task.tag, 3)
               myKey = '%s,%s,%s,%s' % (minx, miny-1, maxx, maxy-1)
               if myKey in rGrids:
                  if not waitingGrids.has_key(myKey):
                     waitingGrids[myKey] = []
                  waitingGrids[myKey].append((1, vect))
                  print "Waiting for:", myKey
               else:
                  print "Submitting for:", myKey
                  tag = currentTag
                  currentTag += 1
                  nTask = getConnectedTask(minx, miny-1, maxx, maxy-1, [vect], [1], tag)
                  if nTask is not None:
                     rGrids.append(myKey)
                     print "Added", myKey, "to running list", tag
                     q.submit(nTask)
         else:
            print task.id
            print task.command
            print task.output
            print dir(task)
   bTime = time.time()
   
   print bTime - aTime