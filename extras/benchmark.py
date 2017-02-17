"""
@summary: This script performs benchmark tests
"""
import argparse
import os
import sys
from work_queue import *

import slr.singleTile.parallelDijkstra

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
   return os.path.abspath(slr.singleTile.parallelDijkstra.__file__)
# ................................................................................................
if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   
   parser.add_argument('testSurface', type=str, 
            help="File path to the surface to use for benchmarking")
   parser.add_argument('outDir', type=str,
            help="Path to directory to write output files")
   parser.add_argument('numRuns', type=int,
            help="The number of times to run the benchmark")
   parser.add_argument('benchmarkDir', type=str,
            help="Directory to store benchmark files")
            
   args = parser.parse_args()
   
   port = WORK_QUEUE_DEFAULT_PORT
   print "Port:" , port
     
   cctools_debug_flags_set("all")
   cctools_debug_config_file("/tmp/myWQ.log")
   q = WorkQueue(port=port)
   #q.enable_monitoring_full(args.benchmarkDir)
            
   for i in range(args.numRuns):
      task = Task('')
      benchmarkFile = os.path.join(args.benchmarkDir, "{0}.txt".format(i))
      costGrid = os.path.join(args.outDir, "{0}.asc".format(i))
      tag = str(i)
      
      cmd = "{python} {pycmd} {inGrid} {costGrid} -t {taskId} -b {b} -e {e}".format(
            python=PYTHON_BIN,
            pycmd=getParallelDijkstraModulePath(),
            inGrid=args.testSurface,
            costGrid=costGrid,
            taskId=tag,
            b=benchmarkFile,
            e=os.path.join("/home/cjgrady/errors", "{0}.txt".format(i)))
      task.specify_command(cmd)
      task.specify_output_file(benchmarkFile)
      task.specify_output_file(costGrid)
      task.specify_tag(tag)
   
      q.submit(task)
 
   cpuTime = {}
   
   while not q.empty():
      task = q.wait(1)
      if task:
            
         print "Task id:", task.id
         print "Task tag:", task.tag
         mc = task.cmd_execution_time
         
         #stats.append((task.id, rc, wc, mc, psum))
         cpuTime[int(task.tag)] = mc
         #stats.append((task.id, task.tag))#, task.resources_measured.memory, 
         #              task.resources_measured.virtual_memory, 
         #              task.resources_measured.cpu_time))
         # 
   
   with open(os.path.join(args.benchmarkDir, 'benchmarks.csv'), 'w') as outF:
      outF.write('"Task id", "Rc", "Wc", "Psum", "Tt", "CPU time"\n')
      for i in range(args.numRuns):
         bFn = os.path.join(args.benchmarkDir, "{0}.txt".format(i))
         with open(bFn) as bmF:
            line = bmF.read().strip()
            rc, wc, psum, tt = line.split(', ')
            outF.write("{0}, {1}, {2}, {3}, {4}, {5}\n".format(i, rc, wc, psum, 
                                                             tt, cpuTime[i]))
