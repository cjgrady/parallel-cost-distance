"""
"""

from work_queue import *

if __name__ == "__main__":
   port = WORK_QUEUE_DEFAULT_PORT
   
   q = WorkQueue(port)
   
   # Tell all cells to find source cells
   # While the queue is not empty
   #   process outputs