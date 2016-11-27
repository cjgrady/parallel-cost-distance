"""
@summary: Calculates Moran's I for a grid
"""

ROOK =  [
     [0, 1, 0],
     [1, 0, 1],
     [0, 1, 0]
    ]

#from math import sqrt
#c2 = [
#      [1/sqrt(2), 1, 1/sqrt(2)],
#      [1, 0, 1],
#      [1/sqrt(2), 1, 1/sqrt(2)],
#     ]
#
#       ]
#

def moransI(xMat, c=ROOK):
   """
   @param xMat: Matrix of x values
   @param c: Connectivity matrix weights
   
     rook -  0 1 0
             1 0 1
             0 1 0
   
     queen - 1/sqrt(2) 1 1/sqrt(2)
             1         0 1
             1/sqrt(2) 1 1/sqrt(2)
   """
   rows = len(xMat)
   cols = len(xMat[0])
   n = rows * cols
   cyWeights = sum(sum(r) for r in c[:len(c)/2])
   cxWeights = sum(sum(r[:len(c[0])/2]) for r in c)
   # wSum = number of cells * number of connections per cell minus cells only 
   #          partially connected
   wSum = n * sum(sum(r) for r in c) - (2 * cols * cyWeights) \
              - (2 * rows * cxWeights)
   sum1 = 0
   sum2 = 0
   tot = sum(sum(r) for r in xMat)
   xBar = 1.0 * tot / n
   
   cyMid = len(c)/2
   cxMid = len(c[0])/2
   
   for row in xrange(rows):
      for col in xrange(cols):
         xiDiff = xMat[row][col] - xBar
         sum2 = sum2 + xiDiff**2
         for j in xrange(len(c)):
            for i in xrange(len(c[j])):
               partial = 0
               try:
                  y = row - cyMid + j
                  x = col - cxMid + i
                  xjDiff = xMat[y][x] - xBar if y >= 0 and x >= 0 and \
                                                y < rows and x < cols else 0
                  partial = c[j][i] * xiDiff * xjDiff
               except:
                  pass
               sum1 = sum1 + partial
   if (wSum * sum2) == 0:
      return 0.0
   else:
      i = (n * sum1) / (wSum * sum2)
   return i
