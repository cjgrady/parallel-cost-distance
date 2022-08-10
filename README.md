# parallel-cost-distance

[![Coverage Status](https://coveralls.io/repos/github/cjgrady/irksome-broccoli/badge.svg?branch=master)](https://coveralls.io/github/cjgrady/irksome-broccoli?branch=master)

Note: This repository is now archived.  It can serve as a good base for additional experimentation and research but it is no longer being actively maintained.

# Description

  This project is a Python implementation of a parallel cost distance tool.  The current version only works for calculating coastal inundation from sea level rise, but that is a coming expansion.  This first version of the tool works with integer ASCII grids but those are two more expansions that can be easily done.  The tool utilizes region decomposition and a calculate and correct approach to utilize parallelization.  This allows for computations to be performed on much larger data sets with less powerful hardware.
  
# How to use

## Tile Splitter

  If you have a large surface that you want to split into tiles, you can use the tile splitter tool to do so.  This tool splits a large raster into multiple tiles that use a naming convention that the multi-tile tool can use for computations.  
  
### Usage
 
    $ python slr/tools/tileSplitter.py --help
    usage: tileSplitter.py [-h] [-x XOFFSET] [-y YOFFSET] [-d {0,1}]
                           fn tileSize outDir
    
    positional arguments:
      fn                    Path to the raster file to split
      tileSize              What size tiles to split the raster into
      outDir                Directory where the tiles should be written
    
    optional arguments:
      -h, --help            show this help message and exit
      -x XOFFSET, --xOffset XOFFSET
                            Offset the grid this many cells in the x direction
      -y YOFFSET, --yOffset YOFFSET
                            Offset the grid this many cells in the y direction
      -d {0,1}, --debug {0,1}
                            Enable debugging (1 for yes, 0 for no)

## Single Tile Parallel Computations

  If you have a single raster that is a reasonable size, you can use the parallelDijkstra.py tool.  This tool uses threads for parallelism but, conceptually, the method is the same as it is for multiple tiles.
  
### Usage

    $ python slr/singleTile/parallelDijkstra.py --help
    usage: parallelDijkstra.py [-h] [-g G] [-t TASKID] [-v [VECT [VECT ...]]]
                               [-s [FROMSIDE [FROMSIDE ...]]] [-o O] [-w W]
                               [--step STEP] [--ts TS] [-e E]
                               dem costSurface
    
    positional arguments:
      dem
      costSurface
    
    optional arguments:
      -h, --help            show this help message and exit
      -g G                  Generate edge vectors for modified cells
      -t TASKID, --taskId TASKID
                            Use this task id for outputs
      -v [VECT [VECT ...]], --vect [VECT [VECT ...]]
                            Use this vector for source cells
      -s [FROMSIDE [FROMSIDE ...]], --fromSide [FROMSIDE [FROMSIDE ...]]
                            Source vector is from this side 0: left, 1: top, 2:
                            right, 3: bottom
      -o O                  File to write outputs
      -w W                  Maximum number of worker threads
      --step STEP           The step size to use
      --ts TS
      -e E                  Log errors to this file location
 
## Multiple Tile Parallel Computations

  For large raster files that have been split into tiles (or multiple volumes split into tiles) you can use the multi-tile tool.  The current implementation relies on a specific naming convention for tile that are all located in a single directory.  For this reason, it is recommended that these tiles are created by utilizing the tile splitter tool.  The tool uses Work Queue in the CCTools package (http://ccl.cse.nd.edu/software/) from the Cooperative Computing Lab at Notre Dame to manage parallel computations.
  
### Usage

    $ python slr/multiTile/wqMultiTile.py --help
    usage: wqMultiTile.py [-h] inputDir cDir oDir tileSize stepSize outputFile
    
    positional arguments:
      inputDir
      cDir
      oDir
      tileSize
      stepSize
      outputFile
    
    optional arguments:
      -h, --help  show this help message and exit
     
