# irksome-broccoli

[![Build Status](https://travis-ci.org/cjgrady/irksome-broccoli.svg?branch=master)](https://travis-ci.org/cjgrady/irksome-broccoli)

[![Coverage Status](https://coveralls.io/repos/github/cjgrady/irksome-broccoli/badge.svg?branch=master)](https://coveralls.io/github/cjgrady/irksome-broccoli?branch=master)

Note: The build says that it is failing but there seems to be an issue deploying CC Tools on Travis CI.  I believe that it is an issue with the tests trying to utilize more cores / resources than are available with Travis.  

# Description

  This project is a Python implementation of a parallel cost distance tool.  The current version only works for calculating coastal inundation from sea level rise, but that is a coming expansion.  This first version of the tool works with integer ASCII grids but those are two more expansions that can be easily done.  The tool utilizes region decomposition and a calculate and correct approach to utilize parallelization.  This allows for computations to be performed on much larger data sets with less powerful hardware.
  
# How to use

## Tile Splitter

  If you have a large surface that you want to split into tiles, you can use the tile splitter tool to do so.  This tool splits a large raster into multiple tiles that use a naming convention that the multi-tile tool can use for computations.  
  
### Usage
 
    asdfasd

## Single Tile Parallel Computations

  If you have a single raster that is a reasonable size, you can use the parallelDijkstra.py tool.  This tool uses threads for parallelism but, conceptually, the method is the same as it is for multiple tiles.
  
### Usage

 
     asdf

## Multiple Tile Parallel Computations

  For large raster files that have been split into tiles (or multiple volumes split into tiles) you can use the multi-tile tool.  The current implementation relies on a specific naming convention for tile that are all located in a single directory.  For this reason, it is recommended that these tiles are created by utilizing the tile splitter tool.  The tool uses Work Queue in the CCTools package (http://ccl.cse.nd.edu/software/) from the Cooperative Computing Lab at Notre Dame to manage parallel computations.
  
### Usage


     aasdf
     
