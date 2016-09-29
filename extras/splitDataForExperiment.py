"""
@summary: This script splits all of the NGDC data so that it can be used for
             experiments
@deprecated: I just don't want to lose this code yet
"""
# .............................................................................
if __name__ == "__main__":
   
   # TODO: Move these to another script, use argparse to parameterize here
   
   # Split into 1 degree by 1 degree chunks
   # First, verify that tiles line up
#    parser = argparse.ArgumentParser()
#    parser.add_argument('-x', '--xOffset', type=int, 
#                      help="Offset the grid this many cells in the x direction")
#    parser.add_argument('-y', '--yOffset', type=int, 
#                      help="Offset the grid this many cells in the y direction")
#    parser.add_argument('-lat', '--startingLatitude', type=int,
#                   help="Integer latitude of the top left corner of the raster")
#    parser.add_argument('-long', '--startingLongitude', type=int,
#                   help="Integer longitude of the top left corner of the raster")
   
   # Volume 1
   #fn = '/home/cjgrady/thesis/ne_atl_crm_v1.asc'
   #readInputData(fn, xOffset=1, yOffset=1, lat=39, lon=-78, xll=-78, yll=39)

   # Volume 2
   fn = '/home/cjgrady/thesis/se_atl_crm_v1.asc'
   #readInputData(fn, xOffset=1, yOffset=1, lat=30, lon=-83, xll=-83, yll=30,
   #              xCells=int(.5*XCELLS), yCells=int(.5*YCELLS), outDir='/home/cjgrady/thesis/.5degree/se/', ts=0.5)
   readInputData(fn, xOffset=1, yOffset=1, lat=30, lon=-83, xll=-83, yll=30,
                 xCells=XCELLS, yCells=YCELLS, outDir='/home/cjgrady/thesis/1degree/se/', ts=1.0)
   readInputData(fn, xOffset=1, yOffset=1, lat=30, lon=-83, xll=-83, yll=30,
                 xCells=2*XCELLS, yCells=2*YCELLS, outDir='/home/cjgrady/thesis/2degree/se/', ts=2.0)
   
   # Volume 3
   #fn = '/home/cjgrady/thesis/fl_east_gom_crm_v1.asc'
   #readInputData(fn, xOffset=1, yOffset=1, lat=24, lon=-88, xll=-88, yll=24,
   #              xCells=int(.5*XCELLS), yCells=int(.5*YCELLS), outDir='/home/cjgrady/thesis/.5degree/fl/', ts=0.5)
   
   # Volume 4
   #fn = '/home/cjgrady/thesis/central_gom_crm_v1.asc'
   #readInputData(fn, xOffset=1, yOffset=1, lat=24, lon=-95, xll=-95, yll=24)
   
   # Volume 5
   #fn = '/home/cjgrady/thesis/western_gom_crm_v1.asc'
   #readInputData(fn, xOffset=1, yOffset=1, lat=25, lon=-100, xll=-100, yll=25)
   
   # Volume 6 - Already 1 degree tiles
   
   # Volume 7
   #fn = '/home/cjgrady/thesis/central_pacific_crm_v1.asc'
   #readInputData(fn, xOffset=1, yOffset=1, lat=36, lon=-127, xll=-127, yll=36)
   
   # Volume 8
   #fn = '/home/cjgrady/thesis/nw_pacific_crm_v1.asc'
   #readInputData(fn, xOffset=1, yOffset=1, lat=43, lon=-127, xll=-127, yll=43)
   
   # Volume 9
   fn = '/home/cjgrady/thesis/puerto_rico_crm_v1.asc'
   readInputData(fn, xOffset=1, yOffset=1, lat=17, lon=-68, xll=-68, yll=17,
                 xCells=int(.5*XCELLS), yCells=int(.5*YCELLS), outDir='/home/cjgrady/thesis/.5degree/pr/', ts=0.5)
   readInputData(fn, xOffset=1, yOffset=1, lat=17, lon=-68, xll=-68, yll=17,
                 xCells=XCELLS, yCells=YCELLS, outDir='/home/cjgrady/thesis/1degree/pr/', ts=1.0)
   readInputData(fn, xOffset=1, yOffset=1, lat=17, lon=-68, xll=-68, yll=17,
                 xCells=2*XCELLS, yCells=2*YCELLS, outDir='/home/cjgrady/thesis/2degree/pr/', ts=2.0)
   
   # Volume 10
   #fn = '/home/cjgrady/thesis/hawaii_crm_v1.asc'
   #readInputData(fn, xOffset=1, yOffset=1, lat=18, lon=-161, xll=-161, yll=18)
   
