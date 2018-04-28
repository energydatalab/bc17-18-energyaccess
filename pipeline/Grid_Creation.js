/*
Generates a grid using given extent in coordinates, specified as geometry.

The Bass Connections team used the output of this & intersected it with the
Uttar Pradesh boundary for making energy access predictions.

The Uttar Pradesh state boundary is available within the dataset from the
2017 Data+ team via Boning Li's GEE Assets, callable via:

var ind = ee.FeatureCollection('users/bl/Ind_admin_shapefiles/ind_states');
var up = ind.filter(ee.Filter.eq('st_name','Uttar Pradesh'))

This scripts output is callable via:

var grid = ee.FeatureCollection('users/xlany/grid_5km_exact')

Script outputs a GeoJSON grid to Google Drive of choice.
 */
var generateGrid = function(xmin, ymin, xmax, ymax, dx, dy, marginx, marginy) {
  var xx = ee.List.sequence(xmin, ee.Number(xmax).subtract(ee.Number(dx).multiply(0.9)), dx)
  var yy = ee.List.sequence(ymin, ee.Number(ymax).subtract(ee.Number(dy).multiply(0.9)), dy)
  
  var cells = xx.map(function(x) {
    return yy.map(function(y) {
      var x1 = ee.Number(x).subtract(marginx)
      var x2 = ee.Number(x).add(ee.Number(dx)).add(marginx)
      var y1 = ee.Number(y).subtract(marginy)
      var y2 = ee.Number(y).add(ee.Number(dy)).add(marginy)
      
      var coords = ee.List([x1, y1, x2, y2]);
      var rect = ee.Algorithms.GeometryConstructors.Rectangle(coords, 'EPSG:4326', false);
    
      return ee.Feature(rect)
    })
  }).flatten();

  return ee.FeatureCollection(cells);
}

// first 4 values are to create a bounding box via coordinates
// 5th and 6th values are for pixel size in coordinates
// note: unknown issues arise when using 0.01 and smaller pixel sizes.
var grid = generateGrid(76, 23, 84, 31, .05, .05, 0, 0)
Map.addLayer(grid)

// make sure each pixel has geographic coordinates.
print(grid.limit(10))

// select location of export
Export.table.toDrive({
  collection: grid,
  description: 'grid_export',
  fileFormat: 'GeoJSON'})