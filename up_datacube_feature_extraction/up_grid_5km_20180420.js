/***
 * Generates a regular grid using given bounds, specified as geometry.
 */
var generateGrid = function(xmin, ymin, xmax, ymax, dx, dy, marginx, marginy) {
  var xx = ee.List.sequence(xmin, ee.Number(xmax).subtract(ee.Number(dx)), dx)
  var yy = ee.List.sequence(ymin, ee.Number(ymax).subtract(ee.Number(dy)), dy)
  
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
// Map.addLayer(ee.Feature(ee.Geometry.Rectangle(x1, y1, x2, y2), {label: cell_id}))
var grid = generateGrid(76, 23, 85, 31, .045, .045, 0, 0)
// Map.addLayer(grid)
// Map.centerObject(up, 8)
Export.table.toDrive({
  collection: grid, 
  description: 'grid_5km_exact', 
  folder: 'Feature_Extraction',
  fileFormat: 'KML'})