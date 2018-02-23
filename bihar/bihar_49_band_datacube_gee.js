/* makeIrriElecMetrics.js

This script works for the Google Earth Engine (Online Code Editor). It integrates 11-band Landsat 8 imagery
(greenest cloud free), 12-month Normalized Different Vegetation Index, 12-month Green Index, 12-month
Rainfall data, and annual min VIIRS Nighttime Lights Imagery into a multi-band geotiff file
(i.e. a composite of 11+ 12 + 12 + 12 + 1 = 48 bands) which represents the irrigation-electrification
metrics for the region of interest.

NOTE: The estimated size of the entire data is ~40 GB; if your Google Drive does not have enough storage,
please split the task.

Authors: Gouttham Chandrasekar, Boning Li
Email: boning.li@duke.edu
Duke Data+ 2017: Electricity Access
Jul 30, 2017

*/

// Set the region of interest
// Asset available at https://code.earthengine.google.com/?asset=users/bl/Ind_admin_shapefiles/ind_states
var ind = ee.FeatureCollection('users/bl/Ind_admin_shapefiles/ind_states');
var br = ind.filter(ee.Filter.eq('st_name','Bihar'));
var roi = br;
var irr_ag = ee.Image("users/bawong/bassConnections/giam_2014_2015_v2")

//=================== First 11 bands to be greenest cloud free landsat imagery ===================
// cloud mask code from: https://developers.google.com/earth-engine/ic_composite_mosaic
// This function masks clouds in Landsat 8 imagery.
var maskClouds = function(image) {
  var scored = ee.Algorithms.Landsat.simpleCloudScore(image);
  return image.updateMask(scored.select(['cloud']).lt(20));
};

// This function masks clouds and adds quality bands to Landsat 8 images.
var addQualityBands = function(image) {
  return maskClouds(image)
    // NDVI
    .addBands(image.normalizedDifference(['B5', 'B4']))
};

// Load a 2014 Landsat 8 ImageCollection.
// Map the cloud masking and quality band function over the collection.
var collection = ee.ImageCollection('LANDSAT/LC8_L1T_TOA')
  .filterDate('2016-01-01', '2016-12-31').filterBounds(roi)
  .map(addQualityBands);

// Create a greenest pixel composite.
var greenestCloudfreeComposite = collection.qualityMosaic('nd').clip(roi)
	.select(['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11'])



//=================== Initialization: ndvi / green index ===================
// landsat 8 imagery collection temporally bounded by month, spatially bounded by roi
var ls8_base = ee.ImageCollection('LANDSAT/LC8_L1T_TOA').filterBounds(roi);
var ls8 = ls8_base.filterDate('2016-01-01', '2016-01-31');
// reduce the collection to a single image by min (to minimize cloud coverage) values of each pixel
var img_temp = ls8.reduce(ee.Reducer.min()).clip(roi);
var img = img_temp.select(['B1_min','B2_min','B3_min','B4_min','B5_min','B6_min','B7_min','B8_min','B9_min','B10_min','B11_min'],
	 					  ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11']);
// ndvi base, namely its values for jan. Other months' data will be appended later in a loop
var ndvi_final = img.normalizedDifference(['B5', 'B4']).float().rename('01NDVI');
// green index base, namely its values for jan. Other months' data will be appended later in a loop
var green_final = img.select('B5').divide(img.select('B3')).float().rename('01GREEN');

//=================== Initialization: rainfall data ===================
// rainfall data collection temporally bounded by month, spatially bounded by roi
var GPM_raw = ee.ImageCollection('NASA/GPM_L3/IMERG_V04').filterBounds(roi);
var GPM = GPM_raw.filterDate('2016-01-01', '2016-01-31');
// sum them up for the monthly rainfall
var img_temp = GPM.reduce(ee.Reducer.sum()).clip(roi);
var img = img_temp.select(['IRprecipitation_sum'],['IRprecipitation_double']);
// rainfall data base, namely its values for jan. Other months' data will be appended later in a loop
var rain_final = img.select('IRprecipitation_double').divide(2).float().rename('01RAIN');

// add other months' ndvi, green index, and rainfall data
for (var i =2; i<=12; i ++) {
	// get it for the year of 2016, and construct a string for the start and end day of the particular month
	var year = 2016
	var dayEnd  = new Date(year, i, 0).getDate();
	var monthString;
	if (i<10){ monthString = '0'+i} else {monthString = i}
	var date_start = year+'-'+monthString+'-01';
	var date_end = year+'-'+monthString+'-'+dayEnd;

	//=================== Data for the month: ndvi/green index ===================
	var ls8 = ls8_base.filterDate(date_start, date_end);
	var img_temp = ls8.reduce(ee.Reducer.min()).clip(roi);
	var img = img_temp.select(['B1_min','B2_min','B3_min','B4_min','B5_min','B6_min','B7_min','B8_min','B9_min','B10_min','B11_min'],
	 						['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11']);
	var ndvi = img.normalizedDifference(['B5', 'B4']).float().rename(monthString+'NDVI');// print('ndvi',ndvi)
	var green = img.select('B5').divide(img.select('B3')).float().rename(monthString+'GREEN');

	//=================== Data for the month: rainfall data ===================
	var GPM = GPM_raw.filterDate(date_start, date_end);
	var img_temp = GPM.reduce(ee.Reducer.sum()).clip(roi);
	var img = img_temp.select(['IRprecipitation_sum'],['IRprecipitation_double']);
	var rain = img.select('IRprecipitation_double').divide(2).float().rename(monthString+'RAIN');

	//=================== Composing (append to each base) ===================
	ndvi_final = ndvi_final.addBands(ndvi)
	green_final = green_final.addBands(green)
  rain_final = rain_final.addBands(rain)

}
//=================== Add annual viirs data (stray light corrected) ===================
var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
	.filterDate('2016-01-01', '2016-12-30') // annual composite
	.reduce(ee.Reducer.median()) // reduce by median
	.clip(roi).select('avg_rad_median').float().rename('VIIRS2016');

//=================== Add irriaged ag band ===================
var irr_band = irr_ag.clip(roi).rename('IRR_AG')

// add all bands together. Order-> 1:12=ndvi, 13:24=green index, 25:36=rainfall, 37=viirs
var all_final = greenestCloudfreeComposite.addBands(ndvi_final).addBands(green_final).addBands(rain_final).addBands(viirs).addBands(irr_band)

// print final information
print(all_final,'49 bands information')

// Export data to drive
Export.image.toDrive({
    image: all_final,
    description: 'final_49',
    scale: 30,
    region: roi,
    maxPixels: 1E13
});

// If you want to visualize certain bands
Map.addLayer(all_final.select('03NDVI'), {"opacity":1,"bands":["03NDVI"],"palette":["ffffff","ff6b6b","d61c09"]}, '49-band Image')
Map.addLayer(all_final.select('IRR_AG'), {"opacity":0.9,"bands":["IRR_AG"],"max":1,"palette":["ffffff","47d001"]}, 'Irrigated Agriculture Band')
Map.centerObject(roi, 8)
Map.setOptions('SATELLITE')