var ind = ee.FeatureCollection('users/bl/Ind_admin_shapefiles/ind_states');
var br = ind.filter(ee.Filter.eq('st_name','Bihar'));
var roi = br;
var irr_ag = ee.Image("users/bawong/bassConnections/giam_2014_2015_v2").clip(roi)  // 0 = non-irr ag, 1 = irr ag
var l8sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(roi)

var hbase = ee.ImageCollection
  .fromImages([
    ee.Image("users/bawong/bassConnections/44R_hbase_probability_30m_bihar_1"), 
    ee.Image("users/bawong/bassConnections/45R_hbase_probability_30m_bihar_2")])
    .mosaic()
    .clip(roi)
    .select('b1')
    .gte(2) // global threshold for LAN
    .rename('HBASE_PROB')

var landscan = ee.Image("users/bawong/bassConnections/bihar_landscan_pop_density_2016")
  .clip(roi)
  .select('b1')
  .rename('POP_DENS')

// FUNCTIONS //
// Function to cloud mask from the Fmask band of Landsat 8 SR data.
function maskL8sr(image) {
  // Bits 3 and 5 are cloud shadow and cloud, respectively.
  var cloudShadowBitMask = ee.Number(2).pow(3).int();
  var cloudsBitMask = ee.Number(2).pow(5).int();
  // Get the pixel QA band.
  var qa = image.select('pixel_qa');
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
      .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
  // Return the masked image, scaled to [0, 1].
  return image.updateMask(mask).divide(10000);
}

function getNdvi (image) {
  var ndvi = image.addBands(image.normalizedDifference(["B5", "B4"]))
  return ndvi
}

// CREATE BASELINE ANNUAL COMPOSITE - GREENEST COMPOSITE METHOD
var l8_sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
  .filterDate('2016-01-01', '2016-12-31')
  .filterBounds(roi)
  .map(maskL8sr)
  .map(getNdvi)
  
var greenestCloudfreeComposite = l8_sr.qualityMosaic('nd').clip(roi).select(['B1','B2','B3','B4','B5','B6','B7','B10','B11'])


//=================== Initialization: ndvi / green index (FOR JANUARY ONLY) before loop =================== 
// landsat 8 imagery collection temporally bounded by month, spatially bounded by roi
var ls8_base = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(roi);
var ls8_mo_comp = ls8_base
  .filterDate('2016-01-01', '2016-01-31')
  .map(maskL8sr)
  .median()
  .select(['B1','B2','B3','B4','B5','B6','B7','B10','B11'])
var ndvi_final = ls8_mo_comp.normalizedDifference(['B5', 'B4']).float().rename('01NDVI')
var green_final = ls8_mo_comp.select('B5').divide(ls8_mo_comp.select('B3')).float().rename('01GREEN')

//=================== Initialization: rainfall data (FOR JANUARY ONLY) before loop ===================
// rainfall data collection temporally bounded by month, spatially bounded by roi
var GPM_raw = ee.ImageCollection('NASA/GPM_L3/IMERG_V04').filterBounds(roi);
var GPM = GPM_raw.filterDate('2016-01-01', '2016-01-31').select('IRprecipitation', 'precipitationCal');
// sum them up for the monthly rainfall
var img_temp = GPM.reduce(ee.Reducer.sum()).clip(roi);
var img = img_temp.select(['IRprecipitation_sum'],['IRprecipitation_double']);
// rainfall data base, namely its values for jan. Other months' data will be appended later in a loop
var rain_final = img.select('IRprecipitation_double').divide(2).float().rename('01RAIN');
//=================== Initialization: VIIRS  (FOR JANUARY ONLY) before loop ===================
var viirs_raw = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').filterBounds(roi);
var viirs_final = viirs_raw.filterDate('2016-01-01', '2016-01-31')
	.reduce(ee.Reducer.median()) // monthly composite pre-made // just a trick to reduce to image
	.clip(roi)
	.mask(hbase)
	.select('avg_rad_median').float().rename('01VIIRS');

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
	var ls8_mo_comp = ls8_base
    .filterDate(date_start, date_end)
    .map(maskL8sr)
    .median()
    .select(['B1','B2','B3','B4','B5','B6','B7','B10','B11'])
    .clip(roi)
  var ndvi = ls8_mo_comp.normalizedDifference(['B5', 'B4']).float().rename(monthString+'NDVI').clip(roi)
  var green = ls8_mo_comp.select('B5').divide(ls8_mo_comp.select('B3')).float().rename(monthString+'GREEN').clip(roi)

	//=================== Data for the month: rainfall data ===================
	var GPM = GPM_raw.filterDate(date_start, date_end).select('IRprecipitation', 'precipitationCal');
	var img_temp = GPM.reduce(ee.Reducer.sum()).clip(roi);
	var img = img_temp.select(['IRprecipitation_sum'],['IRprecipitation_double']);
	var rain = img.select('IRprecipitation_double').divide(2).float().rename(monthString+'RAIN');
	
	//=================== ADD MONTHLY VIIRS===================
	var viirs = viirs_raw
	.filterDate(date_start, date_end)
	.reduce(ee.Reducer.median()) // monthly composite pre-made // just a trick to reduce to image
	.clip(roi)
	.mask(hbase)
	.select('avg_rad_median').float().rename(monthString+'VIIRS');

	//=================== Composing (append to each base) ===================
	ndvi_final = ndvi_final.addBands(ndvi)
	green_final = green_final.addBands(green)
  rain_final = rain_final.addBands(rain)
  viirs_final = viirs_final.addBands(viirs)

}

// make 57 band image by adding them all together
var all_final = greenestCloudfreeComposite.addBands(ndvi_final).addBands(green_final).addBands(rain_final).addBands(viirs_final)
print('57-band image:', all_final)

// visualization for check
// var band = '05NDVI'
// Map.addLayer(all_final.select(band), {"opacity":1,"bands":band,"palette":["ffffff","ff6b6b","d61c09"]}, '57-band Image')
// Map.centerObject(roi, 8)
// Map.setOptions('SATELLITE')


//=================== COLLAPSING NDVI AND GREEN INDEX BANDS TO REDUCE NODATA ISSUE ===================

var ndvi_q1 = all_final.select('01NDVI', '02NDVI', '03NDVI').reduce(ee.Reducer.mean()).rename('01_03_NDVI').mask(irr_ag)
var ndvi_q2 = all_final.select('04NDVI', '05NDVI', '06NDVI').reduce(ee.Reducer.mean()).rename('04_06_NDVI').mask(irr_ag)
var ndvi_q3 = all_final.select('07NDVI', '08NDVI', '09NDVI').reduce(ee.Reducer.mean()).rename('07_09_NDVI').mask(irr_ag)
var ndvi_q4 = all_final.select('10NDVI', '11NDVI', '12NDVI').reduce(ee.Reducer.mean()).rename('10_12_NDVI').mask(irr_ag)

var green_q1 = all_final.select('01GREEN', '02GREEN', '03GREEN').reduce(ee.Reducer.mean()).rename('01_03_GREEN').mask(irr_ag)
var green_q2 = all_final.select('04GREEN', '05GREEN', '06GREEN').reduce(ee.Reducer.mean()).rename('04_06_GREEN').mask(irr_ag)
var green_q3 = all_final.select('07GREEN', '08GREEN', '09GREEN').reduce(ee.Reducer.mean()).rename('07_09_GREEN').mask(irr_ag)
var green_q4 = all_final.select('10GREEN', '11GREEN', '12GREEN').reduce(ee.Reducer.mean()).rename('10_12_GREEN').mask(irr_ag)

//=================== RE-MERGE THE FINAL IMAGE-CUBE ===================

var all_final = all_final.select(['B1','B2','B3','B4','B5','B6','B7','B10','B11', '01RAIN', '02RAIN', '03RAIN', 
  '04RAIN', '05RAIN', '06RAIN', '07RAIN', '08RAIN', '09RAIN', '10RAIN', '11RAIN', '12RAIN', '01VIIRS', '02VIIRS',
  '03VIIRS', '04VIIRS', '05VIIRS', '06VIIRS', '07VIIRS', '08VIIRS', '09VIIRS', '10VIIRS', '11VIIRS', '12VIIRS'])
  .addBands(ndvi_q1).addBands(ndvi_q2).addBands(ndvi_q3).addBands(ndvi_q4)
  .addBands(green_q1).addBands(green_q2).addBands(green_q3).addBands(green_q4)
  .addBands(landscan)

print('42-band image - final:', all_final)

// last visualization for check
// var bands = '10_12_NDVI'
// Map.addLayer(all_final.select(bands), {"opacity":1,"bands":bands,"palette":["ffffff","ff6b6b","d61c09"]}, '49-band Image')
// Map.centerObject(roi, 8)
// Map.setOptions('SATELLITE')
// Map.addLayer(irr_ag)

// new band order:
// b1-b9 = l8
// b10-b21 = rainfall
// b22-b31 = viirs
// b32-b35 = NDVI
// b36-b39 = GI
// b42 = pop density
// b43 = hbase prob

/* Resolution for referece:
viirs = 500m
l8 = 30m
ndvi = 30m
green = 30m
IMERG_V04 = 10km
POP_DENS = 1km
*/


var thirty_m = ['B1','B2','B3','B4','B5','B6','B7','B10','B11',
  '01_03_NDVI', '04_06_NDVI', '07_09_NDVI', '10_12_NDVI', '01_03_GREEN', 
  '04_06_GREEN', '07_09_GREEN', '10_12_GREEN']

var five_hundred_m = ['01VIIRS', '02VIIRS', '03VIIRS', '04VIIRS',
  '05VIIRS', '06VIIRS', '07VIIRS', '08VIIRS', '09VIIRS', '10VIIRS', '11VIIRS', '12VIIRS']

var one_km = ['POP_DENS']

var ten_km = ['01RAIN', '02RAIN', '03RAIN', '04RAIN', '05RAIN', '06RAIN', '07RAIN', 
  '08RAIN', '09RAIN', '10RAIN', '11RAIN', '12RAIN']

//dont forget 'CEN_2011'
// 10th, 25th, 50th, 75th, 90th, min, max, std-dev

var bihar_villages = ee.FeatureCollection("users/bawong/bassConnections/biharElectrificationMap")//.limit(10)
print(bihar_villages.first())

var mappedReduction500m_10th = bihar_villages.map(function(feature) {
  return feature.set(all_final.select(five_hundred_m).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([10]),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

var mappedReduction500m_25th = bihar_villages.map(function(feature) {
  return feature.set(all_final.select(five_hundred_m).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([25]),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

var mappedReduction500m_med = bihar_villages.map(function(feature) {
  return feature.set(all_final.select(five_hundred_m).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([50]),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

var mappedReduction500m_75th = bihar_villages.map(function(feature) {
  return feature.set(all_final.select(five_hundred_m).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([75]),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

var mappedReduction500m_90th = bihar_villages.map(function(feature) {
  return feature.set(all_final.select(five_hundred_m).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([90]),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

var mappedReduction500m_min = bihar_villages.map(function(feature) {
  return feature.set(all_final.select(five_hundred_m).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([0.01]),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

var mappedReduction500m_max = bihar_villages.map(function(feature) {
  return feature.set(all_final.select(five_hundred_m).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([100]),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

var mappedReduction500m_std_dev = bihar_villages.map(function(feature) {
  return feature.set(all_final.select(five_hundred_m).clip(feature).reduceRegion({
    reducer: ee.Reducer.stdDev(),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

// var bihar_att = ['CEN_2011', 'B1','B2','B3','B4','B5','B6','B7','B10','B11', '01RAIN', '02RAIN', '03RAIN', 
//   '04RAIN', '05RAIN', '06RAIN', '07RAIN', '08RAIN', '09RAIN', '10RAIN', '11RAIN', '12RAIN', '01VIIRS', '02VIIRS',
//   '03VIIRS', '04VIIRS', '05VIIRS', '06VIIRS', '07VIIRS', '08VIIRS', '09VIIRS', '10VIIRS', '11VIIRS', '12VIIRS',
//   '01_03_NDVI', '04_06_NDVI', '07_09_NDVI', '10_12_NDVI', '01_03_GREEN', '04_06_GREEN', '07_09_GREEN', '10_12_GREEN',
//   'POP_DENS']

//var bihar_att = ['CEN_2011', 'HH', 'eH', 'NAME']
var bihar_att = ['CEN_2011', 'HH', 'eH', 'NAME', '01VIIRS', '02VIIRS', '03VIIRS', '04VIIRS', '05VIIRS', '06VIIRS', '07VIIRS', '08VIIRS', '09VIIRS', '10VIIRS', '11VIIRS', '12VIIRS']
  
// 10th, 25th, 50th, 75th, 90th, min, max, std-dev

Export.table.toDrive({
  collection: mappedReduction500m_10th.select({propertySelectors: bihar_att, retainGeometry: false}),
  description: "bihar_12viirs_10th",
  folder: 'Feature_Extraction',
  fileFormat: "CSV"
});

Export.table.toDrive({
  collection: mappedReduction500m_25th.select({propertySelectors: bihar_att, retainGeometry: false}),
  description: "bihar_12viirs_25th",
  folder: 'Feature_Extraction',
  fileFormat: "CSV"
});

Export.table.toDrive({
  collection: mappedReduction500m_med.select({propertySelectors: bihar_att, retainGeometry: false}),
  description: "bihar_12viirs_med",
  folder: 'Feature_Extraction',
  fileFormat: "CSV"
});

Export.table.toDrive({
  collection: mappedReduction500m_75th.select({propertySelectors: bihar_att, retainGeometry: false}),
  description: "bihar_12viirs_75th",
  folder: 'Feature_Extraction',
  fileFormat: "CSV"
});

Export.table.toDrive({
  collection: mappedReduction500m_90th.select({propertySelectors: bihar_att, retainGeometry: false}),
  description: "bihar_12viirs_90th",
  folder: 'Feature_Extraction',
  fileFormat: "CSV"
});

Export.table.toDrive({
  collection: mappedReduction500m_min.select({propertySelectors: bihar_att, retainGeometry: false}),
  description: "bihar_12viirs_min",
  folder: 'Feature_Extraction',
  fileFormat: "CSV"
});

Export.table.toDrive({
  collection: mappedReduction500m_max.select({propertySelectors: bihar_att, retainGeometry: false}),
  description: "bihar_12viirs_max",
  folder: 'Feature_Extraction',
  fileFormat: "CSV"
});

Export.table.toDrive({
  collection: mappedReduction500m_std_dev.select({propertySelectors: bihar_att, retainGeometry: false}),
  description: "bihar_12viirs_std_dev",
  folder: 'Feature_Extraction',
  fileFormat: "CSV"
});