/*
Script Objective: feature extraction
Loction: Uttar Pradesh (but can adjusted for anywhere)

The 2017-2018 Bass Connections in Energy Access team used this Google Earth Engine script to extract
features for Uttar Pradesh, India.

Due to differences in pixel sizes for different categories of bands (e.g. VIIRS = 500m resolution, whereas
Landsat-8 spectral bands = 30m), the script needs to be run once for each scale. This can be done simply
by changing the 'resolution' variable on line 207.

The output is a CSV to your Google Drive of choice.

Base Google Earth Engine script: https://code.earthengine.google.com/db05061a8db023ef4e4913ef3b7e4c42

*/

//////---- VARIABLES -----//////
var grid = ee.FeatureCollection('users/xlany/grid_5km_exact'); //grid length = 638k
var ind = ee.FeatureCollection('users/bl/Ind_admin_shapefiles/ind_states');
var up = ind.filter(ee.Filter.eq('st_name','Uttar Pradesh'));
var roi = up;
var irr_ag = ee.Image("users/bawong/bassConnections/giam_2014_2015_v2").clip(roi)  // 0 = non-irr ag, 1 = irr ag
var l8sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(roi)
var quasivillages = grid.filterBounds(up)

var hbase = ee.ImageCollection
  .fromImages([
    ee.Image("users/xlany/43Q_up_hbase_probability_30m"), 
    ee.Image("users/xlany/43R_up_hbase_probability_30m"), 
    ee.Image("users/xlany/44Q_up_hbase_probability_30m"), 
    ee.Image("users/xlany/44R_up_hbase_probability_30m"), 
    ee.Image("users/xlany/45Q_up_hbase_probability_30m"), 
    ee.Image("users/xlany/45R_up_hbase_probability_30m")])
    .mosaic()
    .clip(roi)
    .select('b1')
    .gte(2) // global threshold for LAN
    .rename('HBASE_PROB')

var landscan = ee.Image("users/bassconnectionsenergyaccess/landscan_india_2016")
  .clip(roi)
  .select('b1')
  .rename('POP_DENS')

//////---- FUNCTIONS -----//////

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

var ndvi_base = ee.ImageCollection("MODIS/006/MOD13Q1").filterBounds(roi).select('NDVI')
var ndvi_final = ndvi_base
  .filterDate('2016-01-01', '2016-01-31')
  .mean()
  .clip(roi)
  .select('NDVI')
  .rename('01NDVI')

var evi_base = ee.ImageCollection("MODIS/006/MOD13Q1").filterBounds(roi).select('EVI')
var evi_final = evi_base
  .filterDate('2016-01-01', '2016-01-31')
  .mean()
  .clip(roi)
  .select('EVI')
  .rename('01EVI')
  
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
  
  //=================== Data for the month: ndvi/green index ===================
  
  var ndvi = ndvi_base
    .filterDate(date_start, date_end)
    .mean()
    .clip(roi)
    .select('NDVI')
    .rename(monthString+'NDVI')
  
  var evi = evi_base
    .filterDate(date_start, date_end)
    .mean()
    .clip(roi)
    .select('EVI')
    .rename(monthString+'EVI')

	//=================== ADD MONTHLY VIIRS===================
	var viirs = viirs_raw
	.filterDate(date_start, date_end)
	.reduce(ee.Reducer.median()) // monthly composite pre-made // just a trick to reduce to image
	.clip(roi)
	.mask(hbase)
	.select('avg_rad_median').float().rename(monthString+'VIIRS');

	//=================== Composing (append to each base) ===================
	ndvi_final = ndvi_final.addBands(ndvi)
	evi_final = evi_final.addBands(evi)
  viirs_final = viirs_final.addBands(viirs)

}

// make x band image by adding them all together
// var all_final = greenestCloudfreeComposite.addBands(ndvi_final).addBands(green_final).addBands(rain_final).addBands(viirs_final)
var all_final = greenestCloudfreeComposite
  .addBands(ndvi_final)
  .addBands(evi_final)
  .addBands(viirs_final)
  .addBands(landscan)

print(all_final)  

/* Band Resolution for Referece:
viirs = 500m
l8 = 30m
ndvi = 250m
evi = 250m
POP_DENS = 1km
*/

// lists of band names for loops with system_ind to make joins from imagery to grid
//split into groups by resolution 
//also split by subgroups if necessary (e.g. b vs vegetation) for runtime purposes

var bihar_att_spectral = ['system_ind', '01VIIRS', '02VIIRS', '03VIIRS', '04VIIRS',
  '05VIIRS', '06VIIRS', '07VIIRS', '08VIIRS', '09VIIRS', '10VIIRS', '11VIIRS', '12VIIRS']

var bihar_att_b = ['system_ind','B1','B2','B3','B4','B5','B6','B7','B10','B11']

var bihar_att_veg = ['system_ind', '01NDVI', '02NDVI', '03NDVI', '04NDVI', '05NDVI', 
  '06NDVI', '07NDVI', '08NDVI', '09NDVI', '10NDVI', '11NDVI', '12NDVI', '01EVI', 
  '02EVI', '03EVI', '04EVI', '05EVI', '06EVI', '07EVI', '08EVI', '09EVI', '10EVI', '11EVI', '12EVI']

var bihar_att_pop = ['system_ind', 'POP_DENS']

// lists of band names for loops

var thirty_m = ['B1','B2','B3','B4','B5','B6','B7','B10','B11']

var two_hundred_fifty_m = ['01NDVI', '02NDVI', '03NDVI', '04NDVI', '05NDVI', 
  '06NDVI', '07NDVI', '08NDVI', '09NDVI', '10NDVI', '11NDVI', '12NDVI',
  '01EVI', '02EVI', '03EVI', '04EVI', '05EVI', '06EVI', '07EVI', '08EVI', 
  '09EVI', '10EVI', '11EVI', '12EVI']

var five_hundred_m = ['01VIIRS', '02VIIRS', '03VIIRS', '04VIIRS',
  '05VIIRS', '06VIIRS', '07VIIRS', '08VIIRS', '09VIIRS', '10VIIRS', '11VIIRS', '12VIIRS']

var one_km = ['POP_DENS']

var ten_km = ['01RAIN', '02RAIN', '03RAIN', '04RAIN', '05RAIN', '06RAIN', '07RAIN', 
  '08RAIN', '09RAIN', '10RAIN', '11RAIN', '12RAIN']

// pick which bands to run here. need to run once for each band resolution
var resolution = two_hundred_fifty_m
var fc = quasivillages

// ===================== FEATURE EXTRACTION FUNCTIONS ===================
var mappedReduction_10th_30 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([10]),
    geometry: feature.geometry(),
    scale: 30,
  }))
})

var mappedReduction_25th_30 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([25]),
    geometry: feature.geometry(),
    scale: 30,
  }))
})

var mappedReduction_med_30 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([50]),
    geometry: feature.geometry(),
    scale: 30,
  }))
})

var mappedReduction_75th_30 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([75]),
    geometry: feature.geometry(),
    scale: 30,
  }))
})

var mappedReduction_90th_30 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([90]),
    geometry: feature.geometry(),
    scale: 30,
  }))
})

var mappedReduction_min_30 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([0]),
    geometry: feature.geometry(),
    scale: 30,
  }))
})

var mappedReduction_max_30 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([100]),
    geometry: feature.geometry(),
    scale: 30,
  }))
})

var mappedReduction_std_dev_30 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.stdDev(),
    geometry: feature.geometry(),
    scale: 30,
  }))
})

var mappedReduction_sum_30 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: feature.geometry(),
    scale: 30,
  }))
})

// 250m
var mappedReduction_10th_250 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([10]),
    geometry: feature.geometry(),
    scale: 250,
  }))
})

var mappedReduction_25th_250 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([25]),
    geometry: feature.geometry(),
    scale: 250,
  }))
})

var mappedReduction_med_250 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([50]),
    geometry: feature.geometry(),
    scale: 250,
  }))
})

var mappedReduction_75th_250 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([75]),
    geometry: feature.geometry(),
    scale: 250,
  }))
})

var mappedReduction_90th_250 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([90]),
    geometry: feature.geometry(),
    scale: 250,
  }))
})

var mappedReduction_min_250 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([0]),
    geometry: feature.geometry(),
    scale: 250,
  }))
})

var mappedReduction_max_250 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([100]),
    geometry: feature.geometry(),
    scale: 250,
  }))
})

var mappedReduction_std_dev_250 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.stdDev(),
    geometry: feature.geometry(),
    scale: 250,
  }))
})

var mappedReduction_sum_250 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: feature.geometry(),
    scale: 250,
  }))
})

var mappedReduction_mean_250 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: feature.geometry(),
    scale: 250,
  }))
})

// 500m
var mappedReduction_10th_500 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([10]),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

var mappedReduction_25th_500 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([25]),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

var mappedReduction_med_500 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([50]),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

var mappedReduction_75th_500 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([75]),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

var mappedReduction_90th_500 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([90]),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

var mappedReduction_min_500 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([0]),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

var mappedReduction_max_500 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([100]),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

var mappedReduction_std_dev_500 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.stdDev(),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

var mappedReduction_sum_500 = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: feature.geometry(),
    scale: 500,
  }))
})

// 1km
var mappedReduction_10th_1k = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([10]),
    geometry: feature.geometry(),
    scale: 1000,
  }))
})

var mappedReduction_25th_1k = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([25]),
    geometry: feature.geometry(),
    scale: 1000,
  }))
})

var mappedReduction_med_1k = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([50]),
    geometry: feature.geometry(),
    scale: 1000,
  }))
})

var mappedReduction_75th_1k = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([75]),
    geometry: feature.geometry(),
    scale: 1000,
  }))
})

var mappedReduction_90th_1k = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([90]),
    geometry: feature.geometry(),
    scale: 1000,
  }))
})

var mappedReduction_min_1k = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([0]),
    geometry: feature.geometry(),
    scale: 1000,
  }))
})

var mappedReduction_max_1k = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.percentile([100]),
    geometry: feature.geometry(),
    scale: 1000,
  }))
})

var mappedReduction_std_dev_1k = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.stdDev(),
    geometry: feature.geometry(),
    scale: 1000,
  }))
})

var mappedReduction_sum_1k = fc.map(function(feature) {
  return feature.set(all_final.select(resolution).clip(feature).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: feature.geometry(),
    scale: 1000,
  }))
})

var mappedGeoJSON = fc.map(function(feature){
  return feature;
})

// ======================= EXPORT ===========================
// 10th, 25th, 50th, 75th, 90th, min, max, std-dev
if (resolution == thirty_m) {
  Export.table.toDrive({
    collection: mappedReduction_10th_30.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_veg_10th",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_25th_30.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_veg_25th",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_med_30.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_veg_med",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_75th_30.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_veg_75th",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_90th_30.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_veg_90th",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_min_30.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_veg_min",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_max_30.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_veg_max",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_std_dev_30.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_veg_std_dev",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  // Export.table.toDrive({
  //   collection: mappedReduction_sum_30.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
  //   description: "up_veg_sum",
  //   folder: 'Feature_Extraction',
  //   fileFormat: "CSV"
  // });
}

if (resolution == two_hundred_fifty_m) {
  Export.table.toDrive({
    collection: mappedReduction_10th_250.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_ndvi_10th",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_25th_250.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_ndvi_25th",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_med_250.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_ndvi_med",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_75th_250.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_ndvi_75th",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_90th_250.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_ndvi_90th",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_min_250.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_ndvi_min",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_max_250.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_ndvi_max",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_std_dev_250.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_ndvi_std_dev",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_sum_250.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_ndvi_sum",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_mean_250.select({propertySelectors: bihar_att_veg, retainGeometry: false}),
    description: "up_ndvi_mean",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
}


// 500m
if (resolution == five_hundred_m) {
  Export.table.toDrive({
    collection: mappedReduction_10th_500.select({propertySelectors: bihar_att_spectral, retainGeometry: false}),
    description: "up_spectral_10th",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_25th_500.select({propertySelectors: bihar_att_spectral, retainGeometry: false}),
    description: "up_spectral_25th",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_med_500.select({propertySelectors: bihar_att_spectral, retainGeometry: false}),
    description: "up_spectral_med",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_75th_500.select({propertySelectors: bihar_att_spectral, retainGeometry: false}),
    description: "up_spectral_75th",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_90th_500.select({propertySelectors: bihar_att_spectral, retainGeometry: false}),
    description: "up_spectral_90th",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_min_500.select({propertySelectors: bihar_att_spectral, retainGeometry: false}),
    description: "up_spectral_min",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_max_500.select({propertySelectors: bihar_att_spectral, retainGeometry: false}),
    description: "up_spectral_max",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_std_dev_500.select({propertySelectors: bihar_att_spectral, retainGeometry: false}),
    description: "up_spectral_std_dev",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  // Export.table.toDrive({
  //   collection: mappedReduction_sum_500.select({propertySelectors: bihar_att_spectral, retainGeometry: false}),
  //   description: "up_spectral_sum",
  //   folder: 'Feature_Extraction',
  //   fileFormat: "CSV"
  // });
}

// 1kmm
if (resolution == one_km) {
  Export.table.toDrive({
    collection: mappedReduction_10th_1k.select({propertySelectors: bihar_att_pop, retainGeometry: false}),
    description: "up_pop_10th",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_25th_1k.select({propertySelectors: bihar_att_pop, retainGeometry: false}),
    description: "up_pop_25th",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_med_1k.select({propertySelectors: bihar_att_pop, retainGeometry: false}),
    description: "up_pop_med",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_75th_1k.select({propertySelectors: bihar_att_pop, retainGeometry: false}),
    description: "up_pop_75th",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_90th_1k.select({propertySelectors: bihar_att_pop, retainGeometry: false}),
    description: "up_pop_90th",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_min_1k.select({propertySelectors: bihar_att_pop, retainGeometry: false}),
    description: "up_pop_min",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_max_1k.select({propertySelectors: bihar_att_pop, retainGeometry: false}),
    description: "up_pop_max",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  Export.table.toDrive({
    collection: mappedReduction_std_dev_1k.select({propertySelectors: bihar_att_pop, retainGeometry: false}),
    description: "up_pop_std_dev",
    folder: 'Feature_Extraction',
    fileFormat: "CSV"
  });
  
  // Export.table.toDrive({
  //   collection: mappedReduction_sum_1k.select({propertySelectors: bihar_att_pop, retainGeometry: false}),
  //   description: "up_pop_sum",
  //   folder: 'Feature_Extraction',
  //   fileFormat: "CSV"
  // });
}

var label = ['system_ind'];
var geo_tags = mappedGeoJSON.select({propertySelectors: label, retainGeometry: true})
Export.table.toDrive({
  collection: geo_tags,
  description: "up_geotags_5km",
  // folder: 'Feature_Extraction',
  fileFormat: "GeoJSON"
})

// If you want to visualize certain bands
Map.addLayer(all_final.select('12VIIRS'), {"opacity":1,"bands":["12VIIRS"],"palette":["ffffff","ff6b6b","d61c09"]}, '49-band Image')
Map.addLayer(all_final.select('01NDVI'), {"opacity":0.9,"bands":["01NDVI"],"max":1,"palette":["ffffff","47d001"]}, 'NDVI Band')
Map.addLayer(all_final.select('01EVI'), {"opacity":0.9,"bands":["01EVI"],"max":1,"palette":["ffffff","47d001"]}, 'EVI Band')

Map.addLayer(all_final.select('POP_DENS'), {"opacity":1, "bands":["POP_DENS"], "palette":["ffffff","111c51"]}, 'Population Density Band')
// Map.addLayer(quasivillages, {"opacity":1,"palette":["00ffff","47d001"]}, 'Village Boundaries')
Map.centerObject(roi, 8)
Map.setOptions('SATELLITE')

print("Finished")