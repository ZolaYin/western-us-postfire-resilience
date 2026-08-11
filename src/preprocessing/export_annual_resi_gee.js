// Export annual RESI for the 11-state western U.S. study area.
// Run in the Google Earth Engine Code Editor.

var START_YEAR = 2000;
var END_YEAR = 2023;
var GS_START_MONTH = 5;
var GS_END_MONTH = 9;
var CRS = 'EPSG:5070';
var SCALE = 1000;
var DRIVE_FOLDER = 'WesternUS_RESI_u16_1km';

var states = ee.FeatureCollection('TIGER/2018/States');
var western11Names = [
  'Arizona', 'California', 'Colorado', 'Idaho', 'Montana', 'Nevada',
  'New Mexico', 'Oregon', 'Utah', 'Washington', 'Wyoming'
];
var region = states
  .filter(ee.Filter.inList('NAME', western11Names))
  .geometry()
  .dissolve();

function maskL2(image) {
  var qa = image.select('QA_PIXEL').toInt();
  var ok = qa.bitwiseAnd(1 << 3).eq(0)  // cloud shadow
    .and(qa.bitwiseAnd(1 << 4).eq(0))   // snow
    .and(qa.bitwiseAnd(1 << 5).eq(0))   // cloud
    .and(qa.bitwiseAnd(1 << 7).eq(0));  // cirrus
  return image.updateMask(ok);
}

function scaleSR(image) {
  var scaled = image.select(['SR_B.*']).multiply(0.0000275).add(-0.2);
  return image.addBands(scaled, null, true);
}

function renameTMETM(image) {
  return image.select(
    ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5'],
    ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1']
  );
}

function renameOLI(image) {
  return image.select(
    ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6'],
    ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1']
  );
}

function addIndices(image) {
  var ndvi = image.normalizedDifference(['NIR', 'RED']).rename('NDVI');
  var ndmi = image.normalizedDifference(['NIR', 'SWIR1']).rename('NDMI');
  return image.addBands([ndvi, ndmi]);
}

function prepare(collectionId, start, end, renameFunction) {
  return ee.ImageCollection(collectionId)
    .filterBounds(region)
    .filterDate(start, end)
    .map(maskL2)
    .map(scaleSR)
    .map(renameFunction)
    .map(addIndices);
}

function landsatCollection(start, end) {
  return prepare('LANDSAT/LT05/C02/T1_L2', start, end, renameTMETM)
    .merge(prepare('LANDSAT/LE07/C02/T1_L2', start, end, renameTMETM))
    .merge(prepare('LANDSAT/LC08/C02/T1_L2', start, end, renameOLI))
    .merge(prepare('LANDSAT/LC09/C02/T1_L2', start, end, renameOLI));
}

function annualResiU16(year) {
  year = ee.Number(year);
  var start = ee.Date.fromYMD(year, GS_START_MONTH, 1);
  var end = ee.Date.fromYMD(year, GS_END_MONTH, 30).advance(1, 'day');
  var median = landsatCollection(start, end).select(['NDVI', 'NDMI']).median();
  var resi = median.expression('(ndvi + ndmi) / 2', {
    ndvi: median.select('NDVI'),
    ndmi: median.select('NDMI')
  });
  return resi.unitScale(-0.2, 0.8)
    .clamp(0, 1)
    .multiply(10000)
    .round()
    .toUint16()
    .rename('RESI_u16')
    .clip(region)
    .set('year', year);
}

ee.List.sequence(START_YEAR, END_YEAR).getInfo().forEach(function(year) {
  Export.image.toDrive({
    image: annualResiU16(year),
    description: 'WesternUS_RESI_u16_' + year,
    folder: DRIVE_FOLDER,
    fileNamePrefix: 'WesternUS_RESI_u16_' + year,
    region: region,
    crs: CRS,
    scale: SCALE,
    maxPixels: 1e13
  });
});
