// =====================================================
// Export annual legacy-style RESI for Western US 11 states
// Formula matches the verified Coast 3-state legacy workflow:
// 1) build growing-season (May-Sep) Landsat median composite
// 2) compute NDVI and NDMI
// 3) RESI = 0.5 * (NDVI + NDMI)
// 4) unitScale(-0.2, 0.8).clamp(0, 1)
// 5) multiply by 10000, round, export as UInt16
// Output CRS: EPSG:5070
// Output resolution: 1000 m
// =====================================================

/******************** Parameters ********************/
var START_YEAR = 2000;
var END_YEAR   = 2023;
var GS_START_MONTH = 5;
var GS_END_MONTH   = 9;
var CRS = 'EPSG:5070';
var SCALE = 1000;
var MAX_PIXELS = 1e13;
var DRIVE_FOLDER = 'WesternUS_RESI_u16_1km';
/****************************************************/

// === Western US 11-state boundary ===
var states = ee.FeatureCollection('TIGER/2018/States');

var western11Names = [
  'Arizona',
  'California',
  'Colorado',
  'Idaho',
  'Montana',
  'Nevada',
  'New Mexico',
  'Oregon',
  'Utah',
  'Washington',
  'Wyoming'
];

var REGION = states
  .filter(ee.Filter.inList('NAME', western11Names))
  .geometry()
  .dissolve();

Map.centerObject(REGION, 5);
Map.addLayer(REGION, {color: 'red'}, 'WesternUS11 boundary', false);

// === L2 mask: cloud / shadow / snow / cirrus ===
function maskL2(img) {
  var qa = img.select('QA_PIXEL').toInt();
  var ok = qa.bitwiseAnd(1 << 3).eq(0)   // cloud shadow
    .and(qa.bitwiseAnd(1 << 4).eq(0))    // snow
    .and(qa.bitwiseAnd(1 << 5).eq(0))    // cloud
    .and(qa.bitwiseAnd(1 << 7).eq(0));   // cirrus
  return img.updateMask(ok);
}

// === C2 L2 reflectance scaling ===
// reflectance = DN * 0.0000275 - 0.2
function scaleSR(img) {
  var scaled = img.select(['SR_B.*']).multiply(0.0000275).add(-0.2);
  return img.addBands(scaled, null, true);
}

// === Harmonize band names ===
function rename_TM_ETM(img) {
  return img.select(
    ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5'],
    ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1']
  );
}

function rename_OLI(img) {
  return img.select(
    ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6'],
    ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1']
  );
}

// === Add NDVI / NDMI ===
function addIndices(img) {
  var nir = img.select('NIR');
  var red = img.select('RED');
  var sw1 = img.select('SWIR1');

  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  var ndmi = nir.subtract(sw1).divide(nir.add(sw1)).rename('NDMI');

  return img.addBands([ndvi, ndmi]);
}

// === Merge Landsat 5 / 7 / 8 / 9 collections ===
function lsCol(start, end) {
  var lt5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
    .filterBounds(REGION)
    .filterDate(start, end)
    .map(maskL2)
    .map(scaleSR)
    .map(rename_TM_ETM)
    .map(addIndices);

  var le7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    .filterBounds(REGION)
    .filterDate(start, end)
    .map(maskL2)
    .map(scaleSR)
    .map(rename_TM_ETM)
    .map(addIndices);

  var lc8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(REGION)
    .filterDate(start, end)
    .map(maskL2)
    .map(scaleSR)
    .map(rename_OLI)
    .map(addIndices);

  var lc9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterBounds(REGION)
    .filterDate(start, end)
    .map(maskL2)
    .map(scaleSR)
    .map(rename_OLI)
    .map(addIndices);

  return lt5.merge(le7).merge(lc8).merge(lc9);
}

// === Annual growing-season legacy RESI ===
function annualRESI_u16(year) {
  year = ee.Number(year);
  var start = ee.Date.fromYMD(year, GS_START_MONTH, 1);
  var end = ee.Date.fromYMD(year, GS_END_MONTH, 30).advance(1, 'day');

  var col = lsCol(start, end);

  var resi = ee.Image(ee.Algorithms.If(
    col.size().gt(0),
    (function() {
      var med = col.select(['NDVI', 'NDMI']).median();
      var resi01 = med.expression('(ndvi + ndmi) / 2', {
          ndvi: med.select('NDVI'),
          ndmi: med.select('NDMI')
        })
        .unitScale(-0.2, 0.8)
        .clamp(0, 1);

      return resi01.multiply(10000).round().toUint16().rename('RESI_u16');
    })(),
    ee.Image(0).toUint16().rename('RESI_u16')
  ));

  return resi
    .clip(REGION)
    .reproject({crs: CRS, scale: SCALE})
    .set('year', year);
}

// === Preview ===
var previewYear = 2022;
var preview = annualRESI_u16(previewYear);
Map.addLayer(preview, {min: 0, max: 10000, palette: ['8c510a', 'd8b365', 'f6e8c3', 'c7eae5', '5ab4ac', '01665e']}, 'Legacy RESI_u16 ' + previewYear, false);

// === Export all years ===
ee.List.sequence(START_YEAR, END_YEAR).getInfo().forEach(function(year) {
  var img = annualRESI_u16(year);

  Export.image.toDrive({
    image: img,
    description: 'WesternUS_RESI_u16_' + year,
    folder: DRIVE_FOLDER,
    fileNamePrefix: 'WesternUS_RESI_u16_' + year,
    region: REGION,
    crs: CRS,
    scale: SCALE,
    maxPixels: MAX_PIXELS
  });
});
