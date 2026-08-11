// Export annual USFS/NLCD Tree Canopy Cover for the western 11-state area.
// Run in the Google Earth Engine Code Editor.

var START_YEAR = 2000;
var END_YEAR = 2023;
var COLLECTION_ID = 'USGS/NLCD_RELEASES/2023_REL/TCC/v2023-5';
var BAND = 'NLCD_Percent_Tree_Canopy_Cover';
var CRS = 'EPSG:5070';
var SCALE = 30;
var DRIVE_FOLDER = 'WesternUS_TCC_v2023_5_30m';

var states = ee.FeatureCollection('TIGER/2018/States');
var western11Names = [
  'Arizona', 'California', 'Colorado', 'Idaho', 'Montana', 'Nevada',
  'New Mexico', 'Oregon', 'Utah', 'Washington', 'Wyoming'
];
var region = states
  .filter(ee.Filter.inList('NAME', western11Names))
  .geometry()
  .dissolve();

var tcc = ee.ImageCollection(COLLECTION_ID)
  .filter(ee.Filter.eq('study_area', 'CONUS'));

function annualTcc(year) {
  return tcc
    .filter(ee.Filter.eq('year', year))
    .first()
    .select(BAND)
    .rename('TCC')
    .clip(region)
    .set('year', year)
    .set('source_collection', COLLECTION_ID)
    .set('source_band', BAND);
}

ee.List.sequence(START_YEAR, END_YEAR).getInfo().forEach(function(year) {
  Export.image.toDrive({
    image: annualTcc(year),
    description: 'WesternUS_TCC_' + year,
    folder: DRIVE_FOLDER,
    fileNamePrefix: 'WesternUS_TCC_' + year,
    region: region,
    crs: CRS,
    scale: SCALE,
    maxPixels: 1e13
  });
});
