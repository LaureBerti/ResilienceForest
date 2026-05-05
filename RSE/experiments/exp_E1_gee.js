// =============================================================================
// E1 — Pixel-level kNDVI time series from MODIS MOD13A3 V6.1
//      masked to Intact Forest Landscapes (IFL 2020 or IFL 2025)
//
// Project : Analysis of Motifs of Intact Forest Resilience
// Journal : Remote Sensing of Environment
// Authors : Laure Berti-Équille, Pius N. Nwachukwu (IRD, ESPACE-DEV)
//
// IFL GEE assets (public — no upload required):
//   IFL_2025m    : projects/ee-potapovpeter/assets/IFL/IFL_2025m
//                  Raster, pixel value 1 = intact as of 2025
//   IFL_2025_loss: projects/ee-potapovpeter/assets/IFL/IFL_2025_loss
//                  Raster, pixel value encodes period of loss:
//                    1 = lost 2000–2013 | 2 = lost 2013–2016
//                    3 = lost 2016–2020 | 4 = lost 2020–2025
//                    0 = still intact in 2025
//   IFL 2020 extent = IFL_2025m  OR  IFL_2025_loss == 4
//     (forests that were intact in 2020, including those lost only in 2020-2025)
//
// Output per region (CSV, one row per pixel per month):
//   region | date | pixel_id | latitude | longitude | kNDVI
//
// kNDVI formula (tanh — NOT tan):
//   kNDVI = tanh( ((NIR - RED) / (2 * sigma))^2 )
//   Reference: Smith & Boers (2024) correction; sigma = 0.5
//
// HOW TO RUN:
//   1. Open https://code.earthengine.google.com/
//   2. Paste this entire script
//   3. Set IFL_YEAR and MVP_MODE (see CONFIGURATION below)
//   4. Click Run → export tasks appear in the Tasks tab
//   5. Click Run on each task to send to Google Drive
//   6. Download CSVs to data/raw/kndvi/ in your local project
// =============================================================================

// ---------------------------------------------------------------------------
// CONFIGURATION
// ---------------------------------------------------------------------------

var IFL_YEAR  = 2020;   // 2020 or 2025 — which IFL extent to use as the mask
var MVP_MODE  = false;  // true = Africa only, 2000-2001 (smoke test, ~2 min)

var SIGMA     = 0.5;
var SCALE     = 5000;   // metres — MOD13A3 native at 1 km resampled to 5 km
var DRIVE_FOLDER = 'ForestResilience_E1';
var COLLECTION   = 'MODIS/061/MOD13A3';

var START_DATE = '2000-01-01';
var END_DATE   = MVP_MODE ? '2001-12-31' : '2024-12-31';

// Maximum number of images to process per region when using toList()
// 25 years × 12 months = 300 images; set higher for safety
var MAX_IMAGES = 300;

// ---------------------------------------------------------------------------
// REGION BOUNDING BOXES  [lon_min, lat_min, lon_max, lat_max]
// Sub-regions are exported in addition to continents for finer analysis.
// ---------------------------------------------------------------------------

var ALL_REGIONS = {
  africa:               ee.Geometry.Rectangle([-20, -35,   55,  38]),
  asia:                 ee.Geometry.Rectangle([ 25, -10,  180,  77]),
  europe:               ee.Geometry.Rectangle([-25,  34,   45,  72]),
  north_america:        ee.Geometry.Rectangle([-170,  5,  -50,  83]),
  south_america:        ee.Geometry.Rectangle([-82, -57,  -33,  13]),
  australia_oceania:    ee.Geometry.Rectangle([110, -50,  180,  10]),
  amazon:               ee.Geometry.Rectangle([-75, -15,  -45,   5]),
  congo:                ee.Geometry.Rectangle([ 14,  -6,   30,   6]),
  boreal_siberia:       ee.Geometry.Rectangle([ 60,  58,  120,  72]),
  southeast_asia:       ee.Geometry.Rectangle([ 95, -10,  142,  15]),
  canadian_boreal:      ee.Geometry.Rectangle([-140, 55,  -90,  72]),
  scandinavian_boreal:  ee.Geometry.Rectangle([ 14,  59,   32,  71]),
  papua_new_guinea:     ee.Geometry.Rectangle([140, -10,  156,   0]),
  russian_far_east:     ee.Geometry.Rectangle([130,  44,  145,  60])
};

// In MVP_MODE only run Africa for a quick test
var REGIONS = MVP_MODE
  ? {africa: ALL_REGIONS.africa}
  : ALL_REGIONS;

// ---------------------------------------------------------------------------
// BUILD IFL FOREST MASK (pixel-level raster, no shapefile upload needed)
// ---------------------------------------------------------------------------

var ifl2025m    = ee.Image('projects/ee-potapovpeter/assets/IFL/IFL_2025m');
var ifl2025loss = ee.Image('projects/ee-potapovpeter/assets/IFL/IFL_2025_loss');

var iflMask;
if (IFL_YEAR === 2020) {
  // IFL 2020 = pixels still intact in 2025  OR  lost only in the 2020-2025 window
  //   ifl2025m.eq(1)           → intact in 2025 (subset of IFL 2020)
  //   ifl2025loss.eq(4)        → lost 2020-2025 (was still intact in 2020)
  iflMask = ifl2025m.eq(1).or(ifl2025loss.eq(4)).rename('ifl_mask');
  print('IFL mask: 2020 extent (IFL_2025m ∪ areas lost 2020-2025)');
} else {
  // IFL 2025 — strictest intact-forest definition
  iflMask = ifl2025m.eq(1).rename('ifl_mask');
  print('IFL mask: 2025 extent (IFL_2025m only)');
}

// ---------------------------------------------------------------------------
// HELPER FUNCTIONS
// ---------------------------------------------------------------------------

/** Keep pixels where SummaryQA is 0 (good) or 1 (marginal). */
function applyQualityMask(image) {
  return image.updateMask(image.select('SummaryQA').lte(1));
}

/**
 * Compute kNDVI and attach lat/lon bands.
 * Returns a 3-band image: kNDVI | longitude | latitude
 * kNDVI = tanh(((NIR - RED) / (2 * sigma))^2)   [Smith & Boers 2024 correction]
 * sur_refl_b02 = NIR, sur_refl_b01 = Red; scale factor 0.0001
 */
function computeKNDVI(image) {
  var nir   = image.select('sur_refl_b02').multiply(0.0001);
  var red   = image.select('sur_refl_b01').multiply(0.0001);
  var ratio = nir.subtract(red).divide(2.0 * SIGMA);
  var kndvi = ratio.multiply(ratio).tanh().rename('kNDVI');

  // Attach pixel coordinates so each sampled row carries lat/lon
  var latlon = ee.Image.pixelLonLat();

  return kndvi
    .addBands(latlon.select('longitude').rename('longitude'))
    .addBands(latlon.select('latitude').rename('latitude'))
    .copyProperties(image, ['system:time_start']);
}

/**
 * Sample all IFL pixels within a geometry for a single kNDVI image.
 * Returns a FeatureCollection where each feature is one pixel observation.
 * Columns: kNDVI | longitude | latitude | pixel_id | date | region
 *
 * This is the pixel-level extraction — NOT a region-mean.
 *
 * @param {ee.Image}    image      - Pre-computed kNDVI image (3 bands).
 * @param {ee.Geometry} geometry   - Bounding box for the region.
 * @param {string}      regionName - Region label attached to each row.
 * @returns {ee.FeatureCollection} - One feature per IFL pixel.
 */
function samplePixels(image, geometry, regionName) {
  var date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');

  // Apply the IFL mask before sampling — only IFL pixels are sampled
  var masked = image.updateMask(iflMask);

  // sample() draws one feature per pixel whose mask is valid within geometry
  var samples = masked.sample({
    region:      geometry,
    scale:       SCALE,
    projection:  'EPSG:4326',
    geometries:  false,    // coords come from the longitude/latitude bands, not .geo
    dropNulls:   true
  });

  // Attach metadata columns
  return samples.map(function(feature) {
    // Build a pixel_id from rounded lat/lon (stable across time steps)
    var lon = ee.Number(feature.get('longitude')).format('%.4f');
    var lat = ee.Number(feature.get('latitude')).format('%.4f');
    var pid = ee.String(lon).cat('_').cat(lat);
    return feature
      .set('date',     date)
      .set('region',   regionName)
      .set('pixel_id', pid);
  });
}

// ---------------------------------------------------------------------------
// MAIN LOOP — one export task per region
// ---------------------------------------------------------------------------

var regionNames = Object.keys(REGIONS);

for (var i = 0; i < regionNames.length; i++) {
  (function(regionName) {
    var geometry = REGIONS[regionName];

    // Build the kNDVI ImageCollection for this region
    var kndviCollection = ee.ImageCollection(COLLECTION)
      .filterDate(START_DATE, END_DATE)
      .filterBounds(geometry)
      .map(applyQualityMask)
      .map(computeKNDVI);

    // -----------------------------------------------------------------------
    // PIXEL-LEVEL EXTRACTION
    //
    // Pattern: convert ImageCollection → ee.List, then map over the List.
    // Mapping a function that returns a FeatureCollection over an ee.List
    // gives an ee.List of FeatureCollections, which can be passed to
    // ee.FeatureCollection().flatten() without the "Element type" error.
    //
    // Equivalent but WRONG:  imageCollection.map(fn).flatten()
    //   → .map() on ImageCollection coerces result back to ImageCollection,
    //     so .flatten() sees Image objects, not FeatureCollections.
    // -----------------------------------------------------------------------
    var imageList = kndviCollection.toList(MAX_IMAGES);

    var allRows = ee.FeatureCollection(
      imageList.map(function(imgObj) {
        return samplePixels(ee.Image(imgObj), geometry, regionName);
      })
    ).flatten();

    // Export to Google Drive as CSV
    var suffix = (MVP_MODE ? '_mvp' : '') + '_ifl' + IFL_YEAR;
    Export.table.toDrive({
      collection:     allRows,
      description:    'kndvi_' + regionName + '_pixels' + suffix,
      folder:         DRIVE_FOLDER,
      fileNamePrefix: 'kndvi_' + regionName + '_pixels' + suffix,
      fileFormat:     'CSV',
      selectors:      ['region', 'date', 'pixel_id', 'latitude', 'longitude', 'kNDVI']
    });

    print('Export queued:', regionName,
          '| images:', kndviCollection.size(),
          '| IFL year:', IFL_YEAR);

  })(regionNames[i]);
}

// ---------------------------------------------------------------------------
// DIAGNOSTICS
// ---------------------------------------------------------------------------

// Quick pixel count per continent — shows how many IFL pixels each export will have
var diagRegions = MVP_MODE ? ['africa'] : ['africa', 'asia', 'south_america', 'australia_oceania'];

diagRegions.forEach(function(rname) {
  var pixelCount = iflMask
    .selfMask()
    .reduceRegion({
      reducer:   ee.Reducer.count(),
      geometry:  ALL_REGIONS[rname],
      scale:     SCALE,
      maxPixels: 1e10
    });
  print('IFL pixel count —', rname, ':', pixelCount.get('ifl_mask'));
});

// First image metadata
var firstRaw = ee.ImageCollection(COLLECTION)
  .filterDate('2000-01-01', '2000-02-01')
  .first();
print('First MOD13A3 date:', firstRaw.date().format('YYYY-MM-dd'));
print('Band names:', firstRaw.bandNames());
print('IFL_YEAR:', IFL_YEAR, '| MVP_MODE:', MVP_MODE, '| Scale:', SCALE, 'm');

// ---------------------------------------------------------------------------
// MAP PREVIEW — kNDVI for Jan 2000 over Africa, IFL pixels only
// ---------------------------------------------------------------------------

var previewImg = ee.Image(
  ee.ImageCollection(COLLECTION)
    .filterDate('2000-01-01', '2000-02-28')
    .filterBounds(ALL_REGIONS.africa)
    .map(applyQualityMask)
    .map(computeKNDVI)
    .mean()  // collapses to ee.Image — avoids Element type error on Map.addLayer
).updateMask(iflMask).clip(ALL_REGIONS.africa);

Map.centerObject(ALL_REGIONS.africa, 3);

// IFL mask layer
Map.addLayer(
  iflMask.selfMask().clip(ALL_REGIONS.africa),
  {palette: ['#005500'], opacity: 0.5},
  'IFL ' + IFL_YEAR + ' mask — Africa'
);

// kNDVI layer (IFL pixels only)
Map.addLayer(
  previewImg.select('kNDVI'),
  {min: 0.0, max: 0.9, palette: ['#8B4513', '#FFFF00', '#006400']},
  'kNDVI Jan 2000 — Africa (IFL pixels)'
);
