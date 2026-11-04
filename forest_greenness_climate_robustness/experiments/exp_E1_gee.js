var IFL_YEAR  = 2020;   // 2020 or 2025 — which IFL extent to use as the mask
var MVP_MODE  = false;  // true = Africa only, 2000-2001 (smoke test, ~2 min)

var SIGMA     = 0.5;
var SCALE     = 5000;   // metres — MOD13A3 native at 1 km resampled to 5 km
var DRIVE_FOLDER = 'ForestResilience_E1';
var COLLECTION   = 'MODIS/061/MOD13A3';

var START_DATE = '2000-01-01';
var END_DATE   = MVP_MODE ? '2001-12-31' : '2024-12-31';

var MAX_IMAGES = 300;

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

var REGIONS = MVP_MODE
  ? {africa: ALL_REGIONS.africa}
  : ALL_REGIONS;

var ifl2025m    = ee.Image('projects/ee-potapovpeter/assets/IFL/IFL_2025m');
var ifl2025loss = ee.Image('projects/ee-potapovpeter/assets/IFL/IFL_2025_loss');

var iflMask;
if (IFL_YEAR === 2020) {
  iflMask = ifl2025m.eq(1).or(ifl2025loss.eq(4)).rename('ifl_mask');
  print('IFL mask: 2020 extent (IFL_2025m ∪ areas lost 2020-2025)');
} else {
  iflMask = ifl2025m.eq(1).rename('ifl_mask');
  print('IFL mask: 2025 extent (IFL_2025m only)');
}

function applyQualityMask(image) {
  return image.updateMask(image.select('SummaryQA').lte(1));
}

function computeKNDVI(image) {
  var nir   = image.select('sur_refl_b02').multiply(0.0001);
  var red   = image.select('sur_refl_b01').multiply(0.0001);
  var ratio = nir.subtract(red).divide(2.0 * SIGMA);
  var kndvi = ratio.multiply(ratio).tanh().rename('kNDVI');

  var latlon = ee.Image.pixelLonLat();

  return kndvi
    .addBands(latlon.select('longitude').rename('longitude'))
    .addBands(latlon.select('latitude').rename('latitude'))
    .copyProperties(image, ['system:time_start']);
}

function samplePixels(image, geometry, regionName) {
  var date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');

  var masked = image.updateMask(iflMask);

  var samples = masked.sample({
    region:      geometry,
    scale:       SCALE,
    projection:  'EPSG:4326',
    geometries:  false,    // coords come from the longitude/latitude bands, not .geo
    dropNulls:   true
  });

  return samples.map(function(feature) {
    var lon = ee.Number(feature.get('longitude')).format('%.4f');
    var lat = ee.Number(feature.get('latitude')).format('%.4f');
    var pid = ee.String(lon).cat('_').cat(lat);
    return feature
      .set('date',     date)
      .set('region',   regionName)
      .set('pixel_id', pid);
  });
}

var regionNames = Object.keys(REGIONS);

for (var i = 0; i < regionNames.length; i++) {
  (function(regionName) {
    var geometry = REGIONS[regionName];

    var kndviCollection = ee.ImageCollection(COLLECTION)
      .filterDate(START_DATE, END_DATE)
      .filterBounds(geometry)
      .map(applyQualityMask)
      .map(computeKNDVI);

    var imageList = kndviCollection.toList(MAX_IMAGES);

    var allRows = ee.FeatureCollection(
      imageList.map(function(imgObj) {
        return samplePixels(ee.Image(imgObj), geometry, regionName);
      })
    ).flatten();

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

var firstRaw = ee.ImageCollection(COLLECTION)
  .filterDate('2000-01-01', '2000-02-01')
  .first();
print('First MOD13A3 date:', firstRaw.date().format('YYYY-MM-dd'));
print('Band names:', firstRaw.bandNames());
print('IFL_YEAR:', IFL_YEAR, '| MVP_MODE:', MVP_MODE, '| Scale:', SCALE, 'm');

var previewImg = ee.Image(
  ee.ImageCollection(COLLECTION)
    .filterDate('2000-01-01', '2000-02-28')
    .filterBounds(ALL_REGIONS.africa)
    .map(applyQualityMask)
    .map(computeKNDVI)
    .mean()  // collapses to ee.Image — avoids Element type error on Map.addLayer
).updateMask(iflMask).clip(ALL_REGIONS.africa);

Map.centerObject(ALL_REGIONS.africa, 3);

Map.addLayer(
  iflMask.selfMask().clip(ALL_REGIONS.africa),
  {palette: ['#005500'], opacity: 0.5},
  'IFL ' + IFL_YEAR + ' mask — Africa'
);

Map.addLayer(
  previewImg.select('kNDVI'),
  {min: 0.0, max: 0.9, palette: ['#8B4513', '#FFFF00', '#006400']},
  'kNDVI Jan 2000 — Africa (IFL pixels)'
);
