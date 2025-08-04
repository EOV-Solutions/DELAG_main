// Define ROI
var roi = ee.Geometry.Rectangle([-10, 35, 40, 70]);

// Load MODIS MOD11A1
var dataset = ee.ImageCollection('MODIS/061/MOD11A1')
  .filter(ee.Filter.date('2023-01-01', '2023-01-02'))
  .select(['LST_Day_1km', 'QC_Day']);

// Quality filter
var filterDay = function(image) {
  var qa = image.select('QC_Day');
  var bitMask = 1 << 2;
  var mask = qa.bitwiseAnd(bitMask).eq(0);
  return image.updateMask(mask);
};
var lst_filtered = dataset.map(filterDay);

// Convert to Celsius
var lst_celsius = lst_filtered.map(function(image) {
  return image.select('LST_Day_1km')
    .multiply(0.02)
    .subtract(273.15)
    .copyProperties(image, ['system:time_start']);
});

// Clip to ROI
var mean_lst = lst_celsius.mean().clip(roi);

// Resample to ERA5 grid (0.25°)
var resampled_lst = mean_lst.resample('bilinear').reproject({
  crs: 'EPSG:4326',
  scale: 27830
});

// Visualize
Map.addLayer(resampled_lst, {min: -10, max: 30, palette: ['blue', 'limegreen', 'yellow', 'red']}, 'MODIS LST');

// Export for comparison (e.g., to Google Drive)
Export.image.toDrive({
  image: resampled_lst,
  description: 'MODIS_LST_2023_01_01',
  folder: 'GEE_Exports',
  region: roi,
  scale: 27830,
  crs: 'EPSG:4326'
});