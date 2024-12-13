library("exactextractr")
library("sf")
library("raster")

# this is a large file and can be downloaded from https://www.earthdata.nasa.gov/data/catalog/sedac-ciesin-sedac-gpwv4-popcount-r11-4.11
pop_raster = raster("../data/gpw-v4-population-count-rev11_2005_30_sec_tif/gpw_v4_population_count_rev11_2005_30_sec.tif")

# this is a large file and can be downloaded from https://gadm.org/data.html
reg_raster = st_read("../data/gadm36_levels.gpkg", layer=1)

out = exact_extract(pop_raster, reg_raster, fun="mean")
write.csv(out, "../data/pop_by_region.csv")