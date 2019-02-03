Generated with
latitude_top_left = 44.444122
longitude_top_left = 26.042366
scale = 20
gmd = GoogleMapDownloader(lat, long, scale)
img = gmd.generateImage(tile_width=38, tile_height=41, type="hybrid")

image pixel density 
zoom = 20
center_latitude = 44.439457

(40075017/4294967296.0) * (2 ** (24-zoom)) *  np.cos(np.deg2rad(center_latitude))
-> 0.10659243462677055