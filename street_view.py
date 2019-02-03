#!/usr/bin/python
# GoogleMapDownloader.py 
# Created by Hayden Eskriett [http://eskriett.com]
#
# A script which when given a longitude, latitude and zoom level downloads a
# high resolution google map
# Find the associated blog post at: http://blog.eskriett.com/2013/07/19/downloading-google-maps/
# https://gis.stackexchange.com/a/42423
# 20 - (40075017/4294967296.0) * (2 ** (24-20)) *  np.cos(np.deg2rad(44.439457))
"""
decimal places	decimal degrees	N/S or E/W at equator
2	0.01	1.1132 km
3	0.001	111.32 m
4	0.0001	11.132 m
5	0.00001	1.1132 m

"""
import urllib
from PIL import Image
import os
import math
import numpy as np
import cv2
import pandas as pd
import utm
import matplotlib.pyplot as plt


class GoogleMapDownloader:
    """
      A class which generates high resolution google maps images given
      a longitude, latitude and zoom level
  """

    def __init__(self, lat, lng, zoom=12):
        """
        GoogleMapDownloader Constructor

        Args:
            lat:    The latitude of the location required
            lng:    The longitude of the location required
            zoom:   The zoom level of the location required, ranges from 0 - 23
                    defaults to 12
    """
        self._lat = lat
        self._lng = lng
        self._zoom = zoom

    def getXY(self):
        """
        Generates an X,Y tile coordinate based on the latitude, longitude
        and zoom level

        Returns:    An X,Y tile coordinate
    """

        tile_size = 256

        # Use a left shift to get the power of 2
        # i.e. a zoom level of 2 will have 2^2 = 4 tiles
        numTiles = 1 << self._zoom

        # Find the x_point given the longitude
        point_x = (tile_size / 2 + self._lng * tile_size / 360.0) * numTiles // tile_size

        # Convert the latitude to radians and take the sine
        sin_y = math.sin(self._lat * (math.pi / 180.0))

        # Calulate the y coorindate
        point_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_y) / (1 - sin_y)) * -(
                    tile_size / (2 * math.pi))) * numTiles // tile_size

        return int(point_x), int(point_y)

    def generateImage(self, **kwargs):
        """
        Generates an image by stitching a number of google map tiles together.

        Args:
            start_x:        The top-left x-tile coordinate
            start_y:        The top-left y-tile coordinate
            tile_width:     The number of tiles wide the image should be -
                            defaults to 5
            tile_height:    The number of tiles high the image should be -
                            defaults to 5
            type:           type of map "hybrid" or "standard"

        Returns:
            A high-resolution Goole Map image.
    """

        start_x = kwargs.get('start_x', None)
        start_y = kwargs.get('start_y', None)
        tile_width = kwargs.get('tile_width', 5)
        tile_height = kwargs.get('tile_height', 5)
        type = kwargs.get('type', "standard")

        # Check that we have x and y tile coordinates
        if start_x == None or start_y == None:
            start_x, start_y = self.getXY()

        # Determine the size of the image
        width, height = 256 * tile_width, 256 * tile_height

        # Create a new image of the size require
        map_img = Image.new('RGB', (width, height))
        print (tile_width, tile_height)
        for x in range(0, tile_width):
            for y in range(0, tile_height):
                if type == "hybrid":
                    url = 'https://mt0.google.com/vt/lyrs=y&hl=en&x=' + str(start_x + x) + '&y=' \
                          + str(start_y + y) + '&z=' + str(self._zoom)
                else:
                    url = 'https://mt0.google.com/vt?x='+str(start_x+x)+'&y='+str(start_y+y)+\
                          '&z='+str(self._zoom)
                print (x, y, url)
                current_tile = str(x) + '-' + str(y)
                urllib.request.urlretrieve(url, current_tile)

                im = Image.open(current_tile)
                map_img.paste(im, (x * 256, y * 256))

                os.remove(current_tile)

        return map_img


class ImageWgsHandler:
    def __init__(self, map_path):
        self.map_image = map_image = cv2.imread(map_path)
        self.img_rows, self.img_cols, _ = map_image.shape

        # Load reference points
        base = os.path.splitext(map_path)[0]
        self.reference_points = reference_points = pd.read_csv(f"{base}.csv")

        self.density = None
        if os.path.isfile(f"{base}.density"):
            with open(f"{base}.density", "r") as f:
                self.density = float(f.read().strip())

        if self.density:
            print(f"Map density: {self.density} m /pixel")

        reference_points = reference_points.assign(**{'easting': -1., 'northing': -1.,
                                                      "zone_no": -1., "zone_letter": ""})

        for idx, row in reference_points.iterrows():
            easting, northing, zone_no, zone_letter = utm.from_latlon(row["latitude"],
                                                                      row["longitude"])
            reference_points.at[idx, "easting"] = easting
            reference_points.at[idx, "northing"] = northing
            reference_points.at[idx, "zone_no"] = zone_no
            reference_points.at[idx, "zone_letter"] = zone_letter

        # # # -- Check conversion
        # img = plt.imread(map_path)
        # eastings, northings = reference_points.easting.values, reference_points.northing.values
        # rows = row_f.predict(np.column_stack([eastings, northings]))
        # cols = col_f.predict(np.column_stack([eastings, northings]))
        #
        # # rows = reference_points.pixel_row
        # # cols = reference_points.pixel_column
        #
        # fig = plt.figure()
        # right, top = img_cols, img_rows
        # plt.imshow(img, extent=[0, right, 0, top])
        # plt.scatter(cols, top-rows, s=1.5, c="r")
        # plt.axes().set_aspect('equal')

        (row_f, col_f), (easting_f, northing_f) = self.get_conversion_functions(reference_points)
        self.row_f, self.col_f = row_f, col_f
        self.easting_f, self.northing_f = easting_f, northing_f
        self.reference_points = reference_points

    @staticmethod
    def get_conversion_functions(reference_points):
        # -- Function conversion from WGS to pixel
        x = reference_points.easting.values
        y = reference_points.northing.values

        from sklearn import linear_model

        # classifiers = [
        #     svm.SVR(),
        #     linear_model.SGDRegressor(),
        #     linear_model.BayesianRidge(),
        #     linear_model.LassoLars(),
        #     linear_model.ARDRegression(),
        #     linear_model.PassiveAggressiveRegressor(),
        #     linear_model.TheilSenRegressor(),
        #     linear_model.LinearRegression()]
        #
        # z = reference_points.pixel_row.values
        #
        # for item in classifiers:
        #     print(item)
        #     clf = item
        #     clf.fit(np.column_stack([x, y]), z)
        #     print(np.abs(clf.predict(np.column_stack([x, y])) - z).sum(), '\n')

        z = reference_points.pixel_row.values
        row_f = linear_model.TheilSenRegressor()
        row_f.fit(np.column_stack([x, y]), z)

        z = reference_points.pixel_column.values
        col_f = linear_model.TheilSenRegressor()
        col_f.fit(np.column_stack([x, y]), z)

        # -- Function conversion from Pixels to wgs
        x = reference_points.pixel_row.values
        y = reference_points.pixel_column.values

        z = reference_points.easting.values
        easting_f = linear_model.LinearRegression()
        easting_f.fit(np.column_stack([x, y]), z)
        z = reference_points.northing.values
        northing_f = linear_model.LinearRegression()
        northing_f.fit(np.column_stack([x, y]), z)

        return (row_f, col_f), (easting_f, northing_f)

    def plot_wgs_coord(self, eastings, northings, padding=100, ax=None, show_image=True, c="r"):
        import time
        st = time.time()
        max_cols, max_rows = self.img_cols, self.img_rows
        img = self.map_image

        rows, cols = self.get_image_coord(eastings, northings)

        min_rows, max_rows = int(np.clip(rows.min() - padding, 0, max_rows)), \
                             int(np.clip(rows.max() + padding, 0, max_rows))
        min_cols, max_cols = int(np.clip(cols.min() - padding, 0, max_cols)), \
                             int(np.clip(cols.max() + padding, 0, max_cols))

        img_show = cv2.cvtColor(img[min_rows: max_rows, min_cols: max_cols], cv2.COLOR_BGR2RGB)

        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        if show_image:
            ax.imshow(img_show, extent=[min_cols, max_cols, max_rows, min_rows], aspect="equal")

        ax.scatter(cols, rows, s=1.5, c=c)

        return fig, ax

    def get_image_coord(self, eastings, northings):

        if self.density is not None:
            density = self.density
            ref_points = self.reference_points

            a = np.column_stack([eastings, northings])
            b = ref_points[["easting", "northing"]].values

            dist = np.linalg.norm(a[:, np.newaxis] - b, axis=2)
            ref = ref_points.iloc[dist.argmin(axis=1)]
            cols = (ref.pixel_column + (eastings - ref.easting)/density).values
            rows = (ref.pixel_row - (northings - ref.northing)/density).values
        else:
            row_f, col_f = self.row_f, self.col_f
            rows = row_f.predict(np.column_stack([eastings, northings]))
            cols = col_f.predict(np.column_stack([eastings, northings]))

        return rows, cols

    def get_wgs_coord(self, rows, cols):
        easting_f, northing_f = self.easting_f, self.northing_f

        easting = easting_f.predict(np.column_stack([rows, cols]))
        northing = northing_f.predict(np.column_stack([rows, cols]))
        return easting, northing


def main():
    import utm

    # Create a new instance of GoogleMap Downloader

    lat = 44.444122
    long = 26.042366
    scale = 20
    gmd = GoogleMapDownloader(lat, long, scale)

    print("The tile coorindates are {}".format(gmd.getXY()))
    exit(0)
    try:
        # Get the high resolution image
        img = gmd.generateImage(tile_width=38, tile_height=41, type="hybrid")
    except IOError:
        print(
            "Could not generate the image - try adjusting the zoom level and checking your coordinates")
    else:
        # Save the image to disk
        img.save("/media/andrei/CE04D7C504D7AF291/nemodrive/data_collect/high_resolution_image_full"
                 "_full3.png")
        print("The map has successfully been created")

    exit(0)

    # calculate pixel size
    equator_zoom_24 = 0.009330692
    scale_size = equator_zoom_24 * (2 ** (24-scale))
    pixel_size = np.cos(lat) * scale_size

    # WGS84 conversion from lat_lon GPS
    easting, northing, zone_no, zone_letter = utm.from_latlon(lat, long)

    easting += 2125 * pixel_size
    northing += 8853 * pixel_size
    new_lat, new_long = utm.to_latlon(easting, northing, zone_no, zone_letter)


if __name__ == "__main__":
    main()
