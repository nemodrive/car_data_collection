#!/usr/bin/python
# GoogleMapDownloader.py 
# Created by Hayden Eskriett [http://eskriett.com]
#
# A script which when given a longitude, latitude and zoom level downloads a
# high resolution google map
# Find the associated blog post at: http://blog.eskriett.com/2013/07/19/downloading-google-maps/
# https://gis.stackexchange.com/questions/7430/what-ratio-scales-do-google-maps-zoom-levels-correspond-to
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
        Returns:
            A high-resolution Goole Map image.
    """

        start_x = kwargs.get('start_x', None)
        start_y = kwargs.get('start_y', None)
        tile_width = kwargs.get('tile_width', 5)
        tile_height = kwargs.get('tile_height', 5)

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
                # url = 'https://mt0.google.com/vt/lyrs=y&hl=en&x=' + str(start_x + x) + '&y=' + str(
                #     start_y + y) + '&z=' + str(self._zoom)
                url = 'https://mt0.google.com/vt?x='+str(start_x+x)+'&y='+str(start_y+y)+'&z='+str(
                    self._zoom)
                print (x, y, url)
                current_tile = str(x) + '-' + str(y)
                urllib.urlretrieve(url, current_tile)

                im = Image.open(current_tile)
                map_img.paste(im, (x * 256, y * 256))

                os.remove(current_tile)

        return map_img


class ImageWgsHandler:
    def __init__(self, img_size, reference_points):
        self.reference_points = reference_points

        self.density_points = self.wgs_pixel_transform_matrix(img_size, reference_points)

    def wgs_pixel_transform_matrix(self, img_size, reference_points):
        """
        Calculate WGS coordinate for each pixels according to a weighted average of reference points
        :param img_size: (width, height) in pixels
        :param reference_points: Lis[ ( (row, col), (easting*, northing*) ), ....] *WGS84 format
        :return:
        """
        img_width, img_height = img_size
        max_distance = float(img_width ** 2 + img_height ** 2)

        density_points = []  # list of (col, row), (easting, northing), pixel_wgs_factor
        d_points = dict({})
        for i in range(len(reference_points)-1):
            for j in range(i+1, len(reference_points)):
                (c1, r1), (e1, n1) = reference_points[i]
                (c2, r2), (e2, n2) = reference_points[j]
                rm = (r1+r2) / 2.
                cm = (c1+c2) / 2.
                em = (e1+e2) / 2.
                nm = (n1+n2) / 2.
                # pixels dist / wgs dist
                f = np.sqrt( ((r1-r2)**2 + (c1-c2)**2) / ((e1-e2)**2 + (n1-n2)**2) )
                f_x = (r2-r1)/(e2-e1)
                f_y = (c2-c1)/(n2-n1)
                d_info = ((i, j), (rm, cm), (em, nm), (f_x, f_y), f)
                density_points.append(d_info)
                if i not in d_points.keys():
                    d_points[i] = dict()
                d_points[i][j] = d_info

        return density_points

        # max_points = 3
        # r, c, e, n = [], [], [], []
        # for (r1, c1), (e1, n1) in reference_points:
        #     r.append(r1)
        #     c.append(c1)
        #     e.append(e1)
        #     n.append(n1)
        #
        # r = np.array(r)
        # c = np.array(c)
        # e = np.array(e)
        # n = np.array(n)

        # # wgs 2 pixel
        # lat_, long_ = 44.436336, 26.046967
        # easting, northing, zone_no, zone_letter = utm.from_latlon(lat_, long_)
        #
        #
        # de = e - easting
        # se = np.argsort(de)
        # left_closest = se[de<0][-max_points:]
        # right_closest = se[de>0][:max_points]
        #
        # e[right_closest]
        #
        #
        # # wgs 2 pixel
        # lat_, long_ = 44.435716, 26.045121
        # easting, northing, zone_no, zone_letter = utm.from_latlon(lat_, long_)

    def get_pixel(self, easting, northing):
        reference_points = self.reference_points
        density_points = self.density_points

        w_r = 0
        w_c = 0
        p_r = 0
        p_c = 0
        cnt = 5

        for (i, j), (ref_r, ref_c), (ref_e, ref_n), (p_w_x, p_w_y), p_w in density_points:
            (c1, r1), (e1, n1) = reference_points[i]
            (c2, r2), (e2, n2) = reference_points[j]

            r = p_w_x * (easting - ref_e) + ref_r
            c = p_w_y * (northing - ref_n) + ref_c

            dist = (easting - ref_e) ** 2 + (northing - ref_n) ** 2

            # Must calculate if between points or not -> depends how we weigh dif in points
            d_e = abs(e1-e2)
            if np.sign(easting - e1) == np.sign(easting - e2):
                # same side
                w_r_ = 1/dist * d_e
            else:
                w_r_ = 1/dist * (1/d_e)
            w_r += w_r_
            p_r += r * w_r_

            d_n = abs(n1-n2)
            if np.sign(northing - n1) == np.sign(northing - n2):
                # same side
                w_c_ = 1/(northing - ref_n) * d_n
            else:
                w_c_ = 1/(northing - ref_n) * (1/d_n)
            w_c += w_c_
            p_c += c * w_c_


            cnt += 1

        new_r = p_r / w_r
        new_c = p_c / w_c
        return new_r, new_c

    # lat_, long_ = 44.435716, 26.045121
    # easting, northing, zone_no, zone_letter = utm.from_latlon(lat_, long_)


def main():
    import utm

    # Create a new instance of GoogleMap Downloader

    lat = 44.437774
    long = 26.044453
    scale = 22
    gmd = GoogleMapDownloader(lat, long, scale)

    print("The tile coorindates are {}".format(gmd.getXY()))

    try:
        # Get the high resolution image
        img = gmd.generateImage(tile_width=70, tile_height=60)
    except IOError:
        print(
            "Could not generate the image - try adjusting the zoom level and checking your coordinates")
    else:
        # Save the image to disk
        img.save("/media/andrei/CE04D7C504D7AF291/nemodrive/data_collect/high_resolution_image_full"
                 ".png")
        print("The map has successfully been created")


    # calculate pixel size
    equator_zoom_24 = 0.009330692
    scale_size = equator_zoom_24 * (2 ** (24-scale))
    pixel_size = np.cos(lat) * scale_size


    # WGS84 conversion from lat_lon GPS
    easting, northing, zone_no, zone_letter = utm.from_latlon(lat, long)
    easting, northing, zone_no, zone_letter = utm.from_latlon(lat, long)

    easting += 2125 * pixel_size
    northing += 8853 * pixel_size
    new_lat, new_long = utm.to_latlon(easting, northing, zone_no, zone_letter)

    #
    orig_img_size = (15360, 17920)
    match_coord = [
        # ( (col, row), (lat, long) )
        ((482, 1560), (44.437456, 26.044567)),
        ((14658, 867), (44.437615, 26.049345)),
        ((2238, 12552), (44.434819, 26.045173)),
        ((15380, 13912), (44.434490, 26.049591)),
        ((9724, 6808), (44.436219, 26.047681)),
    ]

    scale = 1.0
    img_size = (orig_img_size[0]*scale, orig_img_size[1]*scale)
    row_scale = img_size[0] / float(orig_img_size[0])
    col_scale = img_size[1] / float(orig_img_size[1])
    match_coord_wgs = []
    for (col, row), (lat, long) in match_coord:
        row_n = img_size[1] - row * row_scale
        col_n = col * col_scale
        easting, northing, zone_no, zone_letter = utm.from_latlon(lat, long)
        match_coord_wgs.append(((row_n, col_n), (easting, northing)))

    reference_points = match_coord_wgs

    import matplotlib.pyplot as plt
    x = [r1 for (c1, r1), (e1, n1) in reference_points]
    y = [c1 for (c1, r1), (e1, n1) in reference_points]
    n = range(len(x))
    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()

    x = [e1 for (r1, c1), (e1, n1) in reference_points]
    y = [n1 for (r1, c1), (e1, n1) in reference_points]
    n = range(len(x))
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))

    plt.show()

    pixels_to_wgs = wgs_pixel_transform_matrix(img_size, match_coord_wgs)



