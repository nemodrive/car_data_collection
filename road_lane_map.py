import numpy as np
import utm
from docutils.nodes import image

MAP1_NAME = "high_resolution_image_full_top_view.png"
MAP1_SIZE = (15360, 17920)
MAP_REF_POINTS = [
    # ( (col, row), (lat, long) )
    ((482, 1560), (44.437456, 26.044567)),
    ((14658, 867), (44.437615, 26.049345)),
    ((2238, 12552), (44.434819, 26.045173)),
    ((15380, 13912), (44.434490, 26.049591)),
    ((9724, 6808), (44.436219, 26.047681)),
]


class ImageWgsHandler:
    def __init__(self, img_size, reference_points, ref_image_size):
        scale = 1.0
        img_size = (ref_image_size[0] * scale, ref_image_size[1] * scale)
        row_scale = img_size[0] / float(ref_image_size[0])
        col_scale = img_size[1] / float(ref_image_size[1])
        match_coord_wgs = []
        for (col, row), (lat, long) in reference_points:
            row_n = img_size[1] - row * row_scale
            col_n = col * col_scale
            easting, northing, zone_no, zone_letter = utm.from_latlon(lat, long)
            match_coord_wgs.append(((row_n, col_n), (easting, northing)))

        self.reference_points = reference_points = match_coord_wgs
        self.img_size = img_size
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
                (r1, c1), (e1, n1) = reference_points[i]
                (r2, c2), (e2, n2) = reference_points[j]
                rm = (c1+c2) / 2.
                cm = (r1+r2) / 2.
                em = (e1+e2) / 2.
                nm = (n1+n2) / 2.
                # pixels dist / wgs dist
                f = np.sqrt(((c1-c2)**2 + (r1-r2)**2) / ((e1-e2)**2 + (n1-n2)**2))
                f_x = (c2-c1)/(e2-e1)
                f_y = (r2-r1)/(n2-n1)
                d_info = ((i, j), (rm, cm), (em, nm), (f_x, f_y), f)
                density_points.append(d_info)
                if i not in d_points.keys():
                    d_points[i] = dict()
                d_points[i][j] = d_info

        return density_points

    def get_pixel(self, easting, northing):
        reference_points = self.reference_points
        density_points = self.density_points
        img_size = self.img_size

        w_r = 0
        w_c = 0
        p_r = 0
        p_c = 0

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

        new_r = p_r / w_r
        new_c = p_c / w_c
        img_size[1] - new_c
        return img_size[1]-new_c, new_r


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2

    # Show reference points
    x = [r1 for (c1, r1), (e1, n1) in MAP_REF_POINTS]
    y = [c1 for (c1, r1), (e1, n1) in MAP_REF_POINTS]
    n = range(len(x))
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()

    # Usage example
    image_handler = ImageWgsHandler((0, 0), MAP_REF_POINTS, MAP1_SIZE)

    latitude, longitude = 44.436827, 26.049155
    easting, northing, zone_no, zone_letter = utm.from_latlon(latitude, longitude)

    row, col = image_handler.get_pixel(easting, northing)

    print (row, col)

    # Show points on image
    map = cv2.imread(MAP1_NAME)
    MAP_VIEW_SIZE = 900
    RADIUS = 10000
    THICK = -1
    COLOR = (0, 0, 255)

    f = MAP_VIEW_SIZE / float(map.shape[0])

    map_view = cv2.resize(map, (0, 0), fx=f, fy=f)
    map_view = cv2.circle(map_view, (int(row), int(col)), radius=RADIUS, color=COLOR,
               thickness=THICK)

    cv2.imshow("Test", map_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




