import sys
import numpy as np
from scipy import interpolate
import alphashape
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from descartes import PolygonPatch

from definitions import CONFIG_PATH_UTILS
sys.path.append(CONFIG_PATH_UTILS)
import utils as utils

from DustShape import SphericalBlub, PlaneDust

class LEImage:
    def __init__(self, LE_geometryanalyticalsource, surface, pixel_resolution = 0.2, cmap = 'magma_r'):
        self.pixel = pixel_resolution # arcsec
        self.cmap = cmap
        self.new_xs = utils.convert_ly_to_arcsec(LE_geometryanalyticalsource.d, LE_geometryanalyticalsource.x_projected)
        self.new_ys = utils.convert_ly_to_arcsec(LE_geometryanalyticalsource.d, LE_geometryanalyticalsource.y_projected)
        self.surface_original = surface
        self.surface_val = 0
        self.surface_img = 0

    def define_image_size(self, DustShape):
        if (isinstance(DustShape, SphericalBlub) or isinstance(DustShape, PlaneDust)):
            x_lim_min, x_lim_max = np.min(self.new_xs), np.max(self.new_xs)
            y_lim_min, y_lim_max = np.min(self.new_ys), np.max(self.new_ys)
        else:
            x_lim_min, x_lim_max = np.min(self.new_xs[0,:,:]) - 10 , np.max(self.new_xs[0,:,:]) + 10
            y_lim_min, y_lim_max = -np.max(self.new_ys[0,:,:]) - 10, np.max(self.new_ys[0,:,:]) + 10
        # print(x_lim_min, x_lim_max, y_lim_min, y_lim_max)

        x_tot_arcsec = np.abs(round((x_lim_max - x_lim_min),0))
        y_tot_arcsec = np.abs(round((y_lim_max - y_lim_min),0))
        print("size arcsec", np.max(self.new_ys), np.min(self.new_ys), x_tot_arcsec, y_tot_arcsec)

        x_size_img = int(x_tot_arcsec / self.pixel)
        y_size_img = int(y_tot_arcsec / self.pixel)
        print("size img pixels", x_size_img, y_size_img)
        x_all, y_all = np.meshgrid(np.linspace(x_lim_min, x_lim_max, x_size_img),
                np.linspace(y_lim_min, y_lim_max, y_size_img ))


        return x_size_img, y_size_img, x_all, y_all 
    
    def create_LE_img(self, x_size_img, y_size_img, x_all, y_all):
        self.surface_img = np.ones((x_all.shape[0], x_all.shape[1], 3), dtype=np.uint8) * 255
        # center = np.zeros([2])
        # print(np.nanmin(self.surface_val), np.nanmax(self.surface_val))
        surface_norm = ( self.surface_val - np.nanmin(self.surface_val)  ) / (np.nanmax(self.surface_val) - np.nanmin(self.surface_val))
        surface_norm[np.isnan(surface_norm)] = 0
        cmap = matplotlib.colormaps.get_cmap(self.cmap)
        normalize = matplotlib.colors.Normalize(vmin=np.nanmin(surface_norm), vmax=np.nanmax(surface_norm))
        for i in range(x_size_img):
            for j in range(y_size_img):
                color_rgb  = cmap(normalize(surface_norm[j,i])) * 255
                self.surface_img[j,i] = np.array(color_rgb[:3]) * 255
                if ((-0.5) < x_all[j,i] < (0.5)):
                    if ((-0.5) < y_all[j,i] < (0.5)):
                        # center = [i, j]
                        self.surface_img[j,i] = np.array([0,0,0]) * 255
        return self.surface_img

class LEImageAnalytical(LEImage):

    def __init__(self, LE_geometryanalyticalsource, geometry, surface, pixel_resolution = 0.2, cmap = 'magma_r'):
        super().__init__(LE_geometryanalyticalsource, surface, pixel_resolution, cmap)
        self.r_le_in = utils.convert_ly_to_arcsec(LE_geometryanalyticalsource.d, LE_geometryanalyticalsource.r_le_in)
        self.r_le_out = utils.convert_ly_to_arcsec(LE_geometryanalyticalsource.d, LE_geometryanalyticalsource.r_le_out)

        self.geometry_to_use = geometry
        self.act = LE_geometryanalyticalsource.ct * (self.geometry_to_use.eq_params[0] / self.geometry_to_use.eq_params[2]) if self.geometry_to_use.eq_params[2] != 0 else 0
        self.act = utils.convert_ly_to_arcsec(LE_geometryanalyticalsource.d, self.act)
        self.bct = LE_geometryanalyticalsource.ct * (self.geometry_to_use.eq_params[1] / self.geometry_to_use.eq_params[2]) if self.geometry_to_use.eq_params[2] != 0 else 0
        self.bct = utils.convert_ly_to_arcsec(LE_geometryanalyticalsource.d, self.bct)

        # self.max_points = int(self.new_xs.shape[2]/2)


    def interpolation_radiis(self):
        f_IN = interpolate.NearestNDInterpolator(list(zip(self.new_xs[0,1,:], self.new_ys[0,1,:])), self.r_le_in)
        f_OUT = interpolate.NearestNDInterpolator(list(zip(self.new_xs[0,0,:], self.new_ys[0,0,:])), self.r_le_out)

        return f_IN, f_OUT

    def interpolation_surface(self):
        surface_inter_y = interpolate.LinearNDInterpolator(list(zip(self.new_xs[0,0,:], self.new_ys[0,0,:])), self.surface_original)
        return surface_inter_y
    
    def create_le_surface(self):

        x_img = []
        y_img = []
        R_IN, R_OUT = self.interpolation_radiis()
        surface_inter_y  = self.interpolation_surface()
        x_size_img, y_size_img, x_all, y_all = self.define_image_size(self.geometry_to_use)
        self.surface_val = np.zeros([y_size_img, x_size_img])

        for i in range(x_size_img):
            for j in range(y_size_img):
                # print(j,i)
                if (R_IN(x_all[j,i], y_all[j,i]) <= np.sqrt((x_all[j,i] + self.act)**2 + (y_all[j,i] + self.bct)**2) <= R_OUT(x_all[j,i], y_all[j,i])):
                    # print(j,i)
                    x_img.append(x_all[j,i])
                    y_img.append(y_all[j,i])
                    self.surface_val[j,i] = surface_inter_y(x_all[j,i], y_all[j,i])

        self.surface_img = self.create_LE_img(x_size_img, y_size_img, x_all, y_all)

        return self.surface_val, self.surface_img, x_img, y_img
    

class LEImageNonAnalytical(LEImage):
    
    def __init__(self, LE_planedust1source1, geometry, surface, pixel_resolution=0.2, cmap='magma_r'):
        super().__init__(LE_planedust1source1, surface, pixel_resolution, cmap)
        self.surface_img = 0
        self.geometry_to_use = geometry
        flat = LE_planedust1source1.xy_matrix[:, :, :2].reshape(self.geometry_to_use.side_x*self.geometry_to_use.side_y, 2)
        cord = np.array([(xf, yf) for xf, yf in flat if xf != 0 or yf != 0])
        self.new_xs = utils.convert_ly_to_arcsec(LE_planedust1source1.d, cord[:, 0])
        self.new_ys = utils.convert_ly_to_arcsec(LE_planedust1source1.d, cord[:, 1])


        flat_sb = surface.reshape(surface.shape[0]*surface.shape[1])
        self.surface_original = flat_sb[flat[:,0]!=0]

    def define_shape_to_interpolate(self, show_shape=False):

        points = list(zip(self.new_xs, self.new_ys))
        index = [i[0] for i in sorted(enumerate(points), key=lambda x: [x[1][1], x[1][0]])]
        arr_points = np.array(points)
        mpoints = [(X, Y) for X, Y in list(zip(arr_points[index, 0], arr_points[index, 1]))]
        alpha_shape = alphashape.alphashape(mpoints, alpha=0.3) #0.2
        # print(alpha_shape)
        if alpha_shape.geom_type == 'MultiPolygon':
            geoms = [g for g in alpha_shape.geoms]
        if show_shape == True:
            fig, ax = plt.subplots()
            ax.scatter(*zip(*mpoints[::100]), alpha=0.2)
            if alpha_shape.geom_type == 'MultiPolygon':
                print("it is multipolygon")
                for i in geoms:
                    ax.add_patch(PolygonPatch(i, alpha=0.3))
            else:
                ax.add_patch(PolygonPatch(alpha_shape, alpha=0.3))

            plt.show()
        # print(self.surface_original[index])
        surface_inter_y = utils.interpolation_surface(arr_points[index, 0], arr_points[index, 1], self.surface_original[index])

        return alpha_shape, surface_inter_y
    

    def create_le_surface(self, show_shape=False):
        x_size_img, y_size_img, x_all, y_all = self.define_image_size(self.geometry_to_use)
        alpha_shape, surface_inter_y = self.define_shape_to_interpolate(show_shape)
        self.surface_val = np.zeros([y_size_img, x_size_img])

        x_img = []
        y_img = []
        poly = alpha_shape
        
        self.surface_val = np.zeros([y_size_img, x_size_img])
        for i in range(x_size_img):
            for j in range(y_size_img):
                if not np.isnan(self.surface_val[j,i]): #UPDATE: no need to test pts that are already NaN
                    p1 = Point(x_all[j,i], y_all[j,i])
                    test = poly.contains(p1) | poly.touches(p1)
                    if test == False:
                        # print("false")
                        self.surface_val[j,i] = "Nan"
                    else:
                        x_img.append(x_all[j,i])
                        y_img.append(y_all[j,i])
                        # print("true")
                        self.surface_val[j,i] = surface_inter_y(x_all[j,i], y_all[j,i])

        # print(self.surface_val)
        self.surface_img = self.create_LE_img(x_size_img, y_size_img, x_all, y_all)

        return self.surface_val, self.surface_img, x_img, y_img