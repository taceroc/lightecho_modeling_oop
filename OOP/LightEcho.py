import sys
from definitions import CONFIG_PATH_UTILS, CONFIG_PATH_constants
sys.path.append(CONFIG_PATH_UTILS)
import numpy as np
from scipy import interpolate
import utils as utils
sys.path.append(CONFIG_PATH_constants+"\\code")
import var_constants as vc
import fix_constants as fc
import dust_constants as dc
from numba import jit
import plotly
import plotly.io as pio
# pio.renderers.default = "iframe"
import plotly.graph_objs as go

class LE:
    """
        LE define the properties of the Light Echo
        Subclasses:
            LE_plane
            LE_sphere_centered
    """
    def __init__(self, ct, source):
        """
            x: initialize x positions in ly
            ct: time of LE observation in years
        """
        self.x = 0
        self.ct = ct
        self.r_le2 = 0
        self.x_inter_values = 0
        self.y_inter_values = 0
        self.z_inter_values = 0
        self.x_projected = 0
        self.y_projected = 0
        self.dt0 = source.dt0
        self.d = source.d   
        
    def calculate_intersection_x_y(self, y_1):
        """
            Calculate the intersection points x,y between a DustShape and the paraboloid
        """
        y_2 = -1*y_1
        # keep no nan values
        y_inter = np.hstack((y_1, y_2))
        self.y_inter_values = y_inter[~np.isnan(y_inter)]
        print(self.y_inter_values.shape)
        # extract x where y is no nan
        x_inv_nan = np.hstack((self.x, self.x.copy()))
        self.x_inter_values = x_inv_nan[~np.isnan(y_inter)]
        print(self.x_inter_values.shape)
        
    def calculate_rho_thickness(self):
        """
            Calculate the rho in the sky plane (xy) and thickness according to Sugermann 2003 and Xi 1994
            and calculate the radius out and radius min according to that thickness
        """
        r_le = np.sqrt(self.r_le2)
        rhos = np.sqrt(2 * self.z_inter_values * self.ct + (self.ct)**2 )
        half_obs_thickness = np.sqrt( (self.ct / rhos) ** 2 * self.dz0 ** 2 + ( (rhos * fc.c / (2 * self.ct)) + ( fc.c * self.ct / (2 * rhos) )) ** 2 * self.dt0  ** 2 ) / 2
        # -- include the thickness in xy plane
        self.r_le_out = r_le + half_obs_thickness
        self.r_le_in = r_le - half_obs_thickness
        
    def final_xy_projected(self):
        """
        Calculate the x,y points in arcseconds
        Only valid when the dust and the paraboloid have a analytical expresion (and the analtyical expression is a circumference)

        Arguments:
            phis: angle in the sky plane
            r_le_out, r_le_in: out and inner radii in arcsec
            act: center of LE in arcsec

        Returns:
            new_xs, new_ys: x,y position in the x-y plane in arcseconds
        """
        self.calculate_rho_thickness()
        phis = np.arctan2(self.y_inter_values, self.x_inter_values)
        print(phis.shape)        
        radii_p = [self.r_le_out, self.r_le_in]

        xs_p = np.concatenate([radii_p[0] * np.cos(phis) - (self.A/self.F)*self.ct, radii_p[1] * np.cos(phis) - (self.A/self.F)*self.ct]).reshape(2, len(phis))
        print("b", self.B)
        ys_p = np.concatenate([radii_p[0] * np.sin(phis) - (self.B/self.F)*self.ct, radii_p[1] * np.sin(phis) - (self.B/self.F)*self.ct]).reshape(2, len(phis))

        self.x_projected = xs_p.reshape(1,2,len(phis))
        self.y_projected = ys_p.reshape(1,2,len(phis))

        return self.x_projected, self.y_projected
    
    def get_xy_projected(self):
        """
            Return x,y,z intersection in ly and x,y projected in the sky plane in arcseconds
        """
        self.x_inter_values, self.y_inter_values, self.z_inter_values = self.get_intersection_xyz()
        self.x_projected, self.y_projected = super().final_xy_projected()
        
        return self.x_inter_values, self.y_inter_values, self.z_inter_values, self.x_projected, self.y_projected
    

class LEPlane(LE):
    """
        LE_plane(LE) defines a subclass of LE
            LE_plane
    """
    def __init__(self, ct, plane, source):
        """
            x: initialize x positions in ly
            ct: time of LE observation in years
            plane.eq_params = [A, B, C, D], Ax + By + Fz + D = 0
            plane.dz0: depth inf plane of dust in ly
            
        """
        super().__init__(ct, source)
        self.A = plane.eq_params[0]
        self.B = plane.eq_params[1]
        self.F = plane.eq_params[2]
        self.D = plane.eq_params[3]
        self.dz0 = plane.dz0
        self.r_le_out = 0
        self.r_le_in = 0
        self.check_time_forLE()

    def check_time_forLE(self):
        """
            If the plane is behind the source check that the ct time is later than the start of the LE
        """
        if not isinstance(self.D, (np.ndarray, list)):
            print(True)
            if -(self.D/self.F) < 0:
                ti = (2 * (self.D/self.F))/(fc.c * (1 + (self.A/self.F)**2 + (self.B/self.F)**2))
                if ti >= self.ct:
                    print(f"There is no LE at {self.ct} years, LE starts to expand at {ti/fc.dtoy} days or {ti} years")
                    raise ValueError('No LE yet')
        
    def calculate_rle2(self):
        """
            Calcualte the radii square of the resultant LE plane
        """
        self.r_le2 = -2 * self.ct * (self.D/self.F) + (self.ct)**2 * (1 + (self.B/self.F)**2 + (self.A/self.F)**2)
        return self.r_le2
    
    def get_intersection_xyz(self):
        """
            Calculate the intersection points x,y,z between a DustShape and the paraboloid
        """
        self.calculate_rle2()
        theta_p = np.linspace(0, 2*np.pi, 1000)
        self.x_inter_values = np.sqrt(self.r_le2) * np.cos(theta_p) - (self.A/self.F)*self.ct
        self.y_inter_values = np.sqrt(self.r_le2) * np.sin(theta_p) - (self.B/self.F)*self.ct
        # calculate z = z0 - ax >> plane equation
        self.z_inter_values = -(self.D/self.F) - (self.A/self.F) * self.x_inter_values - (self.B/self.F) * self.y_inter_values

        return self.x_inter_values, self.y_inter_values, self.z_inter_values
        

class LESphereCentered(LE):
    """
        LE_sphere_centered(LE) defines a subclass of LE
            LE_sphere_centered
    """
    def __init__(self, ct, sphere, source):
        """
            x: initialize x positions in ly
            ct: time of LE observation in years
            sphere.eq_params = [A=0, B=0, F=0, D], (x)2 + y2 + z2 = D2
            sphere.dz0: depth sphere of dust in ly
            
        """
        super().__init__(ct, source)
        self.A = sphere.eq_params[0]
        self.B = sphere.eq_params[1]
        self.F = sphere.eq_params[2]
        self.D = sphere.eq_params[3]
        self.dz0 = sphere.dz0
        self.r_le_out = 0
        self.r_le_in = 0
        
    def calculate_rle2(self):
        """
            Calcualte the radii square of the resultant LE sphere
        """
        self.r_le2 = 2 * self.D * self.ct - (self.ct)**2
        return self.r_le2
    
    def determine_initial_x(self):
        """
            Intialize x values according to geomtry and time
        """
        r_le = np.sqrt(self.r_le2)
        xmin = -r_le
        xmax = r_le
        self.x = np.linspace(xmin,xmax,1000)
        
    def get_intersection_xyz(self):
        """
            Calculate the intersection points x,y,z between a DustShape and the paraboloid
        """
        self.calculate_rle2()
        self.determine_initial_x()
        theta_p = np.linspace(0, 2*np.pi, 1000)
        self.x_inter_values = np.sqrt(self.r_le2) * np.cos(theta_p)
        self.y_inter_values = np.sqrt(self.r_le2) * np.sin(theta_p)
        # # calculate z2 = r2 - x2 - y2 
        self.z_inter_values = np.sqrt(self.D**2 - self.x_inter_values**2 - self.y_inter_values**2)
        return self.x_inter_values, self.y_inter_values, self.z_inter_values
    

class LESphericalBulb(LE):
    def __init__(self, ct, sphericalblub, source):
        """
            ct: time of LE observation in years
            sphericalbulb.eq_params = (x-A)2 + (y-B)2 + (z-F)2 = r2
            sphere.dz0: depth sphere of dust in ly
            
        """
        super().__init__(ct, source)
        self.h = sphericalblub.eq_params[0]
        self.k = sphericalblub.eq_params[1]
        self.l = sphericalblub.eq_params[2]
        self.r0ly = sphericalblub.eq_params[3]
        self.dz0 = sphericalblub.dz0
        self.xy_matrix = []
        self.z_matrix = []
        self.pixel_xs_true = []
        self.pixel_ys_true = []
        self.pixel_zs_true = []
    
    def calculate_rle2(self):
        """
            Calcualte the radii square of the resultant LE sphere
        """
        self.r_le2 = 100 #******************
        return self.r_le2
    
    def determine_initial_x(self):
        """
            Intialize x values according to geomtry and time
        """
        r_le = np.sqrt(self.r_le2)
        xmin = -r_le
        xmax = r_le
        self.x = np.linspace(xmin,xmax,1000)

    def get_intersection_xyz(self):
        """
            Calculate the intersection points x,y,z between a DustShape and the paraboloid
        """
        self.calculate_rle2()
        self.determine_initial_x()
        self.x, self.y = np.meshgrid(self.x, self.x)
        self.z = (self.x**2 + self.y**2 - self.ct**2) / (2 * self.ct)

        intersection_points = utils.cal_inter(self.x, self.y, self.z, self.h, self.k, self.l, self.ct, self.r0ly, self.dt0)

        self.x_inter_values = np.array([inter[0] for inter in intersection_points])
        self.y_inter_values = np.array([inter[1] for inter in intersection_points])
        self.z_inter_values = np.array([inter[2] for inter in intersection_points])

        return self.x_inter_values, self.y_inter_values, self.z_inter_values
    
    
    def XYZ_merge_bulb(self, size_x=100, size_y=100, size_z=100):
        
        """
            Correspond a xyz intersection with a value of the dust sheet image
        """
        #index the x,y,z values
        #create arrays with a size of int(sqrt(len unique arrays))
        x_bins = np.linspace(np.nanmin(self.x_inter_values), np.nanmax(self.x_inter_values), size_x)
        y_bins = np.linspace(np.nanmin(self.y_inter_values), np.nanmax(self.y_inter_values), size_y)
        
        #what a great function, return an array of size=original and the values are the bin at which that value belong
        self.pixel_xs_true = np.digitize(self.x_inter_values, x_bins)
        self.pixel_ys_true = np.digitize(self.y_inter_values, y_bins)

        #same for z, but of size count
        z_bins = np.linspace(np.nanmin(self.z_inter_values), np.nanmax(self.z_inter_values), size_z)
        self.pixel_zs_true = np.digitize(self.z_inter_values, z_bins)

        pairs = np.vstack([self.pixel_xs_true, self.pixel_ys_true]).T
        unique_pairs, ind, inverse, count = np.unique(pairs, axis=0, return_counts=True, return_index=True, return_inverse=True)

        self.xy_matrix = np.zeros((size_y, size_x, 2))
        self.z_matrix = np.zeros((size_y, size_x, size_z))

        for i, up in enumerate(unique_pairs):
            inds = np.where(self.pixel_xs_true == up[0])[0]
            inds5 = np.where(self.pixel_ys_true == up[1])[0] 
            xind = np.intersect1d(inds5, inds)
            zi = self.pixel_zs_true[xind]
            x_ind = up[0] - 1
            y_ind = up[1] - 1
            self.xy_matrix[y_ind, x_ind, :] = x_bins[x_ind], y_bins[y_ind]
            val = np.unique(zi)
            for j in val:
                self.z_matrix[y_ind, x_ind, j-1] = z_bins[j-1]

        return self.xy_matrix, self.z_matrix

    def plot(self):
        """
            Plot Paraboloid + Sphere + intersection points 
            in 3D and interactive
        """

        theta = np.linspace(0, np.pi, 100)
        phi = np.linspace(0, 2*np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)
        fig = go.Figure()
        fig.add_trace(go.Surface(
                x=self.x,
                y=self.y,
                z=self.z, 
                showlegend = False,
                colorscale ='Blues',
                opacity = 0.4,
            ))
        fig.add_trace(go.Surface(
                x=self.r0ly * np.sin(theta) * np.cos(phi) + self.h,
                y=self.r0ly * np.sin(theta) * np.sin(phi) + self.k,
                z=self.r0ly * np.cos(theta) + self.l,
                showlegend = False,
                opacity = 0.4,
            ))
        fig.add_trace(go.Scatter3d(
                x=self.x_inter_p_new ,
                y=self.y_inter_p_new ,
                z=self.z_inter_p_new , 
                marker={'size': 3,
                        'opacity': 0.8}, showlegend = False
            ))
        fig.show()

class LEPlaneDust(LE):
    def __init__(self, ct, planedust, source):
        """
            x: initialize x positions in ly
            ct: time of LE observation in years
            sphere.eq_params = [A, B, C, D], (x)2 + y2 + z2 = D2
            sphere.dz0: depth sphere of dust in ly
            
        """
        super().__init__(ct, source)
        self.A = planedust.eq_params[0]
        self.B = planedust.eq_params[1]
        self.F = planedust.eq_params[2]
        self.D = planedust.eq_params[3]
        self.r_le_out = 0
        self.r_le_in = 0
        self.check_time_forLE()
        self.dz0 = planedust.dz0
        self.xy_matrix = []
        self.pixel_xs_true = []
        self.pixel_ys_true = []
        
    
    def check_time_forLE(self):
        """
            If the plane is behind the source check that the ct time is later than the start of the LE
        """
        if -(self.D/self.F) < 0:
            ti = (2 * (self.D/self.F))/(fc.c * (1 + (self.A/self.F)**2 + (self.B/self.F)**2))
            if ti >= self.ct:
                print(f"There is no LE at {self.ct} years, LE starts to expand at {ti/fc.dtoy} days or {ti} years")
                # raise ValueError('No LE yet')

    
    def calculate_rle2(self):
        """
            Calcualte the radii square of the resultant LE sphere
        """
        # Intersection paraboloid and plane give the LE radii
        self.r_le2 = -2 * self.ct * (self.D/self.F) + (self.ct)**2 * (1 + (self.B/self.F)**2 + (self.A/self.F)**2)
        return self.r_le2
    
    def get_intersection_xyz(self, cube):
        """
            Return x,y,z intersection in ly 
        """
        self.calculate_rle2()
        theta_p = np.linspace(0, 2*np.pi, 1000)

        self.x_inter_values = np.sqrt(self.r_le2) * np.cos(theta_p) - (self.A/self.F)*self.ct
        self.y_inter_values = np.sqrt(self.r_le2) * np.sin(theta_p) - (self.B/self.F)*self.ct
        # calculate the z intersections for each "infinitesimal" plane
        self.z_inter_values = -(self.D/self.F) - (self.A * self.x_inter_values / self.F) - (self.B * self.y_inter_values / self.F)

        x_lim_min, x_lim_max = np.min(self.x_inter_values) , np.max(self.x_inter_values)
        y_lim_min, y_lim_max = np.min(self.y_inter_values), np.max(self.y_inter_values)
        # print("x_lim_min, x_lim_max")
        # print(y_lim_min, y_lim_max)
        act = float(self.ct * (self.A/self.F))
        bct = float(self.ct * (self.B/self.F))
        x_all, y_all = np.meshgrid(np.linspace(x_lim_min, x_lim_max, 1000),
                np.linspace(y_lim_min, y_lim_max, 1000))
        def interpolation_radiis():
            f_IN = interpolate.NearestNDInterpolator(list(zip(self.x_inter_values, self.y_inter_values)), self.r_le_in)
            f_OUT = interpolate.NearestNDInterpolator(list(zip(self.x_inter_values, self.y_inter_values)), self.r_le_out)
            return f_IN, f_OUT
        
        z_all = -(self.D/self.F) - (self.A * x_all / self.F) - (self.B * y_all / self.F)
        R_IN, R_OUT = interpolation_radiis()
        r_in = R_IN(x_all, y_all)
        r_out = R_OUT(x_all, y_all)

        intersection_points = utils.cal_inter_planedust(x_all, y_all, z_all, 
                                                        r_in, r_out, act, bct, cube.x_min, cube.x_max, cube.y_min, cube.y_max, cube.z_min, cube.z_max,
                                                        [self.A, self.B, self.F, self.D])

        self.x_inter_values = np.concatenate([intersection_points[0]])
        self.y_inter_values = np.concatenate([intersection_points[1]])
        self.z_inter_values = np.concatenate([intersection_points[2]])
        # self.x_projected, self.y_projected = super().final_xy_projected()
        
        return self.x_inter_values, self.y_inter_values, self.z_inter_values

            
    
    def XYZ_merge_plane_2ddust(self, planedust, dust_cube_img):
        """
        Given the x,y,z intersection keep only the position that activate a pixel that has a true value

        Arguments:
            Xinter, Yinter, Zinter: intersection plane and paraboloid that fall inside the dust 2d data
            index_plane: indexes where the x,y,z values are valid
            pixel_xs_true, pixel_ys_true: pixels from the dust 2d that are activated
            dust_cube_test: 3d cube of dust. The cube is a boolean

        Return:
            x_inter_values, y_inter_values, z_inter_values: intersection plane and paraboloid that fall inside the dust 2d data and where
                                                        the value of the pixel is true
        """

        pixel_x = ((self.x_inter_values - planedust.x_min) / (planedust.x_max -  planedust.x_min)) * (planedust.side_x - 1)
        pixel_y = ((self.y_inter_values - planedust.y_min) / (planedust.y_max -  planedust.y_min)) * (planedust.side_y - 1)
        # print("LEplanedust")
        # print(x_inter_p_new)
        self.pixel_xs_true = np.round(pixel_x).astype(int) + 1
        self.pixel_ys_true = np.round(pixel_y).astype(int) + 1

        print("pixel", self.pixel_xs_true.shape, self.pixel_ys_true.shape)
        pairs = np.vstack([self.pixel_xs_true, self.pixel_ys_true]).T
        unique_pairs, ind, inverse, count = np.unique(pairs, axis=0, return_counts=True, return_index=True, return_inverse=True)
        self.xy_matrix = np.zeros((planedust.side_y, planedust.side_x, 4))

        x_bins = np.linspace(np.nanmin(self.x_inter_values), np.nanmax(self.x_inter_values), planedust.side_x)
        y_bins = np.linspace(np.nanmin(self.y_inter_values), np.nanmax(self.y_inter_values), planedust.side_y)
        # print(x_bins)
        for i, up in enumerate(unique_pairs):
            x_ind = up[0] - 1
            y_ind = up[1] - 1
            # print(x_ind, y_ind, x_bins[x_ind], y_bins[y_ind])
            zi = -(self.D/self.F) - (self.A * x_bins[x_ind] / self.F) - (self.B * y_bins[y_ind] / self.F)
            self.xy_matrix[y_ind, x_ind, :] = x_bins[x_ind], y_bins[y_ind], zi, dust_cube_img[y_ind, x_ind]

        return self.xy_matrix
    
    def plot(self, planedust):
        figs = go.Figure()
        figs.add_trace(go.Scatter3d(
            x = [planedust.x_min, planedust.x_max, planedust.x_max, planedust.x_min, planedust.x_min, planedust.x_max, planedust.x_max, planedust.x_min],
            y = [planedust.y_min, planedust.y_min, planedust.y_max, planedust.y_max, planedust.y_max, planedust.y_max, planedust.y_min, planedust.y_min,],
            z = [planedust.z_min, planedust.z_min, planedust.z_min, planedust.z_min, planedust.z_max, planedust.z_max, planedust.z_max, planedust.z_max],
              showlegend = False))
        figs.add_trace(go.Scatter3d(
            x=self.x_inter_values,
            y=self.y_inter_values,
            z=self.z_inter_values,
            opacity=0.2))
        figs.add_trace(go.Surface(
            z = (-self.A * self.x_inter_values - self.B * self.y_inter_values - self.D) / self.F,
            opacity=0.2))
        figs.show()