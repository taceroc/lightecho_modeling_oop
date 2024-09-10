import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import scipy.stats as stats
from definitions import CONFIG_PATH_DUST, PATH_TO_DUST_IMG_CODE, CONFIG_PATH_UTILS, PATH_TO_RESULTS_SIMULATIONS, PATH_TO_RESULTS_FIGURES, CUBE_NAME
sys.path.append(PATH_TO_DUST_IMG_CODE)
import generate_cube_dust as gcd
sys.path.append(CONFIG_PATH_DUST)
import var_constants as vc
import fix_constants as fc
import dust_constants as dc
# sys.path.append(ROOT_DIR)
from DustShape import PlaneDust
import LightEcho as LE #import LEPlane, LESphereCentered, LESphericalBulb, LE_PlaneDust
from Source import Source
import SurfaceBrightness as sb
from LE_img import LEImageNonAnalytical

sys.path.append(CONFIG_PATH_UTILS)
import utils as utils


def plane_dust(wavel, dz0, ct, dt0, dust_position, size_cube, params, source1, save=False, show_plots=False, show_initial_object=True):
    """
        Calculate the LE, the LE image and surface brightness

        Arguments:
            wavel: wavelenght of observation
            dz0: thickness of the dust in ly, given by user in in the txt containing arguments: parameters['dz0'] in pc
            ct: LE time of observation in years after max of source, given by user in in the txt containing arguments: parameters['ct'] in days
            dt0: duration of source in years, given by user in in the txt containing arguments: parameters['dt0'] in days
            dust_position: x0, y0, z0 center position of dust flat-cube (e.g. img of SPITZER etacar data) in ly given by user in the txt containing arguments: parameters['dust_position'] in pc
            size_cube: size_cube: size (x,y) of the dust flat-cube in ly, given by user in the txt containing arguments: parameters['size_cube'] in pc
            params: parameter of equation of a plane Ax + By + Dz + F = 0 
                given by user in parameters['plane_coefficients'] or calculate in know_3d_dust_position
            source1: Source object
            save: save results, images, plot and txt with LE params
            show_plots: show plots at the end of simulation
            show_initial_object: show geometrical configuration, dust, plane, paraboloid ct

        Returns:
            Plots
    """
    # load and reshape and boolean batch/stamp of SPITZER data: it must exists already in npy format
    img = gcd.generate_cube_dust() #generate_cube_dust_random()
    sheet_dust_img = np.sum(img.copy(), axis=-1)

    dust_shape = np.array(sheet_dust_img.shape)
    # Initiliaze DustShape object 
    planedust1 = PlaneDust(params, dz0, dust_shape, dust_position, size_cube )

    # Calculate LE object
    LE_planedust1source1 = LE.LESheetDust(ct, planedust1, source1, sheet_dust_img)
    x_inter_values, y_inter_values, z_inter_values, xy_matrix = LE_planedust1source1.run(show_initial_object=show_initial_object)


    if show_initial_object:
        LE_planedust1source1.plot_sheetdust()

    # Calculate surface brightness
    cossigma, surface_total = sb.SurfaceBrightnessDustSheetPlane(wavel, source1, 
                                                                 LE_planedust1source1,
                                                                 planedust1, 
                                                                 xy_matrix).calculate_surface_brightness()

    # Initialize and calculate the LE image from LE and surface brightness
    le_img = LEImageNonAnalytical(LE_planedust1source1, 
                                  planedust1, 
                                  surface_total, 
                                  pixel_resolution = 0.2, 
                                  cmap = 'magma_r')
    surface_val, surface_img, x_img, y_img, z_img_ly = le_img.create_le_surface()

    # name of the files
    n_speci = f"planedust_{CUBE_NAME}_dt0_{int(dt0 / fc.dtoy)}_ct{int(ct / fc.dtoy)}_c{dust_position}_size{size_cube}{dust_shape}dz0{round(dz0 / fc.pctoly, 2)}"


    fig, ax = plt.subplots(1,1, figsize = (8,8))
    ax.imshow(surface_img, origin = "lower")
    ax.set_title(n_speci)
    figs = LE_planedust1source1.plot(le_img.new_xs, le_img.new_ys, le_img.surface_original, n_speci)
    if save == True:
        pathg = PATH_TO_RESULTS_FIGURES
        plt.savefig(pathg+"\\img_"+n_speci+".pdf", dpi = 300 )
        figs.write_image(pathg+"\\scatter_"+n_speci+".pdf")
        info_le = [{
            'dust_shape (ly)': dust_shape,
            'size_cube (ly)': size_cube,
            'dust_position (ly)': dust_position,
            'dz0 (ly)': dz0,
            'dt0 (days)': int(round(dt0 / fc.dtoy)),
            'ct (days)': int(round(ct / fc.dtoy)),
            'file_name': pathg+"\\surface_"+n_speci+".npy"}]

        pathg = PATH_TO_RESULTS_SIMULATIONS 
        with open(pathg+'\\meta_info'+n_speci+'.pkl', 'wb') as f:
            pickle.dump(info_le, f)
        #-- save xy projection, the view as observer and surface brightness
        np.save(pathg+"\\x_inter_arcsec"+n_speci+".npy", le_img.new_xs)
        np.save(pathg+"\\y_inter_arcsec"+n_speci+".npy", le_img.new_ys)
        np.save(pathg+"\\xy_matrix"+n_speci+".npy", xy_matrix)
        np.save(pathg+"\\surface_"+n_speci+".npy", surface_img)
        np.save(pathg+"\\surface_values"+n_speci+".npy", surface_val)

        np.save(pathg+"\\ximg_arcsec"+n_speci+".npy", x_img)
        np.save(pathg+"\\yimg_arcsec"+n_speci+".npy", y_img)
        np.save(pathg+"\\zimgly_arcsec"+n_speci+".npy", z_img_ly)


        # -- save the intersection points in xyz system in ly
        np.save(pathg+"\\x_inter_ly"+n_speci+".npy", x_inter_values)
        np.save(pathg+"\\y_inter_ly"+n_speci+".npy", y_inter_values)
        np.save(pathg+"\\z_inter_ly"+n_speci+".npy", z_inter_values)

    if show_plots == True:
        plt.show()
        figs.show()



    #### from inf plane -> depth
    # z0ly = 10 * fc.pctoly
    # alpha = 15
    # a = np.tan(np.deg2rad(alpha))
    # size_cube = np.array([1,1]) * fc.pctoly #ly
    # params = [a, 0.0, 10.0, -z0ly]
    # x0 = 1 * fc.pctoly
    # y0 = 1.0 * fc.pctoly
    # z0 = (-params[-1] - params[0]*x0 - params[1]*y0) / params[2]

def know_3d_dust_position(x0, y0, z0, size_cube):
    """
        Given the center position (x0, y0, z0) of the dust flat-cube that user have information, e.g. the image-batch of SPITZER etacar data.
        Calculate 4 points inside the cube (size_cube) and fit the equation of a plane to the points.

        Arguments:
            x0, y0, z0: center position of dust flat-cube (e.g. img of SPITZER etacar data) given by user in the txt containing arguments: parameters['dust_position'] in pc
            size_cube: size (x,y) of the dust flat-cube in pc, given by user in the txt containing arguments: parameters['size_cube']
        Returns:
            Params: parameters of equation of a plane Ax + By + Dz + F = 0
    """

    from scipy.optimize import leastsq
    # x0 = 1 * fc.pctoly
    # y0 = 3 * fc.pctoly
    # z0 = 10 * fc.pctoly
    dust_position = np.array([x0, y0, z0]) * fc.pctoly
    def cal_4points(dust_position, size_cube):
        # points are at a distance=size dust sheet over sqrt(2)
        rr = np.linalg.norm(size_cube)
        p1 = dust_position + size_cube[0]*rr
        p2 = dust_position - size_cube[0]*rr
        p3 = dust_position + size_cube[1]*rr
        p4 = dust_position - size_cube[1]*rr
        ps = np.array([p1, p2, p3, p4])
        # define the x and y limits given the points just calcualted
        x_min = np.min(ps, axis=0)[0]
        x_max = np.max(ps, axis=0)[0]
        y_min = np.min(ps, axis=0)[1]
        y_max = np.max(ps, axis=0)[1]
        z_min = np.min(ps, axis=0)[2]
        z_max = np.max(ps, axis=0)[2]
        pointsxyz = np.array([[x_min, y_min, z_min],
                        [x_max, y_min, z_min],
                        [x_max, y_max, z_min],
                        [x_min, y_max, z_min],
                        [x_min, y_max, z_max],
                        [x_max, y_max, z_max],
                        [x_max, y_min, z_max],
                        [x_min, y_min, z_max]])
        xs = pointsxyz[:, 0]
        ys = pointsxyz[:, 1]
        zs = pointsxyz[:, 2]
        points = np.stack((np.array(xs), np.array(ys), np.array(zs)))
        return points
    points = cal_4points(dust_position, size_cube)
    
    def f_min(X,p):
        plane_xyz = p[0:3]
        distance = (plane_xyz*X.T).sum(axis=1) + p[3]
        return distance / np.linalg.norm(plane_xyz)

    def residuals(params, X):
        return f_min(X, params)
    
    pini  = [0.5, -0.1, -1.4, 1.3]
    params = leastsq(residuals, pini, args=(points))[0]
    print(params)
    return params

def run(file_name, args):
    """
        Read txt containing a dict with the mandatory parameters for each type simulation. 
        Two options here: 
            parameters['know_3d_dust_position'] = False:
                no information about the z/depth position of the dust -> user needs to define equation of a plane in the txt file parameters['plane_coefficients']
            parameters['know_3d_dust_position'] = True:
                information about the z/depth position of the dust -> call know_3d_dust_position() and determine equation of the plane

        Initiliaze the Source object.
        Call func to initilize LE calculation.

        Arguments:
            file_name = args: ins.txtparameters
            args: ins.bool_save, ins.bool_show_plots, ins.bool_show_initial_object

    """
    with open(file_name, 'rb') as handle:
        parameters = json.load(handle)
    dt0 =  parameters['dt0'] * fc.dtoy #years
    d =  vc.d #parameters['d']
    Flmax =  dc.Flmax #parameters['dt0']
    dz0 =  parameters['dz0'] * fc.pctoly
    ct = parameters['ct'] * fc.dtoy#in y
    size_cube = np.array(parameters['size_cube']) * fc.pctoly
    if bool(parameters['know_3d_dust_position']) == False:
        print('unkown')
        a = parameters['plane_coefficients'][0]
        ay = parameters['plane_coefficients'][1]
        az = parameters['plane_coefficients'][2] 
        z0ly = parameters['plane_coefficients'][3] * fc.pctoly
        params = [a, ay, az, -z0ly]
        x0 = parameters['dust_position'][0] *fc.pctoly
        y0 = parameters['dust_position'][1] *fc.pctoly
        z0 = (z0ly - (a*x0) - (ay*y0)) / (az) if az != 0 else 0
        dust_position = np.array([x0, y0, z0])
    else:
        print("known")
        x0 = parameters['dust_position'][0]
        y0 = parameters['dust_position'][1]
        z0 = parameters['dust_position'][2]
        dust_position = np.array([x0, y0, z0]) * fc.pctoly
        params = know_3d_dust_position(x0, y0, z0, size_cube)

    bool_save = args[0]
    bool_show_plots = args[1]
    bool_show_initial_object = args[2]

    wavel = parameters['wave'] #in um

    

    source1 = Source(dt0, d, Flmax)
    plane_dust(wavel, dz0, ct, dt0, dust_position, size_cube, 
               params, source1, save=bool_save, 
               show_plots=bool_show_plots, show_initial_object=bool_show_initial_object)