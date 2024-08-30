import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import scipy.stats as stats
from definitions import CONFIG_PATH_DUST, CONFIG_PATH_UTILS, PATH_TO_RESULTS_SIMULATIONS, PATH_TO_RESULTS_FIGURES
sys.path.append(CONFIG_PATH_DUST)
import var_constants as vc
import fix_constants as fc
import dust_constants as dc
# sys.path.append(ROOT_DIR)
from DustShape import InfPlane
import LightEcho as LE #import LEPlane, LESphereCentered, LESphericalBulb, LE_PlaneDust
from Source import Source
import SurfaceBrightness as sb
from LE_img import LEImageAnalytical

sys.path.append(CONFIG_PATH_UTILS)
import utils as utils


def plane(wavel, dz0, ct, dt0, params, source1, save=False, show_plots=False):
    """
        Calculate the LE, the LE image and surface brightness

        Arguments:
            wavel: wavelenght of observation
            dz0: thickness of the dust in ly, given by user in in the txt containing arguments: parameters['dz0'] in pc
            ct: LE time of observation in years after max of source, given by user in in the txt containing arguments: parameters['ct'] in days
            dt0: duration of source in years, given by user in in the txt containing arguments: parameters['dt0'] in days
            params: parameter of equation of a plane Ax + By + Dz + F = 0 
            source1: Source object
            save: save results, images, plot and txt with LE params
            show_plots: show plots at the end of simulation

        Returns:
            Plots
    """
    # Initiliaze DustShape object 
    plane1 = InfPlane(params, dz0)
    # Calculate LE object
    LE_plane1source1 = LE.LEPlane(ct, plane1, source1)
    x_inter_values, y_inter_values, z_inter_values, new_xs, new_ys = LE_plane1source1.run()
    # Calculate surface brightness
    sb_plane = sb.SurfaceBrightnessAnalytical(wavel, source1, LE_plane1source1, [x_inter_values, y_inter_values, z_inter_values])

    ###### temporal LC from source
    mu = 6
    variance = 800
    sigma = np.sqrt(variance)
    x = np.linspace(0, 360, 1000)
    mag = -500*stats.norm.pdf(x, sigma, mu)+20
    lc = {}
    lc['mag'] = mag
    lc['time'] = x * fc.dtoy
    sb_plane.lc = lc
    #####

    # Calculate surface brightness
    cossigma, surface = sb_plane.calculate_surface_brightness()

    # fig, ax = plt.subplots(1,1, figsize = (8,8))
    n_speci = f"InfPlane_dt0_{int(dt0 / fc.dtoy)}_ct{int(ct / fc.dtoy)}_loc{params}_dz0{round(dz0 / fc.pctoly, 2)}"
    
    # Initialize and calculate the LE image from LE and surface brightness
    le_img = LEImageAnalytical(LE_plane1source1, plane1, surface, pixel_resolution = 0.2, cmap = 'magma_r')
    surface_val, surface_img, x_img, y_img = le_img.create_le_surface()
    

    info_le = [{'r_in (arcsec)': le_img.r_le_in,
                'r_out (arcsec)': le_img.r_le_out,
                'act (arcsec)': le_img.act,
                'bct (arcsec)': le_img.bct,
                'params': params,
                'dz0 (ly)': dz0,
                'dt0 (days)': int(round(dt0 / fc.dtoy)),
                'ct (days)': int(round(ct / fc.dtoy))}]
    
    fig, ax = plt.subplots(1,1, figsize = (8,8))
    ax.imshow(surface_img, origin = "lower")
    ax.set_title(n_speci)
    figs = LE_plane1source1.plot(le_img.new_xs, le_img.new_ys, surface, n_speci)
    if save == True:
        pathg = PATH_TO_RESULTS_FIGURES
        plt.savefig(pathg+"\\img_"+n_speci+".pdf", dpi = 300 )
        figs.write_image(pathg+"\\scatter_"+n_speci+".pdf")

        pathg = PATH_TO_RESULTS_SIMULATIONS 
        with open(pathg+'\\meta_info'+n_speci+'.pkl', 'wb') as f:
            pickle.dump(info_le, f)
        #-- save xy projection, the view as observer and surface brightness
        np.save(pathg+"\\x_inter_arcsec"+n_speci+".npy", le_img.new_xs)
        np.save(pathg+"\\y_inter_arcsec"+n_speci+".npy", le_img.new_ys)
        np.save(pathg+"\\surface_"+n_speci+".npy", surface)
        np.save(pathg+"\\surface_img"+n_speci+".npy", surface_img)
        np.save(pathg+"\\surface_values"+n_speci+".npy", surface_val)


        # -- save the intersection points in xyz system in ly
        np.save(pathg+"\\x_inter_ly"+n_speci+".npy", x_inter_values)
        np.save(pathg+"\\y_inter_ly"+n_speci+".npy", y_inter_values)
        np.save(pathg+"\\z_inter_ly"+n_speci+".npy", z_inter_values)

    if show_plots == True:
        plt.show()
        figs.show()

# dz0 = 0.02 * fc.pctoly #vc.dz0 #ly
# ct =  180 * fc.dtoy #180
# dt0 = 50 * fc.dtoy #180
# # this is the same solution as v838
# z0ly = 1 * fc.pctoly
# alpha = 15
# a = np.tan(np.deg2rad(alpha))

def run(file_name, args):
    """
        Read txt containing a dict with the mandatory parameters for each type simulation. 
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
    a = parameters['plane_coefficients'][0]
    ay = parameters['plane_coefficients'][1]
    az = parameters['plane_coefficients'][2] 
    z0ly = parameters['plane_coefficients'][3] * fc.pctoly

    bool_save = args[0]
    bool_show_plots = args[1]

    wavel = parameters['wave']
    source1 = Source(dt0, d, Flmax)
    plane(wavel, dz0, ct, dt0, [a, ay, az, -z0ly], source1, save=bool_save, show_plots=bool_show_plots)