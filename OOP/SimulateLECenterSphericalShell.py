import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats
from definitions import CONFIG_PATH_DUST, CONFIG_PATH_UTILS, PATH_TO_RESULTS_SIMULATIONS, PATH_TO_RESULTS_FIGURES
sys.path.append(CONFIG_PATH_DUST)
import var_constants as vc
import fix_constants as fc
import dust_constants as dc
# sys.path.append(ROOT_DIR)
from DustShape import SphereCenter
import LightEcho as LE #import LEPlane, LESphereCentered, LESphericalBulb, LE_PlaneDust
from Source import Source
import SurfaceBrightness as sb
from LE_img import LEImageAnalytical

sys.path.append(CONFIG_PATH_UTILS)
import utils as utils


def sphere(wavel, dz0, ct, dt0, r0ly, source1, save=False, show_plots=False):
    A = 0
    B = 0
    F = 0
    D = r0ly
    params = [A, B, F, D]

    sphere1 = SphereCenter(params, dz0)
    LE_sphere1source1 = LE.LESphereCentered(ct, sphere1, source1)
    x_inter_values, y_inter_values, z_inter_values, new_xs, new_ys = LE_sphere1source1.run()
    cossigma, surface = sb.SurfaceBrightnessAnalytical(wavel, source1, LE_sphere1source1, [x_inter_values, y_inter_values, z_inter_values]).calculate_surface_brightness()

    n_speci = f"ShereCentered_dt0_{int(dt0 / fc.dtoy)}_ct{int(ct / fc.dtoy)}_r{params[-1]}_dz0{round(dz0 / fc.pctoly, 2)}"

    le_img = LEImageAnalytical(LE_sphere1source1, sphere1, surface, pixel_resolution = 0.2, cmap = 'magma_r')
    surface_val, surface_img, x_img, y_img = le_img.create_le_surface()

    info_le = [{'r_in (arcsec)': le_img.r_le_in,
                'r_out (arcsec)': le_img.r_le_out,
                'act (arcsec)': le_img.act,
                'bct (arcsec)': le_img.bct,
                'r0ly': params[-1],
                'dz0 (ly)': dz0,
                'dt0 (days)': int(round(dt0 / fc.dtoy)),
                'ct (days)': int(round(ct / fc.dtoy))}]
    

    figs = LE_sphere1source1.plot(le_img.new_xs, le_img.new_ys, surface, n_speci)

    fig, ax = plt.subplots(1,1, figsize = (8,8))
    ax.imshow(surface_img, origin = "lower")
    ax.set_title(n_speci)
    if save == True:
        print('heresss')
        pathg = PATH_TO_RESULTS_FIGURES
        plt.savefig(pathg+"\\img_"+n_speci+".pdf", dpi = 300 )
        figs.write_image(pathg+"\\scatter_"+n_speci+".pdf")

        pathg = PATH_TO_RESULTS_SIMULATIONS 
        with open(pathg+'\\meta_info'+n_speci+'.pkl', 'wb') as f:
            pickle.dump(info_le, f)
        #-- save xy projection, the view as observer and surface brightness
        np.save(pathg+"\\x_inter_arcsec"+n_speci+".npy", le_img.new_xs)
        np.save(pathg+"\\y_inter_arcsec"+n_speci+".npy", le_img.new_ys)
        np.save(pathg+"\\surface_"+n_speci+".npy", surface_img)

        # -- save the intersection points in xyz system in ly
        np.save(pathg+"\\x_inter_ly"+n_speci+".npy", x_inter_values)
        np.save(pathg+"\\y_inter_ly"+n_speci+".npy", y_inter_values)
        np.save(pathg+"\\z_inter_ly"+n_speci+".npy", z_inter_values)

    if show_plots == True:
        plt.show()
        figs.show()

# dz0 = 0.02 #* fc.pctoly #vc.dz0 #ly
# ct =  180 * fc.dtoy #180
# dt0 = 50 * fc.dtoy #180
# r0ly = 0.1 * fc.pctoly
# source1 = Source(dt0, vc.d, dc.Flmax)
# sphere(dz0, ct, dt0, r0ly, source1, save=True)

def run(file_name, args):
    import json
    with open(file_name, 'rb') as handle:
        parameters = json.load(handle)
    dt0 =  parameters['dt0'] * fc.dtoy #years
    d =  vc.d #parameters['d']
    Flmax =  dc.Flmax #parameters['dt0']
    dz0 =  parameters['dz0'] * fc.pctoly
    ct = parameters['ct'] * fc.dtoy#in y
    r0ly = parameters['radii'] * fc.pctoly

    bool_save = args[0]
    bool_show_plots = args[1]

    wavel = parameters['wave']
    source1 = Source(dt0, d, Flmax)
    sphere(wavel, dz0, ct, dt0, r0ly, source1, save=bool_save, show_plots=bool_show_plots)