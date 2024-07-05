import sys
import numpy as np
from astropy import units as u
import astropy.cosmology.units as cu
from astropy.cosmology import FlatLambdaCDM
from shapely.geometry import Polygon, Point
import matplotlib
import matplotlib.pyplot as plt
from scipy import integrate
import pickle
import scipy.stats as stats
from definitions import CONFIG_PATH_DUST, PATH_TO_DUST_IMG_CODE, CONFIG_PATH_UTILS, PATH_TO_RESULTS_SIMULATIONS, PATH_TO_RESULTS_FIGURES, CUBE_NAME
sys.path.append(PATH_TO_DUST_IMG_CODE)
import generate_cube_dust as gcd
sys.path.append(CONFIG_PATH_DUST)
import var_constants as vc
import fix_constants as fc
import dust_constants as dc
# sys.path.append(ROOT_DIR)
from DustShape import InfPlane, SphereCenter, SphericalBlub, PlaneDust
import LightEcho as LE #import LEPlane, LESphereCentered, LESphericalBulb, LE_PlaneDust
from Source import Source
import SurfaceBrightness as sb
from LE_img import LEImageAnalytical, LEImageNonAnalytical

sys.path.append(CONFIG_PATH_UTILS)
import utils as utils


import plotly
import plotly.io as pio
# pio.renderers.default = "iframe"
import plotly.graph_objs as go



def plane(dz0, ct, dt0, params, source1, save = False):


    plane1 = InfPlane(params, dz0)
    LE_plane1source1 = LE.LEPlane(ct, plane1, source1)
    x_inter_values, y_inter_values, z_inter_values, new_xs, new_ys = LE_plane1source1.run()
    sb_plane = sb.SurfaceBrightnessAnalytical(source1, LE_plane1source1, [x_inter_values, y_inter_values, z_inter_values])
    ######
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
    
    cossigma, surface = sb_plane.calculate_surface_brightness()

    # fig, ax = plt.subplots(1,1, figsize = (8,8))
    n_speci = f"InfPlane_dt0_{int(dt0 / fc.dtoy)}_ct{int(ct / fc.dtoy)}_loc{params}_dz0{dz0}"
    

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
    


    figs = LE_plane1source1.plot(le_img.new_xs, le_img.new_ys, surface, n_speci)
    fig, ax = plt.subplots(1,1, figsize = (8,8))
    ax.imshow(surface_img, origin = "lower")
    ax.set_title(n_speci)
    plt.show()
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

        # -- save the intersection points in xyz system in ly
        np.save(pathg+"\\x_inter_ly"+n_speci+".npy", x_inter_values)
        np.save(pathg+"\\y_inter_ly"+n_speci+".npy", y_inter_values)
        np.save(pathg+"\\z_inter_ly"+n_speci+".npy", z_inter_values)

def sphere(dz0, ct, dt0, r0ly, source1, save = False):
    A = 0
    B = 0
    F = 0
    D = r0ly
    params = [A, B, F, D]

    sphere1 = SphereCenter(params, dz0)
    LE_sphere1source1 = LE.LESphereCentered(ct, sphere1, source1)
    x_inter_values, y_inter_values, z_inter_values, new_xs, new_ys = LE_sphere1source1.run()
    cossigma, surface = sb.SurfaceBrightnessAnalytical(source1, LE_sphere1source1, [x_inter_values, y_inter_values, z_inter_values]).calculate_surface_brightness()

    n_speci = f"ShereCentered_dt0_{int(dt0 / fc.dtoy)}_ct{int(ct / fc.dtoy)}_r{params[-1]}_dz0{dz0 / fc.pctoly}"

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
    plt.show()

    if save == True:
        pathg = PATH_TO_RESULTS_FIGURES
        plt.savefig(pathg+"\\img_"+n_speci+".pdf", dpi = 300 )
        figs.write_image(pathg+"\\scatter_"+n_speci+".pdf")

        pathg = PATH_TO_RESULTS_SIMULATIONS 
        with open(pathg+'\\meta_info'+n_speci+'.pkl', 'wb') as f:
            pickle.dump(info_le, f)
        #-- save xy projection, the view as observer and surface brightness
        np.save(pathg+"\\x_inter_arcsec"+n_speci+".npy", new_xs)
        np.save(pathg+"\\y_inter_arcsec"+n_speci+".npy", new_ys)
        np.save(pathg+"\\surface_"+n_speci+".npy", surface)

        # -- save the intersection points in xyz system in ly
        np.save(pathg+"\\x_inter_ly"+n_speci+".npy", x_inter_values)
        np.save(pathg+"\\y_inter_ly"+n_speci+".npy", y_inter_values)
        np.save(pathg+"\\z_inter_ly"+n_speci+".npy", z_inter_values)

def blub(dz0, ct, params, source1, save = False):

    bulb = SphericalBlub(params, dz0)
    LE_bulbsource1 = LE.LESphericalBulb(ct, bulb, source1, size=100)
    size_x = LE_bulbsource1.size_x
    size_y = LE_bulbsource1.size_y
    size_z = LE_bulbsource1.size_z
    x_inter_values, y_inter_values, z_inter_values, xy_matrix, z_matrix = LE_bulbsource1.run()

    
    print(z_matrix.shape)
    flat = xy_matrix.reshape(size_x*size_y,2)
    cord = np.array([(xf, yf) for xf, yf in flat if xf != 0 or yf != 0])
    arris = []
    for i in range(size_x):
        for j in range(size_y):
            for k in range(size_z):
                arris.append([xy_matrix[j, i, 0], xy_matrix[j, i, 1], z_matrix[j, i, k]])
    arris = np.array(arris)
    
    cossigma, surface = sb.SurfaceBrightnessBulb(source1, LE_bulbsource1, xy_matrix, z_matrix).calculate_surface_brightness()

    flat_sb = surface.reshape(surface.shape[0]*surface.shape[1])

    x_inter_arcsec = utils.convert_ly_to_arcsec(source1.d, cord[:,0])
    y_inter_arcsec = utils.convert_ly_to_arcsec(source1.d, cord[:,1])
    surface_total = flat_sb[flat[:,0]!=0]

    n_speci = f"SphericalBlulb_dt0_{int(dt0 / fc.dtoy)}_ct{int(ct / fc.dtoy)}_loc{params}"
    surface_val, surface_img, x_img, y_img = LEImageNonAnalytical(x_inter_arcsec, y_inter_arcsec, surface_total, pixel_resolution = 0.2, cmap = 'magma_r').create_le_surface(bulb)
    if save == True:
        pathg = PATH_TO_RESULTS_SIMULATIONS 
        #-- save xy projection, the view as observer and surface brightness
        np.save(pathg+"\\x_inter_arcsec"+n_speci+".npy", x_inter_arcsec)
        np.save(pathg+"\\y_inter_arcsec"+n_speci+".npy", y_inter_arcsec)
        np.save(pathg+"\\xy_matrix"+n_speci+".npy", xy_matrix)
        np.save(pathg+"\\z_matrix"+n_speci+".npy", z_matrix)
        np.save(pathg+"\\surface_"+n_speci+".npy", surface_total)

        # -- save the intersection points in xyz system in ly
        np.save(pathg+"\\x_inter_ly"+n_speci+".npy", x_inter_values)
        np.save(pathg+"\\y_inter_ly"+n_speci+".npy", y_inter_values)
        np.save(pathg+"\\z_inter_ly"+n_speci+".npy", z_inter_values)

    print(len(surface_total), np.nanmin(surface_total), np.nanmax(surface_total))
    surface_300_norm = ( surface_total - np.nanmin(surface_total)  ) / (np.nanmax(surface_total) - np.nanmin(surface_total))
    cmap = matplotlib.colormaps.get_cmap('magma_r')
    normalize = matplotlib.colors.Normalize(vmin=np.nanmin(surface_300_norm), vmax=np.nanmax(surface_300_norm))

    figs = go.Figure()

    figs.add_trace(go.Scatter(
        x=x_inter_arcsec,
        y=y_inter_arcsec,
        marker=dict(color=[f'rgb({int(cc1[0])}, {int(cc1[1])}, {int(cc1[2])})' for cc1 in cmap(normalize(surface_300_norm)) * 255], size=10),
        mode="markers")
                )
    figs.show()


    fig, ax = plt.subplots(1,1, figsize = (8,8))
    ax.imshow(surface_img, origin = "lower")
    if save == True:
        pathg = PATH_TO_RESULTS_FIGURES
        plt.savefig(pathg+"\\img_"+n_speci+".pdf", dpi = 300 )
        figs.write_image(pathg+"\\scatter_"+n_speci+".pdf")
    plt.show()

def plane_dust(dz0, ct, dt0, dust_position, size_cube, params, source1, save = False):
    img = gcd.generate_cube_dust() #generate_cube_dust_random()
    sheet_dust_img = np.sum(img.copy(), axis=-1)

    dust_shape = np.array(sheet_dust_img.shape)
    planedust1 = PlaneDust(params, dz0, dust_shape, dust_position, size_cube )

 
    LE_planedust1source1 = LE.LESheetDust(ct, planedust1, source1, sheet_dust_img)
    x_inter_values, y_inter_values, z_inter_values, xy_matrix = LE_planedust1source1.run(show_initial_object=True)
    LE_planedust1source1.plot_sheetdust()

    cossigma, surface_total = sb.SurfaceBrightnessDustSheetPlane(source1, 
                                                                 LE_planedust1source1,
                                                                 planedust1, 
                                                                 xy_matrix).calculate_surface_brightness()


    le_img = LEImageNonAnalytical(LE_planedust1source1, 
                                  planedust1, 
                                  surface_total, 
                                  pixel_resolution = 0.2, 
                                  cmap = 'magma_r')
    surface_val, surface_img, x_img, y_img = le_img.create_le_surface()


    n_speci = f"planedust_{CUBE_NAME}_dt0_{int(dt0 / fc.dtoy)}_ct{int(ct / fc.dtoy)}_c{dust_position}_size{size_cube}{dust_shape}dz0{dz0}"

    
 
    figs = LE_planedust1source1.plot(le_img.new_xs, le_img.new_ys, le_img.surface_original, n_speci)

    fig, ax = plt.subplots(1,1, figsize = (8,8))
    ax.imshow(surface_img, origin = "lower")
    plt.show()
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

        # -- save the intersection points in xyz system in ly
        np.save(pathg+"\\x_inter_ly"+n_speci+".npy", x_inter_values)
        np.save(pathg+"\\y_inter_ly"+n_speci+".npy", y_inter_values)
        np.save(pathg+"\\z_inter_ly"+n_speci+".npy", z_inter_values)
    

# dz0 = 0.02 #* fc.pctoly #vc.dz0 #ly
# ct =  180 * fc.dtoy #180
# dt0 = 50 * fc.dtoy #180
# r0ly = 0.1 * fc.pctoly
# source1 = Source(dt0, vc.d, dc.Flmax)
# sphere(dz0, ct, dt0, r0ly, source1, save=True)

# dz0 = 0.02 * fc.pctoly #vc.dz0 #ly
# ct =  180 * fc.dtoy #180
# dt0 = 50 * fc.dtoy #180
# # this is the same solution as v838
# z0ly = 1 * fc.pctoly
# alpha = 15
# a = np.tan(np.deg2rad(alpha))
# source1 = Source(dt0, vc.d, dc.Flmax)
# plane(dz0, ct, dt0, [a, 0, 1, -z0ly], source1, save=False)

dz0 = 0.02 * fc.pctoly#ly
ct =  180*5 * fc.dtoy #180
dt0 = 180*2 * fc.dtoy #180
z0ly = 10 * fc.pctoly
alpha = 15
a = np.tan(np.deg2rad(alpha))
size_cube = np.array([1,1]) * fc.pctoly #ly
params = [a, 0.0, 10.0, -z0ly]
x0 = 1 * fc.pctoly
y0 = 1.0 * fc.pctoly
z0 = (-params[-1] - params[0]*x0 - params[1]*y0) / params[2]
dust_position = np.array([x0, y0, z0])
print(x0, y0, z0)

# xp = np.linspace(-10,10,100)
# xp, yp = np.meshgrid(xp, xp)
# zp = (-params[0] * xp - params[1] * yp - params[-1]) / params[2]
# figs = go.Figure()
# figs.add_trace(go.Surface(
#     x=xp,
#     y=yp,
#     z=zp,
#     opacity=0.2))

# figs.add_trace(go.Scatter3d(
#     x=[0, x0, p1[0], p2[0], p3[0], p4[0]],
#     y=[0, y0, p1[1], p2[1], p3[1], p4[1]],
#     z=[0, z0, p1[2], p2[2], p3[2], p4[2]],
#     mode="markers",
#     marker=dict(color="red", size=4),
#     opacity=1)
#             )
# figs.show()
source1 = Source(dt0, vc.d, dc.Flmax)
plane_dust(dz0, ct, dt0, dust_position, size_cube, params, source1, save=False)



# dz0 = 0.02 * fc.pctoly #vc.dz0 #ly
# ct =  180 * fc.dtoy #180
# dt0 = 50 * fc.dtoy #180
# A = 7.17
# B = 2.30
# F = 51.44
# D = 2 * fc.pctoly
# params = [A, B, F, D]
# source1 = Source(dt0, vc.d, dc.Flmax)
# blub(dz0, ct, params, source1, save = False)