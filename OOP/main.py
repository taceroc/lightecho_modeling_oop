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
    # print(LE_plane1source1.r_le_out, LE_plane1source1.r_le_in)
    sb_plane = sb.SurfaceBrightnessAnalytical(source1, LE_plane1source1, [x_inter_values, y_inter_values, z_inter_values])
    ######
    mu = 6
    variance = 60
    sigma = np.sqrt(variance)
    x = np.linspace(0, 50, 1000)
    mag = -100*stats.norm.pdf(x, sigma, mu)+20
    lc = {}
    lc['mag'] = mag
    lc['time'] = x
    lc['time'] = lc['time'] - lc['time'][np.argmin(lc['mag'])]
    sb_plane.lc = lc
    #####
    
    cossigma, surface = sb_plane.calculate_surface_brightness()
    new_xs = utils.convert_ly_to_arcsec(source1.d, new_xs)
    new_ys = utils.convert_ly_to_arcsec(source1.d, new_ys)
    
    print("new_xs", new_xs.shape)
    r_in = utils.convert_ly_to_arcsec(source1.d, LE_plane1source1.r_le_in)
    r_out = utils.convert_ly_to_arcsec(source1.d, LE_plane1source1.r_le_out)
    print(plane1.eq_params[0],  plane1.eq_params[2])
    act = LE_plane1source1.ct * (plane1.eq_params[0] / plane1.eq_params[2])
    act = utils.convert_ly_to_arcsec(source1.d, act)
    bct = LE_plane1source1.ct * (plane1.eq_params[1] / plane1.eq_params[2])
    bct = utils.convert_ly_to_arcsec(source1.d, bct)

    # fig, ax = plt.subplots(1,1, figsize = (8,8))
    n_speci = f"InfPlane_dt0_{int(dt0 / fc.dtoy)}_ct{int(ct / fc.dtoy)}_loc{params}_dz0{dz0}"
    # n_speci = f"InfPlane_{random_uuid}"
    # utils.plot(new_xs, new_ys, surface, act, bct, ax, fig, save = False, name = n_speci)

    surface_val, surface_img, x_img, y_img = LEImageAnalytical(new_xs, new_ys, surface, r_in, r_out, pixel_resolution = 0.2, cmap = 'magma_r').create_le_surface(plane1, [act, bct])
    info_le = [{'r_in (arcsec)': r_in,
                'r_out (arcsec)': r_out,
                'act (arcsec)': act,
                'bct (arcsec)': bct,
                'params': params,
                'dz0 (ly)': dz0,
                'dt0 (days)': int(round(dt0 / fc.dtoy)),
                'ct (days)': int(round(ct / fc.dtoy))}]
    

    if save == True:
        pathg = PATH_TO_RESULTS_SIMULATIONS 
        with open(pathg+'\\meta_info'+n_speci+'.pkl', 'wb') as f:
            pickle.dump(info_le, f)
        #-- save xy projection, the view as observer and surface brightness
        np.save(pathg+"\\x_inter_arcsec"+n_speci+".npy", new_xs)
        np.save(pathg+"\\y_inter_arcsec"+n_speci+".npy", new_ys)
        np.save(pathg+"\\surface_"+n_speci+".npy", surface)
        np.save(pathg+"\\surface_img"+n_speci+".npy", surface_img)

        # -- save the intersection points in xyz system in ly
        np.save(pathg+"\\x_inter_ly"+n_speci+".npy", x_inter_values)
        np.save(pathg+"\\y_inter_ly"+n_speci+".npy", y_inter_values)
        np.save(pathg+"\\z_inter_ly"+n_speci+".npy", z_inter_values)

    surface_300_norm = ( surface - np.nanmin(surface)  ) / (np.nanmax(surface) - np.nanmin(surface))
    cmap = matplotlib.colormaps.get_cmap('magma_r')
    normalize = matplotlib.colors.Normalize(vmin=np.nanmin(surface_300_norm), vmax=np.nanmax(surface_300_norm))

    figs = go.Figure()
    figs.add_trace(go.Scatter(
        x=new_xs[0,0,:],
        y=new_ys[0,0,:],
        marker=dict(color=[f'rgb({int(cc1[0])}, {int(cc1[1])}, {int(cc1[2])})' for cc1 in cmap(normalize(surface_300_norm)) * 255], size=10),
        mode="markers")
              )
    figs.add_trace(go.Scatter(
        x=new_xs[0,1,:],
        y=new_ys[0,1,:],
        marker=dict(color=[f'rgb({int(cc1[0])}, {int(cc1[1])}, {int(cc1[2])})' for cc1 in cmap(normalize(surface_300_norm)) * 255], size=10),
        mode="markers")
              )
    # figs.show()


    fig, ax = plt.subplots(1,1, figsize = (8,8))
    ax.imshow(surface_img, origin = "lower")
    plt.show()
    if save == True:
        pathg = PATH_TO_RESULTS_FIGURES
        plt.savefig(pathg+"\\img_"+n_speci+".pdf", dpi = 300 )
        figs.write_image(pathg+"\\scatter_"+n_speci+".pdf")

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
    new_xs = utils.convert_ly_to_arcsec(source1.d, new_xs)
    new_ys = utils.convert_ly_to_arcsec(source1.d, new_ys)
    r_in = utils.convert_ly_to_arcsec(source1.d, LE_sphere1source1.r_le_in)
    r_out = utils.convert_ly_to_arcsec(source1.d, LE_sphere1source1.r_le_out)


    # fig, ax = plt.subplots(1,1, figsize = (8,8))
    n_speci = f"ShereCentered_dt0_{int(dt0 / fc.dtoy)}_ct{int(ct / fc.dtoy)}_r{params[-1]}_dz0{dz0 / fc.pctoly}"
    # utils.plot(new_xs, new_ys, surface, 0, 0, ax, fig, save = False, name = n_speci)

    surface_val, surface_img, x_img, y_img = LEImageAnalytical(new_xs, new_ys, surface, r_in, r_out, pixel_resolution = 0.2, cmap = 'magma_r').create_le_surface(sphere1, [0, 0])

    if save == True:
        pathg = PATH_TO_RESULTS_SIMULATIONS 
        #-- save xy projection, the view as observer and surface brightness
        np.save(pathg+"\\x_inter_arcsec"+n_speci+".npy", new_xs)
        np.save(pathg+"\\y_inter_arcsec"+n_speci+".npy", new_ys)
        np.save(pathg+"\\surface_"+n_speci+".npy", surface)

        # -- save the intersection points in xyz system in ly
        np.save(pathg+"\\x_inter_ly"+n_speci+".npy", x_inter_values)
        np.save(pathg+"\\y_inter_ly"+n_speci+".npy", y_inter_values)
        np.save(pathg+"\\z_inter_ly"+n_speci+".npy", z_inter_values)

    surface_300_norm = ( surface - np.nanmin(surface)  ) / (np.nanmax(surface) - np.nanmin(surface))
    cmap = matplotlib.colormaps.get_cmap('magma_r')
    normalize = matplotlib.colors.Normalize(vmin=np.nanmin(surface_300_norm), vmax=np.nanmax(surface_300_norm))

    figs = go.Figure()

    figs = go.Figure()
    figs.add_trace(go.Scatter(
        x=new_xs[0,0,:],
        y=new_ys[0,0,:],
        marker=dict(color=[f'rgb({int(cc1[0])}, {int(cc1[1])}, {int(cc1[2])})' for cc1 in cmap(normalize(surface_300_norm)) * 255], size=10),
        mode="markers")
              )
    figs.add_trace(go.Scatter(
        x=new_xs[0,1,:],
        y=new_ys[0,1,:],
        marker=dict(color=[f'rgb({int(cc1[0])}, {int(cc1[1])}, {int(cc1[2])})' for cc1 in cmap(normalize(surface_300_norm)) * 255], size=10),
        mode="markers")
              )
    # figs.show()

    fig, ax = plt.subplots(1,1, figsize = (8,8))
    ax.imshow(surface_img, origin = "lower")
    plt.show()

    if save == True:
        pathg = PATH_TO_RESULTS_FIGURES
        plt.savefig(pathg+"\\img_"+n_speci+".pdf", dpi = 300 )
        figs.write_image(pathg+"\\scatter_"+n_speci+".pdf")

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
    x_inter_values, y_inter_values, z_inter_values, xy_matrix = LE_planedust1source1.run()
    LE_planedust1source1.plot()

    # print(x_inter)
    arris = []
    for i in range(planedust1.side_x):
        for j in range(planedust1.side_y):
            arris.append([xy_matrix[j, i, 0], xy_matrix[j, i, 1], xy_matrix[j, i, 2]])

    arris = np.array(arris)

    cossigma, surface_total = sb.SurfaceBrightnessDustSheetPlane(source1, 
                                                                 LE_planedust1source1,
                                                                 planedust1, 
                                                                 xy_matrix).calculate_surface_brightness()

    flat = xy_matrix[:, :, :2].reshape(planedust1.side_x*planedust1.side_y, 2)
    cord = np.array([(xf, yf) for xf, yf in flat if xf != 0 or yf != 0])
    flat_sb = surface_total.reshape(surface_total.shape[0]*surface_total.shape[1])

    x_inter_arcsec = utils.convert_ly_to_arcsec(source1.d, cord[:, 0])
    y_inter_arcsec = utils.convert_ly_to_arcsec(source1.d, cord[:, 1])
    surface_total = flat_sb[flat[:,0]!=0]
    print(flat[:,1])

    print("surface shape")
    print(surface_total.shape)

    surface_val, surface_img, x_img, y_img = LEImageNonAnalytical(x_inter_arcsec, y_inter_arcsec, surface_total, pixel_resolution = 0.2, cmap = 'magma_r').create_le_surface(planedust1)


    n_speci = f"planedust_{CUBE_NAME}_dt0_{int(dt0 / fc.dtoy)}_ct{int(ct / fc.dtoy)}_c{dust_position}_size{size_cube}{dust_shape}dz0{dz0}"
    info_le = [{
                'dust_shape (ly)': dust_shape,
                'size_cube (ly)': size_cube,
                'dust_position (ly)': dust_position,
                'dz0 (ly)': dz0,
                'dt0 (days)': int(round(dt0 / fc.dtoy)),
                'ct (days)': int(round(ct / fc.dtoy))}]
    

    if save == True:
        pathg = PATH_TO_RESULTS_SIMULATIONS 
        with open(pathg+'\\meta_info'+n_speci+'.pkl', 'wb') as f:
            pickle.dump(info_le, f)
        #-- save xy projection, the view as observer and surface brightness
        np.save(pathg+"\\x_inter_arcsec"+n_speci+".npy", x_inter_arcsec)
        np.save(pathg+"\\y_inter_arcsec"+n_speci+".npy", y_inter_arcsec)
        np.save(pathg+"\\xy_matrix"+n_speci+".npy", xy_matrix)
        np.save(pathg+"\\surface_"+n_speci+".npy", surface_total)

        # -- save the intersection points in xyz system in ly
        np.save(pathg+"\\x_inter_ly"+n_speci+".npy", x_inter_values)
        np.save(pathg+"\\y_inter_ly"+n_speci+".npy", y_inter_values)
        np.save(pathg+"\\z_inter_ly"+n_speci+".npy", z_inter_values)



    
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

    # figs1 = go.Figure()
    # figs1.add_trace(go.Scatter3d(
    #     x=np.array(arris)[:,0],
    #     y=np.array(arris)[:,1],
    #     z=np.array(arris)[:,2],
    #     marker=dict(size=10),
    #     mode="markers")
    #             )
    # figs1.show()


    fig, ax = plt.subplots(1,1, figsize = (8,8))
    ax.imshow(surface_img, origin = "lower")
    if save == True:
        pathg = PATH_TO_RESULTS_FIGURES
        plt.savefig(pathg+"\\img_"+n_speci+".pdf", dpi = 300 )
        figs.write_image(pathg+"\\scatter_"+n_speci+".pdf")
    plt.show()

# dz0 = 0.02 #* fc.pctoly #vc.dz0 #ly
# ct =  180 * fc.dtoy #180
# dt0 = 50 * fc.dtoy #180
# r0ly = 0.1 * fc.pctoly
# source1 = Source(dt0, vc.d, dc.Flmax)
# sphere(dz0, ct, dt0, r0ly, source1, save=True)

dz0 = 0.02 * fc.pctoly #vc.dz0 #ly
ct =  180 * fc.dtoy #180
dt0 = 50 * fc.dtoy #180
# this is the same solution as v838
z0ly = 1 * fc.pctoly
alpha = 15
a = np.tan(np.deg2rad(alpha))
source1 = Source(dt0, vc.d, dc.Flmax)
plane(dz0, ct, dt0, [a, 0, 1, -z0ly], source1, save=False)

# dz0 = 0.02 * fc.pctoly#ly
# ct =  365*30 * fc.dtoy #180
# dt0 = 365*20 * fc.dtoy #180
# z0ly = 10 * fc.pctoly
# alpha = 15
# a = np.tan(np.deg2rad(alpha))
# size_cube = np.array([1,1]) * fc.pctoly #ly
# params = [a, 0.0, 1.0, -z0ly]
# x0 = 15 * fc.pctoly
# y0 = 1.0 * fc.pctoly
# z0 = (-params[-1] - params[0]*x0 - params[1]*y0) / params[2]
# dust_position = np.array([x0, y0, z0])
# print(x0, y0, z0)

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
# source1 = Source(dt0, vc.d, dc.Flmax)
# plane_dust(dz0, ct, dt0, dust_position, size_cube, params, source1, save=False)



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