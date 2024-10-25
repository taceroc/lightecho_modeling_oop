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
    x_inter_values, y_inter_values, z_inter_values, new_xs, new_ys, new_zs = LE_plane1source1.run()
    # Calculate surface brightness
    sb_plane = sb.SurfaceBrightnessAnalytical(wavel, source1, LE_plane1source1, [x_inter_values, y_inter_values, z_inter_values])

    # ###### temporal LC from source
    # mu = 20
    # variance = 8000
    # sigma = np.sqrt(variance)
    # x = np.linspace(0, 360, 1000)
    # mag = -800*stats.norm.pdf(x, sigma, mu)+20
    # lc = {}
    # lc['mag'] = mag
    # lc['time'] = x * fc.dtoy
    # sb_plane.lc = lc
    # #####

    ######
    sys.path.append("C:\\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\LC\\")
    import extract_lc
    name_= 'sn2011fe'
    lc_sn2011fe = extract_lc.read_from_open_catalog(name_)
    small_sources = [lc_sn2011fe.groupby('source').size().index[2],
                 lc_sn2011fe.groupby('source').size().index[8]]
    D11 = small_sources[-1]
    lc_sn2011fe = lc_sn2011fe[lc_sn2011fe.source != D11]
    lc = {}
    lc['mag'] = lc_sn2011fe[(lc_sn2011fe['band'] == 'B') & (lc_sn2011fe['time'] <= lc_sn2011fe['time'].min() + 365)]['magnitude'].values
    lc['time'] = (lc_sn2011fe[(lc_sn2011fe['band'] == 'B') & (lc_sn2011fe['time'] <= lc_sn2011fe['time'].min() + 365)]['time'].values - lc_sn2011fe['time'].min())* fc.dtoy
    sb_plane.lc = lc
    ######

    def find_nearest(time_array, value):
        time_array = np.asarray(time_array)
        idx = (np.abs(time_array - value)).argmin()
        return idx

    # diff_dt = sb_plane.determine_flux_time_loop()
    # print(diff_dt)
    times_to_loop = lc['time']
    print(times_to_loop.shape)
    all_x_inter_values = []
    all_y_inter_values = []
    all_z_inter_values = []
    all_new_xs = []
    all_new_ys = []
    all_new_zs = []
    all_r_le_in = []
    all_r_le_out = []
    times_lc = []
    all_surface = []
    print("TIMEEE", ct / fc.dtoy)
    tttts = np.linspace(lc['time'][lc['mag'] == lc['mag'].min()]-0.02, lc['time'][lc['mag'] == lc['mag'].min()]+0.02, 10)
    time_count = 0
    for idx, tt in enumerate(tttts):
        print(lc['time'][lc['mag'] == lc['mag'].min()], lc['time'][lc['mag'] == lc['mag'].min()]/fc.dtoy)
        ctt = ct + (lc['time'][lc['mag'] == lc['mag'].min()] - tt)
        idxs = find_nearest(lc['time'], tt)
        flux_to_use = sb_plane.lc['mag'][idxs]
        flux_to_use = 10**((-48.6 - flux_to_use) / 2.5)
        # ctt = ct-tt
        
        LE_plane1source1_tt = LE.LEPlane(ctt, plane1, source1)
        print("TIMEEE", tt / fc.dtoy, ctt/ fc.dtoy)
        x_inter_values, y_inter_values, z_inter_values, new_xs, new_ys, new_zs = LE_plane1source1_tt.run()
        
        print(sb_plane.d, ctt - ct, ct)
        titlde = np.linalg.norm([x_inter_values, y_inter_values, z_inter_values], axis=0) + \
                np.linalg.norm([x_inter_values, y_inter_values, (sb_plane.d - z_inter_values)], axis=0) - sb_plane.d + ctt - ct
        
        # for i in titlde:
        #     if np.isclose(i, ct, atol=1e-01):
        #         print(i, "closee")
        if np.isclose(np.mean(titlde), ct, atol=1e-02):
            times_lc.append((lc['time'][lc['mag'] == lc['mag'].min()][0]- tt)/fc.dtoy)

            time_count = time_count + 1
            print("is close in days", abs((np.mean(titlde) -ct)/fc.dtoy))
            sb_plane_all = sb.SurfaceBrightnessAnalytical(wavel, source1, LE_plane1source1, [x_inter_values, y_inter_values, z_inter_values])
            # Calculate surface brightness
            sb_plane_all.lc = lc
            cossigma, surface = sb_plane_all.calculate_surface_brightness(tt)

            all_x_inter_values.extend(x_inter_values)
            all_y_inter_values.extend(y_inter_values)
            all_z_inter_values.extend(z_inter_values)
            all_new_xs.extend(new_xs)
            all_new_ys.extend(new_ys)
            all_new_zs.extend(new_zs)
            print("RLE")
            print(new_xs.shape)
            print(len(all_new_zs))
            # print(LE_plane1source1_tt.r_le)
            all_r_le_in.extend(LE_plane1source1_tt.r_le_in)
            all_r_le_out.extend(LE_plane1source1_tt.r_le_out)

            all_surface.extend(surface)
        

        else:
            print("no close", abs((np.mean(titlde) -ct)/fc.dtoy), ((np.mean(titlde))/fc.dtoy))
    if time_count == 0:
        print('No arrival time match desire time')
        sys.exit(0)
    else:
        print(f"times respective to max LC {times_lc}")

       
    # print(surface)
    all_x_inter_values = np.array(all_x_inter_values)
    all_y_inter_values = np.array(all_y_inter_values)
    all_z_inter_values = np.array(all_z_inter_values)
    # all_new_xs = np.array(all_new_xs).reshape(1,2,len(all_x_inter_values))
    
    # all_new_ys = np.array(all_new_ys).reshape(1,2,len(all_x_inter_values))
    # all_new_zs = np.array(all_new_zs).reshape(1,2,len(all_x_inter_values))
    
    def proper_shape_new_s(all_new_xs):
        temp_new_xs = np.array(all_new_xs)
        temp_out_xs = temp_new_xs[:, 0, :]
        temp_in_xs = temp_new_xs[:, 1, :]

        return np.concatenate([temp_out_xs.flatten(), temp_in_xs.flatten()])

    all_new_xs = proper_shape_new_s(all_new_xs).reshape(1,2,len(all_x_inter_values))
    all_new_ys = proper_shape_new_s(all_new_ys).reshape(1,2,len(all_x_inter_values))
    all_new_zs = proper_shape_new_s(all_new_zs).reshape(1,2,len(all_x_inter_values))

    print("all_new_xs shape")
    print(all_new_xs.shape)

    all_surface = np.array(all_surface).reshape(len(all_x_inter_values))

    LE_plane1source1.x_projected = all_new_xs
    LE_plane1source1.y_projected = all_new_ys
    LE_plane1source1.z_projected = all_new_zs
    LE_plane1source1.r_le_in = np.array(all_r_le_in)
    LE_plane1source1.r_le_out = np.array(all_r_le_out)
    print(LE_plane1source1.r_le_in.shape)

    # sb_plane_all = sb.SurfaceBrightnessAnalytical(wavel, source1, LE_plane1source1, [all_x_inter_values, all_y_inter_values, all_z_inter_values])

    # sb_plane_all.Fl = all_flux

    # # Calculate surface brightness
    # cossigma, surface = sb_plane_all.calculate_surface_brightness()
    print(np.mean(-2.5*np.log10(all_surface)-48.6))
    fig, ax = plt.subplots()
    ax.scatter(np.sqrt(all_x_inter_values**2 + all_y_inter_values**2), -2.5*np.log10(all_surface)-48.6)
    plt.show()


    n_speci = f"InfPlane_dt0_loop_ct{int(ct / fc.dtoy)}_loc{params}_dz0{round(dz0 / fc.pctoly, 2)}"
    
    # Initialize and calculate the LE image from LE and surface brightness
    le_img = LEImageAnalytical(LE_plane1source1, plane1, all_surface, pixel_resolution = 0.2, cmap = 'magma_r')
    surface_val, surface_img, x_img, y_img, z_img_ly = le_img.create_le_surface()

    print(x_img.shape, surface_val.shape, surface_val.max(), surface_val.min())
    print(surface_val[surface_val > 0].shape)
    fig, ax = plt.subplots(1,1, figsize = (8,8))
    # mags = np.zeros_like(surface_val)
    # mags = -2.5*np.log10(surface_val * (1/(9.461e+17)**3) *(1/(3.154e+7)**2)) -48.6
    mags = -2.5*np.log10(surface_val) -48.6
    
    mags = np.nan_to_num(mags, nan=0.0, posinf=0.0, neginf=None)
    
    aja = ax.imshow(mags, origin = "lower", cmap="RdPu")
    plt.colorbar(aja)
    ax.set_title(n_speci)
    plt.show()
"""
    info_le = [{'r_in (arcsec)': le_img.r_le_in,
                'r_out (arcsec)': le_img.r_le_out,
                'act (arcsec)': le_img.act,
                'bct (arcsec)': le_img.bct,
                'params': params,
                'dz0 (ly)': dz0,
                'dt0 (days)': int(round(dt0 / fc.dtoy)),
                'ct (days)': int(round(ct / fc.dtoy))}]
    
    

    
    
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

        np.save(pathg+"\\ximg_arcsec"+n_speci+".npy", x_img)
        np.save(pathg+"\\yimg_arcsec"+n_speci+".npy", y_img)
        np.save(pathg+"\\zimgly_arcsec"+n_speci+".npy", z_img_ly)


        # # -- save the intersection points in xyz system in ly
        # np.save(pathg+"\\x_inter_ly"+n_speci+".npy", x_inter_values)
        # np.save(pathg+"\\y_inter_ly"+n_speci+".npy", y_inter_values)
        # np.save(pathg+"\\z_inter_ly"+n_speci+".npy", z_inter_values)

        # np.save(pathg+"\\z_inter_ly"+n_speci+".npy", z_inter_values)

    if show_plots == True:
        plt.show()
    #     figs.show()

"""

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
    dt0 =  0 #parameters['dt0'] * fc.dtoy #years
    d =  parameters['d'] * fc.pctoly#parameters['d']
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