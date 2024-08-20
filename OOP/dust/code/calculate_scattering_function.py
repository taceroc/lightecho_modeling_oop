import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from astropy import units as u
from scipy.special import erf
from scipy import integrate

import sys
# sys.path.append(r"C:\\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\code\\dust\\code")

sys.path.append(r"C:\\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\lightecho_modeling_oop\\OOP\\dust\\code")

import var_constants as vc
import dust_constants as dc
import fix_constants as fc
import scattering_function as sf
import size_dist as sd


def calculate_scattering_function_values(wave, sizeg, waveg, Qcarb, gcarb,
                                        Qsili, gsili):
    B1 = sd.Bi_carb(dc.a01, dc.bc1)
    # calculate B2
    B2 = sd.Bi_carb(dc.a02, dc.bc2)
    carbon_distribution = [sd.Dist_carb(idx, B1, B2).value for idx in 1e-4*sizeg*u.cm] #in cm
    silicone_distribution = [sd.Dist_sili(idx).value for idx in 1e-4*sizeg*u.cm] #in cm
    
    Qc_scs, gc_s, sizes, carbon_distribution = sf.dS_pre(sizeg, waveg, wave, Qcarb, gcarb, carbon_distribution)
    Qs_scs, gs_s, sizes, silicone_distribution = sf.dS_pre(sizeg, waveg, wave, Qsili, gsili, silicone_distribution)

    return  sizes, Qc_scs, gc_s, carbon_distribution, Qs_scs, gs_s, silicone_distribution

def calculate_scattering_function(mu, sizes, Qc_scs, gc_s, carbon_distribution,
                                  Qs_scs, gs_s, silicone_distribution):

    # Qscs, gs, sizes, carbon_distribution = calculate_scattering_function_values(mu, sizeg, waveg, wave, Qcarb, gcarb)
    
    ds_c = sf.dS(mu, Qc_scs, gc_s, sizes, carbon_distribution)
    S_c = sf.S(ds_c, sizes)

    ds_s = sf.dS(mu, Qs_scs, gs_s, sizes, silicone_distribution)
    S_s = sf.S(ds_s, sizes)
    return ds_c+ds_s, S_c+S_s

def load_data():
    # path_dustdata = '/content/drive/MyDrive/LE2023/dust/data/'
    # path_dustdata = r"C:\\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\code\\dust\\data"
    path_dustdata = r"C:\\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\lightecho_modeling_oop\\OOP\\dust\\data"
    # pull out available wavelengths for g values, convert to cm from um, and take the 
    waveg = np.loadtxt(path_dustdata+r'\\dustmodels_WD01\\LD93_wave.dat', unpack=True) #micronm
    # pull out available sizes for the g values, convert to cm from um, and take the log
    sizeg = np.loadtxt(path_dustdata+r'\\dustmodels_WD01\\LD93_aeff.dat', unpack=True) #micron

    # older models used for g (degree of forward scattering) values
    # carbonaceous dust
    carbonQ = path_dustdata+r'\\dustmodels_WD01\\Gra_81.dat'
    Qcarb_sca = np.loadtxt(carbonQ, usecols=(2), unpack=True)
    Qcarb_abs = np.loadtxt(carbonQ, usecols=(1), unpack=True)
    Qcarb = Qcarb_sca / (Qcarb_sca + Qcarb_abs)

    # silicate dust
    siliconQ = path_dustdata+r'\\dustmodels_WD01\\suvSil_81.dat'
    # Qsil = np.loadtxt(siliconQ, usecols=(2), unpack=True)
    Qsili_sca = np.loadtxt(siliconQ, usecols=(2), unpack=True)
    Qsili_abs = np.loadtxt(siliconQ, usecols=(1), unpack=True)
    Qsili = Qsili_sca / (Qsili_sca + Qsili_abs)

    # older models used for g (degree of forward scattering) values
    # carbonaceous dust
    # carbong = path_dustdata+r'\\dustmodels_WD01\\Gra_81.dat'
    gcarb = np.loadtxt(carbonQ, usecols=(3), unpack=True)
    # silicate dust
    # silicong = path_dustdata+r'\\dustmodels_WD01\\suvSil_81.dat'
    gsili = np.loadtxt(siliconQ, usecols=(3), unpack=True)

    return sizeg, waveg, Qcarb, gcarb, Qsili, gsili


def main(mu, sizes, Qc_scs, gc_s, carbon_distribution, Qs_scs, gs_s, silicone_distribution):
    # sizeg, waveg, Qcarb, gcarb = load_data

    ds, S = calculate_scattering_function(mu, sizes, Qc_scs, gc_s, carbon_distribution,
                                  Qs_scs, gs_s, silicone_distribution)

    return ds, S


# if __name__ == "__main__":
#     ds, S = main(mu, wave)