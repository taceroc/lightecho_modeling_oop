# list of constants for dust analysis
# Refs:
#   Weingartner & Draine (2001); ApJ 548,296
#   Draine et al. (2021); ApJ 917,3
#   Chakradhari et al. (2019)
#   Riess et al. (2019)
#   Draine & Hensley (2021)
#   Sugerman (2003)

# NOTE: All constants with units are in light year - year
import astropy.units as u
from astropy.units import cds
cds.enable()

# Weingartner & Draine (2001)
# carbonaceous dust
# constants
a01 = 3.5e-8*u.cm  # cm, size of grain
a01 = a01.to(u.lyr)
a02 = 30.0e-8*u.cm  # cm, size of grain
a02 = a01.to(u.lyr)
bc = 6.0e-5  # total C abund per H in log-normal pops
bc1 = 0.75*bc
bc2 = 0.25*bc
alphag = -1.54
betag = -0.165
atg = 0.0107e-4*u.cm #cm
atg = atg.to(u.lyr)
acg = 0.428e-4*u.cm #cm
acg = acg.to(u.lyr)
Cg = 9.99e-12

# silicate dust
# constants
alphas = -2.21
betas = 0.300
ats = 0.164e-4*u.cm  # cm
ats = ats.to(u.lyr)
acs = 0.1e-4*u.cm  # cm
acs = acs.to(u.lyr)
Cs = 1.00e-13

# global constants
sig = 0.4
rho = 2.24*u.g/u.cm**3 #g/cm^3, density of graphite
mc = 1.994e-23*u.g #g, mass of 1 carbon atom
Rv = 3.1 #extinction, unitless
nH = 10/u.cm**3 #atoms/cm^3, density of hydrogen
c = 3e10*u.cm/u.s #cm/s, speed of light


# Flmax = 1.08e-14 #w/m2
Flmax = 9.12e-12 #watts/cm2um1

wavel = 7.499E-01 #um #inside the rubin limits 0.3-1microns