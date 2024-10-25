import numpy as np
import sys
import scipy.integrate as integrate
from definitions import CONFIG_PATH_DUST

sys.path.append(CONFIG_PATH_DUST)
import var_constants as vc
import fix_constants as fc
import dust_constants as dc
import calculate_scattering_function as csf


class SurfaceBrightness:
    def __init__(self, wavel, source, LE, lc=None):
        """
        Calculate the surface brightness at a position r = (x_inter, y_inter, z_inter):
        Sugermann 2003 equation 7:
            SB(lambda, t) = F(lambda)nH(r) * (c dz0 / (4 pi r rhodrho) )* S(lambda, mu)
            S(lambda, mu) = \int Q(lamdda, a) sigma Phi(mu, lambda, a) f(a) da
            lambda: given wavelength in micrometer [1/lenght]
            dz0: dust thickness [lenght]
            r: position dust [lenght]
            rhodrho: x-y of LE [lenght^2]
            mu: cos theta, theta: scattering angle
            Q: albedo
            sigma: cross section [lenght^2]
            Phi: scattering function
            f(a): dust distribution [1/lenght]
            S: scattering integral [lenght^2]

            Return units [mass / (time^2 lenght^3)] >> [flux / lenght^3]
        """
        self.wavel = wavel
        self.Fl = source.Flmax
        self.dt0 = source.dt0
        self.d = source.d
        self.ct = LE.ct
        self.dz0 = LE.dz0
        self.r_le2 = LE.calculate_rle2()
        self.rhos = 0
        self.surface = 0
        self.cossigma = 0
        self.lc = lc
        self.sb_true_matrix = 0

    def define_bandpass_rubin(self):
#         {'band_name': ['lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty'],
#  'min_wave_A': [3104.9999999999995,
#   3865.9999999999995,
#   5369.999999999999,
#   6759.999999999999,
#   8029.999999999998,
#   9083.999999999998],
#  'max_wave_A': [4085.9999999999995,
#   5669.999999999999,
#   7059.999999999999,
#   8329.999999999998,
#   9384.999999999998,
#   10944.999999999998]}
        
        lsstu = [np.array([3104.9999999999995, 4085.9999999999995]) * 1e-4, 'u', 24.5, 10**((-48.6 - 24.5) / 2.5)]
        lsstg = [np.array([3865.9999999999995, 5669.999999999999]) * 1e-4, 'g', 25, 10**((-48.6 - 25) / 2.5)]
        lsstr = [np.array([5369.999999999999, 7059.999999999999]) * 1e-4, 'r', 25, 10**((-48.6 - 25) / 2.5)]
        lssti = [np.array([6759.999999999999, 8329.999999999998]) * 1e-4, 'i', 24, 10**((-48.6 - 24) / 2.5)]
        lsstz = [np.array([8029.999999999998, 9384.999999999998]) * 1e-4, 'z', 23.5, 10**((-48.6 - 23.5) / 2.5)]
        lssty = [np.array([9083.999999999998, 10944.999999999998]) * 1e-4, 'y', 23.5, 10**((-48.6 - 23.5) / 2.5)]

        for ixi, bandpasses in enumerate([lsstu, lsstg, lsstr, lssti, lsstz, lssty]):
            if bandpasses[0][0] <= self.wavel <= bandpasses[0][1]:
                # band = bandpasses[1]
                self.band_pass_index = ixi


    def rhos_half(self):
        """
            Calculate the thickness of the visible light echo, the rho coordiante, Sugermann 2003. Eq 11
            Convolution of the thickness due to dust thickness and duration of pulse from source
            Arguments:
                Values of z: intersection paraboloid+dust
            
            Return:
                rhodrho: Sugermann 2003 Eq 7, rho = sqrt(x**2 + y**2)
                rhos = sqrt(x**2 + y**2)
                half_obs_thickness = thickness of LE

        """
        self.rhos = np.sqrt(self.x_inter_values**2 + self.y_inter_values**2)#np.sqrt(2 * self.z_inter_values * self.ct + (self.ct) ** 2)
        self.half_obs_thickness = (
            np.sqrt((self.ct / self.rhos) ** 2 * self.dz0**2
                + ((self.rhos * fc.c / (2 * self.ct)) + (fc.c * self.ct / (2 * self.rhos))) ** 2 * self.dt0**2)/ 2
        )
        self.rhodrho = self.rhos * self.half_obs_thickness
        return self.rhodrho, self.rhos, self.half_obs_thickness
    
    def light_curve_integral(self, tilde=0):
        """
            Calculate integral below Sugermann 2003 equation 5.
            Integral of the light curve F(lambda) = \int F(lamnda, t_tilde) dt_tilde
            t_tilde: time at the dust position (state of light curve when light interact with dust at a tiem t_tilde)
            t_tilde: between eq 4 and 5 from Sugermann 2003 (also in Xu & Crotts, 1994ApJ 435)

            Arguments:
                Values of z: intersection paraboloid+dust
                light curve: {'time': values in days centered at peak, 'mag': values in mag}
            
            Return:
                Flux at wavelenght lambda
        """
        self.Fl = 0 # it stores the flux for each x,y,z, that gives a differnte LC time
        # self.t_tilde = self.lc['time'][self.lc['mag'] == self.lc['mag'].min()] 
        lc_integral_time = self.lc['time'][self.lc['time'] <= tilde] 
        lc_integral_mag = self.lc['mag'][self.lc['time'] <= tilde] 
        # flux_upto = 10**((-48.6 - lc_integral_mag) / 2.5)
        flux_upto = 10**((-lc_integral_mag) / 5) * 4.26e-20
        print(flux_upto)
        # print(lc_integral_mag - 5)
        # print(lc_integral_time/fc.dtoy)
        self.Fl = integrate.simpson(flux_upto, lc_integral_time)
        print("ITNEGRALL", self.Fl)
    
        # print('FL intergra')
        # print(find_nearest(self.lc['time'], self.t_tilde[0]+self.d))
            # print(self.Fl)
         
        
    def load_dust_values(self):
        sizeg, waveg, Qcarb, gcarb, Qsili, gsili= csf.load_data()
        # Calculate the scattering integral and the surface brightness
        (sizes, Qc_scs, gc_s, carbon_distribution, 
         Qs_scs, gs_s, silicone_distribution) = csf.calculate_scattering_function_values(
            self.wavel, sizeg, waveg, Qcarb, gcarb, Qsili, gsili
        )

        return (sizes, Qc_scs, gc_s, carbon_distribution, 
                Qs_scs, gs_s, silicone_distribution)
    

    def determine_flux_time_loop(self, tilde=0):
        # r = np.sqrt(
            # self.x_inter_values**2 + self.y_inter_values**2 + self.z_inter_values**2)
        self.Ir = 0 #np.ones(len(r))
        if self.lc == None:
            # self.rhos_half()
            Fl = self.Fl #* (fc.ytos**3)  # kg,ly,y
            self.Ir = self.Ir * Fl * fc.n_H * fc.c#*1.25 * 0.5 * self.dt0  #* fc.c
        else:
            self.light_curve_integral(tilde)
            Fl = np.array(self.Fl) #* (fc.ytos**2)  # kg,ly,y
            # print(self.Fl)
            self.Ir = Fl * fc.n_H 
            # print(Ir)
    

class SurfaceBrightnessAnalytical(SurfaceBrightness):
    def __init__(self, wavel, source, LE, xyz_intersection ):
        super().__init__(wavel, source, LE)
        self.x_inter_values = xyz_intersection[0]
        self.y_inter_values = xyz_intersection[1]
        self.z_inter_values = xyz_intersection[2]
        

    def calculate_surface_brightness(self, tilde):
        """
        Arguments:
            x_inter, y_inter, z_inter: intersection paraboloid + dust in ly
            dz0: thickness dust in ly
            ct: time where the LE is observed in y

        Return
            Surface brightness in units of kg/(ly^2 y^2) ###kg/ly^3 [erg/(s cm4)]
            cos(scatter angle)
        """

        # calculate r, source-dust
        r = np.sqrt(
            self.x_inter_values**2 + self.y_inter_values**2 + self.z_inter_values**2
        )

        # Sugerman 2003 after eq 15 F(lambda) = 1.25*F(lambda, tmax)*0.5*dt0
        # F 1.08e-14 # watts / m2
        # Ir = np.ones(len(r))
        if self.lc == None:
            super().determine_flux_time_loop()
        else:
            super().determine_flux_time_loop(tilde)
        

        self.sb_true_matrix = np.zeros(len(r))
        # rhodrho, rhos, half_obs_thickness = super().rhos_half()
        # dust-observer
        ll = np.sqrt(
            self.x_inter_values**2
            + self.y_inter_values**2
            + (self.z_inter_values - self.d) ** 2
        )
        # calcualte scatter angle, angle between source-dust , dust-observer
        self.cossigma = (
            self.x_inter_values**2
            + self.y_inter_values**2
            + self.z_inter_values * (self.z_inter_values - self.d)
        ) / (r * ll)
        S = np.zeros(len(r))
        (sizes, Qc_scs, gc_s, carbon_distribution, 
         Qs_scs, gs_s, silicone_distribution) = super().load_dust_values()
        for ik, rm in enumerate(self.cossigma):
            if (rm >= -1) and (rm <= 1):
                ds, Scm = csf.main(
                    rm, sizes, 
                    Qc_scs, gc_s, carbon_distribution, 
                    Qs_scs, gs_s, silicone_distribution
                )  # 1.259E+00 in um
                S[ik] = (Scm[0][0])# * fc.pctoly**2) / ((100 * fc.pctom) ** 2 )
            else:
                S[ik] = 0
                # print("no cosi")
            Inte_z = self.dz0 * 9.461e+17
            # print(r[ik])
            self.sb_true_matrix[ik] = (
                self.Ir * S[ik] * Inte_z / (4 * np.pi * (r[ik]* 9.461e+17)**2)
            )  
            # print(S[ik])

        return self.cossigma, self.sb_true_matrix


class SurfaceBrightnessBulb(SurfaceBrightness):
    def __init__(self, wavel, source, LE, xy_matrix, z_matrix):
        super().__init__(wavel, source, LE)
        self.xy_matrix = xy_matrix
        self.z_matrix = z_matrix
        self.size_x = LE.size_x
        self.size_y = LE.size_y
        self.size_z = LE.size_z

    def calculate_surface_brightness(self):
        """
        Arguments:
            x_inter, y_inter, z_inter: intersection paraboloid + dust in ly
            dz0: thickness dust in ly
            ct: time where the LE is observed in y

        Return
            Surface brightness in units of kg/(ly^2 y^2) ##kg/ly^3 [erg/(s cm4)]
            cos(scatter angle)
        """

        # Sugerman 2003 after eq 15 F(lambda) = 1.25*F(lambda, tmax)*0.5*dt0
        # F 1.08e-14 # watts / m2
        Fl = self.Fl * (fc.ytos**3)  # kg,ly,y
        Ir = 1.25 * Fl * 0.5 * self.dt0 * fc.n_H * fc.c

        # calculate r, source-dust
        r_le = np.sqrt(self.r_le2)

        self.sb_true_matrix = np.zeros((self.size_y, self.size_x))

        # Calculate the scattering integral and the surface brightness
        S = np.zeros((self.size_y, self.size_x))
        (sizes, Qc_scs, gc_s, carbon_distribution, 
         Qs_scs, gs_s, silicone_distribution) = super().load_dust_values()

        for i in range(self.size_x):
            for j in range(self.size_y):
                xs = self.xy_matrix[j, i, 0]
                ys = self.xy_matrix[j, i, 1]
                zs = self.z_matrix[j, i, :]
                ll = np.sqrt(xs**2 + ys**2 + (zs - self.d) ** 2)
                r = np.sqrt(xs**2 + ys**2 + zs**2)
                rhodrho, rhos, half_obs_thickness = super().rhos_half(zs)
                rhodrho = np.mean(rhodrho)
                cossigma = (xs**2 + ys**2 + zs * (zs - self.d)) / (r * ll)
                # print(r*ll)
                cossigmam = np.mean(cossigma)
                if (cossigmam >= -1) and (cossigmam <= 1):
                    ds, Scm = csf.main(
                        cossigmam, sizes, 
                        Qc_scs, gc_s, carbon_distribution, 
                        Qs_scs, gs_s, silicone_distribution
                        )   # 1.259E+00 in um
                    S[j, i] = (Scm[0][0] * fc.pctoly**2) / (
                        100 * fc.pctom
                    ) ** 2  # conver to ly
                else:
                    S[j, i] = 0
                    # print("no cosi")
                arr = self.z_matrix[j, i][self.z_matrix[j, i] != 0]
                if arr.shape[0] == 0:
                    self.sb_true_matrix[j, i] = 0
                else:
                    temporal_s = (
                        Ir * S[j, i] * self.dz0 / (4 * np.pi * np.mean(r) * rhodrho)
                    )
                    self.sb_true_matrix[j, i] = temporal_s

        return self.cossigma, self.sb_true_matrix


class SurfaceBrightnessDustSheetPlane(SurfaceBrightness):

    def __init__(self, wavel, source, LE, sheetdust, xy_matrix):
        super().__init__(wavel, source, LE)
        self.xy_matrix = xy_matrix
        self.size_x = sheetdust.side_x
        self.size_y = sheetdust.side_y

        self.x_inter_values = LE.x_inter_values
        self.y_inter_values = LE.y_inter_values
        self.z_inter_values = LE.z_inter_values

    def calculate_surface_brightness(self):
        """
        Arguments:
            size_x, size_y: pixel size dust image
            x_inter, y_inter, z_inter: intersection paraboloid + dust in ly
            dz0: thickness dust in ly
            ct: time where the LE is observed in y

        Return
            Surface brightness in units of kg/(ly^2 y^2) ##kg/ly^3 [erg/(s cm4)]
            cos(scatter angle)
        """

        # Sugerman 2003 after eq 15 F(lambda) = 1.25*F(lambda, tmax)*0.5*dt0
        # F 1.08e-14 # watts / m2

        # Ir = np.ones(len(r))
        if self.lc == None:
            # self.rhos_half()
            Fl = self.Fl * (fc.ytos**3)  # kg,ly,y
            Ir = Fl * 1.25 * Fl * 0.5 * self.dt0 * fc.n_H #* fc.c
        else:
            super().light_curve_integral()
            Fl = np.array(self.Fl) * (fc.ytos**2)  # kg,ly,y
            # print(self.Fl)
            Ir = Fl * fc.n_H #* fc.c


        # Fl = self.Fl * (fc.ytos**3)  # kg,ly,y #Fl = np.array(self.Fl) * (fc.ytos**2) * (1/1000)
        # Ir = 1.25 * Fl * 0.5 * self.dt0 * fc.n_H * fc.c

        # calculate r, source-dust
        r_le = np.sqrt(self.r_le2)

        self.sb_true_matrix = np.zeros((self.size_y, self.size_x))

        # Calculate the scattering integral and the surface brightness
        S = np.zeros((self.size_y, self.size_x))
        (sizes, Qc_scs, gc_s, carbon_distribution, 
         Qs_scs, gs_s, silicone_distribution) = super().load_dust_values()

        for i in range(self.size_x):
            for j in range(self.size_y):
                self.x_inter_values = self.xy_matrix[j, i, 0]
                self.y_inter_values = self.xy_matrix[j, i, 1]
                self.z_inter_values = self.xy_matrix[j, i, 2]
                ll = np.sqrt(self.x_inter_values**2 + self.y_inter_values**2 + (self.z_inter_values - self.d) ** 2)
                r = np.sqrt(self.x_inter_values**2 + self.y_inter_values**2 + self.z_inter_values**2)
                cossigma = (self.x_inter_values**2 + self.y_inter_values**2 + self.z_inter_values * (self.z_inter_values - self.d)) / (r * ll)

                super().light_curve_integral()
                Fl = np.array(self.Fl) * (fc.ytos**2)  # kg,ly,y
                # print(self.Fl)
                Ir = Fl * fc.n_H #* fc.c
                # print(Ir)
                # super().rhos_half()
                # print(r*ll)
                cossigmam = np.mean(cossigma)
                if (cossigmam >= -1) and (cossigmam <= 1):
                    ds, Scm = csf.main(
                        cossigmam, sizes, 
                        Qc_scs, gc_s, carbon_distribution, 
                        Qs_scs, gs_s, silicone_distribution
                    )  # 1.259E+00 in um
                    S[j, i] = (Scm[0][0] * fc.pctoly**2) / ((100 * fc.pctom) ** 2)  # conver to ly
                else:
                    S[j, i] = 0
                    # print("no cosi")
                arr = self.xy_matrix[j, i, 2][self.xy_matrix[j, i, 2] != 0]
                if arr.shape[0] == 0:
                    self.sb_true_matrix[j, i] = 0
                else:
                    self.sb_true_matrix[j, i] = (
                        self.xy_matrix[j, i, 3] * Ir * S[j, i] * self.dz0 / (4 * np.pi * r**2)
                    )

        return self.cossigma, self.sb_true_matrix
