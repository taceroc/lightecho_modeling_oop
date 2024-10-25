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
            lambda: given wavelength in micrometer [lenght]
            dz0: dust thickness [lenght]
            r: position dust [lenght]
            rhodrho: x-y of LE [lenght^2]
            mu: cos theta, theta: scattering angle
            Q: albedo
            sigma: cross section [lenght^2]
            Phi: scattering function
            f(a): dust distribution [1/lenght]
            S: scattering integral [lenght^2]
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
            np.sqrt(
                (self.ct / self.rhos) ** 2 * self.dz0**2
                + ((self.rhos * fc.c / (2 * self.ct)) + (fc.c * self.ct / (2 * self.rhos))) ** 2
                * self.dt0**2
            )
            / 2
        )
        self.rhodrho = self.rhos * self.half_obs_thickness
        # return self.rhodrho, self.rhos, half_obs_thickness
    
    def light_curve_integral(self):
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
        self.Fl = []
        self.rhos_half()
        # self.lc['time'] = self.lc['time'] * fc.dtoy
        # t_tilde =  self.ct - (np.sqrt(self.rhos**2 + self.z_inter_values**2) / fc.c) - (np.abs(self.z_inter_values) / fc.c)
        # t_tilde = self.ct - np.linalg.norm([self.x_inter_values, self.y_inter_values, self.z_inter_values], axis=0) + (np.abs(self.z_inter_values) / fc.c)
        self.t_tilde = (self.ct + self.lc['time'][self.lc['mag'] == self.lc['mag'].min()] - 
                        np.linalg.norm([self.x_inter_values, self.y_inter_values, self.z_inter_values], axis=0) -
                        np.linalg.norm([self.x_inter_values, self.y_inter_values, (self.d - self.z_inter_values)], axis=0))
        # print((self.d+self.t_tilde)/fc.dtoy)
        # add +d/c because the zero of your data is when detecting 1st light
        for it, t_tildi in enumerate(self.t_tilde):
            if self.t_tilde[it]+self.d < 0:
                self.t_tilde[it] = -self.d     
            # print(self.lc['time']<=t_tildi)
            time_up = self.lc['time'][self.lc['time']<=self.t_tilde[it]+self.d]
            mag_upto = self.lc['mag'][self.lc['time']<=self.t_tilde[it]+self.d]
            flux_upto = 10**((-48.6 - mag_upto) / 2.5)
                
            self.Fl.append(integrate.simpson(flux_upto, time_up))
        
    def load_dust_values(self):
        sizeg, waveg, Qcarb, gcarb, Qsili, gsili= csf.load_data()
        # Calculate the scattering integral and the surface brightness
        (sizes, Qc_scs, gc_s, carbon_distribution, 
         Qs_scs, gs_s, silicone_distribution) = csf.calculate_scattering_function_values(
            self.wavel, sizeg, waveg, Qcarb, gcarb, Qsili, gsili
        )

        return (sizes, Qc_scs, gc_s, carbon_distribution, 
                Qs_scs, gs_s, silicone_distribution)
    

class SurfaceBrightnessAnalytical(SurfaceBrightness):
    def __init__(self, wavel, source, LE, xyz_intersection):
        super().__init__(wavel, source, LE)
        self.x_inter_values = xyz_intersection[0]
        self.y_inter_values = xyz_intersection[1]
        self.z_inter_values = xyz_intersection[2]
        

    def calculate_surface_brightness(self):
        """
        Arguments:
            x_inter, y_inter, z_inter: intersection paraboloid + dust in ly
            dz0: thickness dust in ly
            ct: time where the LE is observed in y

        Return
            Surface brightness in units of kg/ly^3 [erg/(s cm4)]
            cos(scatter angle)
        """

        # calculate r, source-dust
        r = np.sqrt(
            self.x_inter_values**2 + self.y_inter_values**2 + self.z_inter_values**2
        )

        # Sugerman 2003 after eq 15 F(lambda) = 1.25*F(lambda, tmax)*0.5*dt0
        # F 1.08e-14 # watts / m2
        Ir = np.ones(len(r))
        if self.lc == None:
            self.rhos_half()
            Fl = self.Fl * (fc.ytos**3)  # kg,ly,y
            Ir = Ir * 1.25 * Fl * 0.5 * self.dt0 * fc.n_H * fc.c
        else:
            super().light_curve_integral()
            Fl = np.array(self.Fl) * (fc.ytos**3)  # kg,ly,y
            Ir = Fl * fc.n_H * fc.c

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
                S[ik] = (Scm[0][0] * fc.pctoly**2) / (
                    100 * fc.pctom
                ) ** 2  # conver to ly
            else:
                S[ik] = 0
                # print("no cosi")
            Inte_z = self.dz0
            self.sb_true_matrix[ik] = (
                Ir[ik] * S[ik] * Inte_z / (4 * np.pi * r[ik] * self.rhodrho[ik])
            )

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
            Surface brightness in units of kg/ly^3 [erg/(s cm4)]
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

    def calculate_surface_brightness(self):
        """
        Arguments:
            size_x, size_y: pixel size dust image
            x_inter, y_inter, z_inter: intersection paraboloid + dust in ly
            dz0: thickness dust in ly
            ct: time where the LE is observed in y

        Return
            Surface brightness in units of kg/ly^3 [erg/(s cm4)]
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
                self.x_inter_values = self.xy_matrix[j, i, 0]
                self.y_inter_values = self.xy_matrix[j, i, 1]
                self.z_inter_values = self.xy_matrix[j, i, 2]
                ll = np.sqrt(self.x_inter_values**2 + self.y_inter_values**2 + (self.z_inter_values - self.d) ** 2)
                r = np.sqrt(self.x_inter_values**2 + self.y_inter_values**2 + self.z_inter_values**2)
                cossigma = (self.x_inter_values**2 + self.y_inter_values**2 + self.z_inter_values * (self.z_inter_values - self.d)) / (r * ll)
                super().rhos_half()
                # print(r*ll)
                cossigmam = np.mean(cossigma)
                if (cossigmam >= -1) and (cossigmam <= 1):
                    ds, Scm = csf.main(
                        cossigmam, sizes, 
                        Qc_scs, gc_s, carbon_distribution, 
                        Qs_scs, gs_s, silicone_distribution
                    )  # 1.259E+00 in um
                    S[j, i] = (Scm[0][0] * fc.pctoly**2) / (
                        100 * fc.pctom
                    ) ** 2  # conver to ly
                else:
                    S[j, i] = 0
                    # print("no cosi")
                arr = self.xy_matrix[j, i, 2][self.xy_matrix[j, i, 2] != 0]
                if arr.shape[0] == 0:
                    self.sb_true_matrix[j, i] = 0
                else:
                    self.sb_true_matrix[j, i] = (
                        self.xy_matrix[j, i, 3]
                        * Ir
                        * S[j, i]
                        * self.dz0
                        / (4 * np.pi * r * self.rhodrho)
                    )

        return self.cossigma, self.sb_true_matrix
