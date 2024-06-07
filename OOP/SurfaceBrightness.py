import numpy as np
import sys
from definitions import CONFIG_PATH_DUST
sys.path.append(CONFIG_PATH_DUST)
import var_constants as vc
import fix_constants as fc
import dust_constants as dc
import calculate_scattering_function as csf


class SurfaceBrightness:
    def __init__(self, source, LE):
        self.F = source.Flmax
        self.dt0 = source.dt0
        self.d = source.d
        self.ct = LE.ct
        self.dz0 = LE.dz0
        self.r_le2 = LE.calculate_rle2()
        # self.x_inter_values = xyz_intersection[0]
        # self.y_inter_values = xyz_intersection[1]
        # self.z_inter_values = xyz_intersection[2]
        self.surface = 0
        self.cossigma = 0

class SurfaceBrightnessAnalytical(SurfaceBrightness):
    def __init__(self, source, LE, xyz_intersection):
        super().__init__(source, LE)
        self.x_inter_values = xyz_intersection[0]
        self.y_inter_values = xyz_intersection[1]
        self.z_inter_values = xyz_intersection[2]
        self.sb_true_matrix = 0

    def calculate_surface_brightness(self):
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

            Arguments:
                x_inter, y_inter, z_inter: intersection paraboloid + dust in ly
                dz0: thickness dust in ly
                ct: time where the LE is observed in y

            Return
                Surface brightness in units of kg/ly^3 [erg/(s cm4)]
                cos(scatter angle)
        """
        
        
        # Sugerman 2003 after eq 15 F(lambda) = 1.25*F(lambda, tmax)*0.5*dt0
        #F 1.08e-14 # watts / m2
        F = self.F * (fc.ytos**3) # kg,ly,y
        Ir = 1.25*F*0.5*self.dt0 * fc.n_H * fc.c
        
        # calculate r, source-dust
        r = np.sqrt(self.x_inter_values**2 + self.y_inter_values**2 + self.z_inter_values**2)
        r_le = np.sqrt(self.r_le2)

        self.sb_true_matrix = np.zeros(len(r))
        rhos = np.sqrt(2 * self.z_inter_values * self.ct + (self.ct)**2 )
        half_obs_thickness = np.sqrt( (self.ct / rhos) ** 2 * self.dz0 ** 2 + ( (rhos * fc.c / (2 * self.ct)) + ( fc.c * self.ct / (2 * rhos) )) ** 2 * self.dt0  ** 2 ) / 2
        rhodrho = rhos * half_obs_thickness

        # dust-observer
        ll = np.sqrt(self.x_inter_values**2 + self.y_inter_values**2 + (self.z_inter_values - self.d)**2)
        # calcualte scatter angle, angle between source-dust , dust-observer
        self.cossigma = ((self.x_inter_values**2 + self.y_inter_values**2 + self.z_inter_values * (self.z_inter_values - self.d)) / (r * ll))
        sizeg, waveg, Qcarb, gcarb = csf.load_data()
        # Calculate the scattering integral and the surface brightness
        S = np.zeros(len(r))
        Qscs, gs, sizes, carbon_distribution = csf.calculate_scattering_function_values(sizeg, waveg, dc.wavel, Qcarb, gcarb)

        for ik, rm in enumerate(self.cossigma):
            if ((rm >= -1) and (rm <= 1)):
                ds, Scm = csf.main(rm, Qscs, gs, sizes, carbon_distribution) # 1.259E+00 in um
                S[ik] = (Scm[0][0] * fc.pctoly**2) / (100 * fc.pctom )**2 # conver to ly
            else:
                S[ik] = 0
                # print("no cosi")
            Inte_z = self.dz0
            self.sb_true_matrix[ik] = Ir * S[ik] * Inte_z / ( 4 * np.pi * r[ik] * rhodrho[ik] )

        return self.cossigma, self.sb_true_matrix

class SurfaceBrightnessBulb(SurfaceBrightness):
    def __init__(self, source, LE, xy_matrix, z_matrix):
        super().__init__(source, LE)
        self.xy_matrix = xy_matrix
        self.z_matrix = z_matrix
        self.size_x = LE.size_x
        self.size_y = LE.size_y
        self.size_z = LE.size_z
        self.sb_true_matrix = 0

    def calculate_surface_brightness(self):
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

            Arguments:
                x_inter, y_inter, z_inter: intersection paraboloid + dust in ly
                dz0: thickness dust in ly
                ct: time where the LE is observed in y

            Return
                Surface brightness in units of kg/ly^3 [erg/(s cm4)]
                cos(scatter angle)
        """
        
        # Sugerman 2003 after eq 15 F(lambda) = 1.25*F(lambda, tmax)*0.5*dt0
        #F 1.08e-14 # watts / m2
        F = self.F * (fc.ytos**3) # kg,ly,y
        Ir = 1.25*F*0.5*self.dt0 * fc.n_H * fc.c
        
        # calculate r, source-dust
        r_le = np.sqrt(self.r_le2)

        self.sb_true_matrix = np.zeros((self.size_y, self.size_x))

        sizeg, waveg, Qcarb, gcarb = csf.load_data()
        # Calculate the scattering integral and the surface brightness
        S = np.zeros((self.size_y, self.size_x))
        Qscs, gs, sizes, carbon_distribution = csf.calculate_scattering_function_values(sizeg, waveg, dc.wavel, Qcarb, gcarb)

        for i in range(self.size_x):
            for j in range(self.size_y):
                xs = self.xy_matrix[j, i, 0]
                ys = self.xy_matrix[j, i, 1]
                zs = self.z_matrix[j, i, :]
                ll = np.sqrt(xs**2 + ys**2 + (zs - self.d)**2)
                r = np.sqrt(xs**2 + ys**2 + zs**2)
                rhos = np.sqrt(2 * zs * self.ct + (self.ct)**2 )
                half_obs_thickness = np.sqrt( (self.ct / rhos) ** 2 * self.dz0 ** 2 + ( (rhos * fc.c / (2 * self.ct)) + ( fc.c * self.ct / (2 * rhos) )) ** 2 * self.dt0  ** 2 ) / 2
                rhodrho = np.mean(rhos * half_obs_thickness)
                cossigma = ((xs**2 + ys**2 + zs * (zs - self.d)) / (r * ll))
                # print(r*ll)
                cossigmam = np.mean(cossigma)
                if ((cossigmam >= -1) and (cossigmam <= 1)):
                    ds, Scm = csf.main(cossigmam, Qscs, gs, sizes, carbon_distribution) # 1.259E+00 in um
                    S[j,i] = (Scm[0][0] * fc.pctoly**2) / (100 * fc.pctom )**2 # conver to ly
                else:
                    S[j,i] = 0
                    # print("no cosi")
                arr = self.z_matrix[j,i][self.z_matrix[j,i]!=0]
                if arr.shape[0] == 0:
                    self.sb_true_matrix[j,i] = 0
                else:
                    temporal_s =  Ir * S[j,i] * self.dz0 / ( 4 * np.pi * np.mean(r) * rhodrho )
                    self.sb_true_matrix[j,i] = temporal_s
        
        return self.cossigma, self.sb_true_matrix

class SurfaceBrightnessDustSheetPlane(SurfaceBrightness):

    def __init__(self, source, LE, sheetdust, xy_matrix):
        super().__init__(source, LE)
        self.xy_matrix = xy_matrix
        self.sb_true_matrix = 0
        self.size_x = sheetdust.side_x
        self.size_y = sheetdust.side_y

    def calculate_surface_brightness(self):
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
        #F 1.08e-14 # watts / m2
        F = self.F * (fc.ytos**3) # kg,ly,y
        Ir = 1.25*F*0.5*self.dt0 * fc.n_H * fc.c
        
        # calculate r, source-dust
        r_le = np.sqrt(self.r_le2)

        self.sb_true_matrix = np.zeros((self.size_y, self.size_x))
        

        sizeg, waveg, Qcarb, gcarb = csf.load_data()
        # Calculate the scattering integral and the surface brightness
        S = np.zeros((self.size_y, self.size_x))
        Qscs, gs, sizes, carbon_distribution = csf.calculate_scattering_function_values(sizeg, waveg, dc.wavel, Qcarb, gcarb)

        for i in range(self.size_x):
            for j in range(self.size_y):
                xs = self.xy_matrix[j, i, 0]
                ys = self.xy_matrix[j, i, 1]
                zs = self.xy_matrix[j, i, 2]
                ll = np.sqrt(xs**2 + ys**2 + (zs - self.d)**2)
                r = np.sqrt(xs**2 + ys**2 + zs**2)
                cossigma = ((xs**2 + ys**2 + zs * (zs - self.d)) / (r * ll))
                rhos = np.sqrt(2 * zs * self.ct + (self.ct)**2 )
                half_obs_thickness = np.sqrt( (self.ct / rhos) ** 2 * self.dz0 ** 2 + ( (rhos * fc.c / (2 * self.ct)) + ( fc.c * self.ct / (2 * rhos) )) ** 2 * self.dt0  ** 2 ) / 2
                rhodrho = rhos * half_obs_thickness
                # print(r*ll)
                cossigmam = np.mean(cossigma)
                if ((cossigmam >= -1) and (cossigmam <= 1)):
                    ds, Scm = csf.main(cossigmam, Qscs, gs, sizes, carbon_distribution) # 1.259E+00 in um
                    S[j,i] = (Scm[0][0] * fc.pctoly**2) / (100 * fc.pctom )**2 # conver to ly
                else:
                    S[j,i] = 0
                    # print("no cosi")
                arr = self.xy_matrix[j, i, 2][self.xy_matrix[j, i, 2]!=0]
                if arr.shape[0] == 0:
                    self.sb_true_matrix[j,i] = 0
                else:
                    self.sb_true_matrix[j,i] = self.xy_matrix[j, i, 3] * Ir * S[j,i] * self.dz0 / ( 4 * np.pi * r* rhodrho )
    
        
        return self.cossigma, self.sb_true_matrix