import numpy as np
from scipy import integrate
from scipy import optimize #minimize
import sys
from definitions import CONFIG_PATH_constants, ROOT_DIR
sys.path.append(CONFIG_PATH_constants+"\\code")
import var_constants as vc
import fix_constants as fc
import dust_constants as dc
import calculate_scattering_function as csf
sys.path.append(ROOT_DIR)
from DustShape import SphericalBlub


class SurfaceBrightness:
    def __init__(self, source, LE, xyz_intersection):
        self.F = source.Flmax
        self.dt0 = source.dt0
        self.d = source.d
        self.ct = LE.ct
        self.dz0 = LE.dz0
        self.r_le2 = LE.calculate_rle2()
        self.x_inter_values = xyz_intersection[0]
        self.y_inter_values = xyz_intersection[1]
        self.z_inter_values = xyz_intersection[2]
        self.surface = 0
        self.cossigma = 0

    def calculate_surface_brightness(self, DustShape, x_indices=0, y_indices=0, size_x=0, size_y=0, arr_init=0, count=0, inverse=0, unique_pairs=0):
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

        if isinstance(DustShape, SphericalBlub):
            print("integral arcsenh")
            z_matrix = np.zeros((size_y, size_x, count.max()))
            print(size_y, size_x)
            xy_matrix = np.zeros((size_y, size_x, 2))
            self.sb_true_matrix = np.zeros((xy_matrix.shape[0], xy_matrix.shape[1]))
            def rhos(z, ct=self.ct):
                return np.sqrt(2 * z * ct + (ct)**2 )
        #     Inte_z = integrate.simpson(y=1/r, x=self.z_inter_values)
        else:
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
            if isinstance(DustShape, SphericalBlub):
                x_index = x_indices[ik] - 1  # Adjust for 0-based indexing
                y_index = y_indices[ik] - 1
                z_matrix[y_index, x_index, :] = arr_init[inverse[ik]]
                xy_matrix[y_index, x_index, :] = unique_pairs[inverse[ik]]
                mins = np.nanmin(arr_init[inverse[ik]])
                maxs = np.nanmax(arr_init[inverse[ik]])
                # print(mins, maxs)
                Inte_z_min = np.arcsinh(mins/rhos(mins))
                Inte_z_max = np.arcsinh(maxs/rhos(maxs))
                if ((mins == maxs) & (mins != 0)):
                    Inte_z_min = np.arcsinh((mins-0.00001)/rhos(mins-0.00001))
                temporal_s =  Ir * S[ik] * (Inte_z_max-Inte_z_min) / ( 4 * np.pi )
                self.sb_true_matrix[y_index, x_index] = temporal_s
                
            else:
                Inte_z = self.dz0
                self.sb_true_matrix[ik] = Ir * S[ik] * Inte_z / ( 4 * np.pi * r[ik] * rhodrho[ik] )

        if isinstance(DustShape, SphericalBlub):
            return self.cossigma, self.sb_true_matrix, xy_matrix, z_matrix
        else:
            return self.cossigma, self.sb_true_matrix



    def calculate_surface_brightness_3(self, DustShape, xy_matrix, size_x, size_y):
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

        self.sb_true_matrix = np.zeros((size_y, size_x))

        sizeg, waveg, Qcarb, gcarb = csf.load_data()
        # Calculate the scattering integral and the surface brightness
        S = np.zeros((size_y, size_x))
        Qscs, gs, sizes, carbon_distribution = csf.calculate_scattering_function_values(sizeg, waveg, dc.wavel, Qcarb, gcarb)

        for i in range(size_x):
            for j in range(size_y):
                xs = xy_matrix[j, i, 0]
                ys = xy_matrix[j, i, 1]
                zs = xy_matrix[j, i, 2]
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
                arr = xy_matrix[j, i, 2][xy_matrix[j, i, 2]!=0]
                if arr.shape[0] == 0:
                    self.sb_true_matrix[j,i] = 0
                else:
                    self.sb_true_matrix[j,i] = xy_matrix[j, i, 3] * Ir * S[j,i] * self.dz0 / ( 4 * np.pi * r* rhodrho )
    
        
        return self.cossigma, self.sb_true_matrix