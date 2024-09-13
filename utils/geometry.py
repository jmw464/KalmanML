#!/usr/bin/env python

import h5py
import numpy as np
import pandas as pd


def check_module_boundary(u, v, module):
    """
    ---
    u               : bool  : local coordinate as defined on the kaggle site
    v               : bool  : local coordinate as defined on the kaggle site
    module          : int   : module id as specified in the detector file
    ---
    within_module   : bool  : If the point is within the boundary of the module trapezoid
    """
    if module.module_maxhu == module.module_minhu:
        within_module = abs(v) <= module.module_hv and abs(u) <= module.module_maxhu
    else:
        side_boundary = lambda x, side: side*(2*module.module_hv / (module.module_maxhu - module.module_minhu)) * (x - side*0.5*(module.module_maxhu + module.module_minhu))
        within_module = abs(v) <= module.module_hv and v >= side_boundary(u, 1) and v >= side_boundary(u, -1)
    return within_module


class Geometry:
    VOLUMES = np.array([7, 8, 9, 12, 13, 14, 16, 17, 18]) # volume IDs in the detector
    BARRELS = {8, 13, 17} # barrel shaped volumes
    Z_BUFFER = 50 # buffer around z to define volumes
    R_BUFFER = 25 # buffer around r to define volumes

    detector = None # Pandas dataframe of modules
    bmap = None # BFieldMap object
    volume_bounds = {} # volume bounds: [r_min, r_max, z_min, z_max]
    t_bounds = {} # z if barrel, r if endcap
    u_bounds = {} # thickness bounds of layers: r if barrel and z if endcap
    detector_map = {} # dictionary with detector_map[volume][layer] being a pandas Dataframe module
    transformations_map = {} # dictionary with transformations_map[volume, layer, module] being (rotation_matrix, center) -> transform local to global with rotation_matrix @ uvw + center

    def __init__(self, detector_path, bmap):
        """
        Initializes Geometry object from detector file specifying all modules.
        ---
        detector_path   : string        : path to detector file from https://www.kaggle.com/competitions/trackml-particle-identification/data
        bmap            : BFieldMap     : B field object
        """
        self.detector = pd.read_csv(detector_path)
        self.bmap = bmap

        for vol in self.VOLUMES:
            vol_modules = self.detector.loc[self.detector.volume_id == vol]
            layer_ids = set(vol_modules.layer_id)
            self.detector_map[vol] = {}

            r_min = np.inf
            r_max = -np.inf
            z_min = np.inf
            z_max = -np.inf
            
            for lay in layer_ids:
                lay_modules = vol_modules.loc[vol_modules.layer_id == lay]
                self.detector_map[vol][lay] = lay_modules

                u_min = np.inf
                u_max = -np.inf
                t_min = np.inf
                t_max = -np.inf

                for i in range(lay_modules.shape[0]):
                    module = lay_modules.iloc[i]
                    rotation_matrix = np.array([
                        [module.rot_xu, module.rot_xv, module.rot_xw],
                        [module.rot_yu, module.rot_yv, module.rot_yw],
                        [module.rot_zu, module.rot_zv, module.rot_zw]
                    ])
                    center = np.array([module.cx, module.cy, module.cz])
                    
                    transform_xyz = lambda uvw: rotation_matrix @ uvw + center
                    self.transformations_map[(vol, lay, module.module_id)] = (rotation_matrix, center)
                    for pm_du, dv in [(module.module_maxhu, module.module_hv), (module.module_minhu, -module.module_hv)]:
                        for du in [pm_du, -pm_du]:
                            corner = transform_xyz(np.array([du, dv, 0]))
                            corner_r = np.sqrt(np.sum(corner[:2]**2))
                            corner_z = corner[2]
                            if corner_r > r_max:
                                r_max = corner_r
                            elif corner_r < r_min:
                                r_min = corner_r
                            if corner_z > z_max:
                                z_max = corner_z
                            elif corner_z < z_min:
                                z_min = corner_z 

                            corner_u = corner_r if vol in self.BARRELS else corner_z

                            if corner_u > u_max:
                                u_max = corner_u
                            elif corner_u < u_min:
                                u_min = corner_u
                
                if vol not in self.u_bounds:
                    self.u_bounds[vol] = {}
                self.u_bounds[vol][lay] = np.array([u_min, u_max])

            if vol in self.BARRELS:
                self.t_bounds[vol] = np.array([z_min, z_max])
            else:
                # Unfortunate hack
                self.t_bounds[vol] = np.array([r_min-2, r_max+2])

            r_min -= self.R_BUFFER
            r_max += self.R_BUFFER
            z_min -= self.Z_BUFFER
            z_max += self.Z_BUFFER
            self.volume_bounds[vol] = np.array([r_min, r_max, z_min, z_max])


    def nearest_layer(self, point, only_volume=False):
        """
        Find the nearest layer to a given space point
        ---
        point       : array(3)  : space point in cartesian coordinates
        only_volume : bool      : Whether or not to only return the volume of the space point. Saves time if this is all you want.
        ---
        volume      : int       : volume id for nearest layer
        layer       : int       : layer id for nearest layer
        distance    : float     : distance to nearest layer in mm
        """
        assert len(point) == 3
        point_r = np.sqrt(point[0]**2 + point[1]**2)
        point_z = point[2]

        volume = None
        for vol in self.volume_bounds:
            if self.volume_bounds[vol][0] <= point_r <= self.volume_bounds[vol][1] and self.volume_bounds[vol][2] <= point_z <= self.volume_bounds[vol][3]:
                volume = vol
                break

        if only_volume:
            return volume
        
        if volume is None:
            layer = None
            distance = None
        else:
            distance = np.inf
            point_s = point_r if vol in self.BARRELS else point_z
            
            for lay in self.u_bounds[vol]:
                lay_dist = abs(np.mean(self.u_bounds[vol][lay]) - point_s)
                if lay_dist < distance:
                    distance = lay_dist
                    layer = lay

        return volume, layer, distance


    def nearest_modules(self, volume, layer, point, nmodules=4):
        """
        Find the nmodules nearest modules
        ---
        volume          : int           : volume in which to search
        layer           : int           : layer in which to search
        point           : float(3)      : Global cartesian coordinates of the point
        nmodules        : int           : Number of modules to return. Default = 4 for the case where a point is on or near a 4-point module corner.
        ---
        closest_modules : pd.DataFrame  : dataframe of module information
        """
        lay_modules = self.detector_map[volume][layer]
        square_distances = (lay_modules.cx - point[0])**2 + (lay_modules.cy - point[1])**2 + (lay_modules.cz - point[2])**2
        indices = np.argpartition(square_distances, nmodules)
        closest_modules = lay_modules.iloc[indices[:nmodules]]
        return closest_modules


    def get_transformation(self, volume, layer, module_id):
        """
        Getter for transformation information for a specific module. Transform local to global coordinates with rotation_matrix @ uvw + center
        ---
        volume          : int           : volume of module
        layer           : int           : layer of module
        module_id       : int           : module id as specified in the detector file
        ---
        rotation_matrix : float(3, 3)   : rotates module local coordinates to be in line with global coordinates
        center          : float(3)      : shifts to global coordinates centered on the beamline
        """
        return self.transformations_map[volume, layer, module_id]
    

class BFieldMap:
    """
    B Field object. Currently only supports a uniform magnetic field in the z direction.
    In principle, rewriting the get method that returns the magnetic field direction at a point should work, but this is untested.
    """
    def get(self, pos):
        return np.array([0, 0, 2])
