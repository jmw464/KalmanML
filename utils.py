import numpy as np
import h5py
import pandas as pd

def arctan(y, x):
    """
    Computes the polar coordinate of a point (x, y). Ranges from [0, 2pi)
    ---
    x   : float
    y   : float
    ---
    phi : float : Polar coordinate
    """
    raw_phi = np.arctan2(y, x)
    phi = raw_phi if raw_phi >= 0 else 2*np.pi + raw_phi
    return phi

class Geometry:
    # Volume ids in the detector
    VOLUMES = np.array([7, 8, 9, 12, 13, 14, 16, 17, 18])
    # Volumes that are barrel shaped
    BARRELS = {8, 13, 17}
    # Buffer around layers with which to define volumes
    Z_BUFFER = 50
    R_BUFFER = 25

    # Pandas dataframe of modules
    detector = None
    # BFieldMap object
    bmap = None
    # Bounds of volumes: [r_min, r_max, z_min, z_max]
    volume_bounds = {}
    # z if barrel, r if endcap
    t_bounds = {}
    # Thickness bounds of layers: r if barrel and z if endcap
    u_bounds = {}
    # Dictionary with detector_map[volume][layer] being a pandas Dataframe module
    detector_map = {}
    # Dictionary with transformations_map[volume, layer, module] being (rotation_matrix, center)
    # Local cartesian coordinates can be transformed to global coordinates with rotation_matrix @ uvw + center
    transformations_map = {}

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

class HitLocator:
    # Naked data structure that stores the hits
    hit_map = {}
    # arrays of all hits on layer
    full_layers = {}
    # Geometry object for the detector
    geometry = None

    def __init__(self, resolution, geometry):
        """
        Initialize the data structure with empty cells using Geometry object.
        ---
        resolution      : float     : width of cells in mm
        geometry        : Geometry  : geometry object for the detector
        """
        assert type(geometry) == Geometry
        self.geometry = geometry
        #volume_id, layer_id, module_id, cx, cy, cz, hv = np.loadtxt(detector_path, delimiter=",", skiprows=1, usecols=[0,1,2,3,4,5,18], unpack=True)

        for volume in self.geometry.VOLUMES:
            vol_map = {}
            if volume in self.geometry.BARRELS:
                min_z, max_z = self.geometry.t_bounds[volume]
                for layer in self.geometry.u_bounds[volume]:              
                    diameter = 2*np.mean(self.geometry.u_bounds[volume][layer])

                    z_dim = round(np.ceil((max_z - min_z) / resolution))
                    phi_dim = round(np.ceil(np.pi * diameter / resolution))
                    vol_map[layer] = np.empty((phi_dim, z_dim), dtype=list)
            else:
                min_r, max_r = self.geometry.t_bounds[volume]

                r_dim = round(np.ceil((max_r - min_r) / resolution))
                phi_dim = round(np.ceil(np.pi * (max_r + min_r) / resolution))
                
                for layer in self.geometry.u_bounds[volume]:
                    vol_map[layer] = np.empty((phi_dim, r_dim), dtype=list)

            for layer in self.geometry.u_bounds[volume]:
                for row in vol_map[layer]:
                    for i in range(len(row)):
                        row[i] = []

            self.hit_map[volume] = vol_map

    def load_hits(self, hits_path, event_id, hit_type="m", layers_to_save=None):
        """
        Load hits into data structure from hits file
        ---
        hits_path       : String        : path to hits file
        event_id        : int           : event to store
        hit_type        : char          : Type of hit, "m" for measurements and "t" for truth hits
        layers_to_save  : set (tuples)  : layers that should be stored in full arrays
        """
        assert hit_type == "m" or hit_type == "t", "hit_type must be t or m"
        if layers_to_save is None:
            layers_to_save={(8,2), (8,4), (7,14), (9,2), (8,6), (7,12), (9,4)}
        volumes_to_save = {vol for vol, lay in layers_to_save}

        event_id = str(event_id)
        f = h5py.File(hits_path, "r")

        for volume_id in f[event_id].keys():
            volume = int(volume_id)
            vol_range = self.geometry.t_bounds[volume] #self.t_range[volume]
            vol_hits = f[event_id + "/" + volume_id + "/hits"]

            if volume in volumes_to_save:
                for layer in self.geometry.u_bounds[volume]:
                    if (volume, layer) in layers_to_save:
                        self.full_layers[(volume, layer)] = vol_hits[vol_hits[:,0] == layer]

            for hit in vol_hits:
                layer = hit[0]
                lay_map = self.hit_map[volume][layer]
                x, y, z = hit[10:13] if hit_type == "m" else hit[6:9]

                phi = arctan(y, x)
                phi_coord = round((phi / (2 * np.pi)) * (lay_map.shape[0] - 1))

                t = z if volume in self.geometry.BARRELS else np.sqrt(x**2 + y**2)
                
                assert vol_range[0] <= t <= vol_range[1], str(t) + " Vol: " + str(volume) + " " + str(vol_range)
                t_coord = round((lay_map.shape[1] - 1) * (t - vol_range[0]) / (vol_range[1] - vol_range[0]))

                lay_map[phi_coord, t_coord].append(hit)

        f.close()

    def get_layer_hits(self, volume, layer):
        """
        Get all hits on a layer
        ---
        volume  : int
        layer   : int
        ---
        hits    : array(d,n)
        """
        return self.full_layers[volume, layer]

    def get_near_hits(self, volume, layer, center, area):
        """
        Get all hits near some point on a layer using a range of coordinates.
        ---
        volume  : int               : volume that contains the layer
        layer   : int               : layer number
        center  : (float, float)    : point around which to collect hits. Of the form (phi, t) where t = z if barrel volume and r if endcap
        area    : (float, float)    : range with which to collect hits. Essentially collect hits with coordinate in center +- area
        ---
        hits    : float(d,n)        : array of hits
        """
        assert area[0] > 0 and area[1] > 0

        lay_map = self.hit_map[volume][layer]
        lay_range = self.geometry.t_bounds[volume]

        get_phi_coord = lambda phi: round((lay_map.shape[0] - 1) * phi / (2 * np.pi))
        get_t_coord = lambda t: round((lay_map.shape[1] - 1) * (t - lay_range[0]) / (lay_range[1] - lay_range[0]))

        start_phi = get_phi_coord(center[0] - area[0]) % lay_map.shape[0]
        end_phi = get_phi_coord(center[0] + area[0]) % lay_map.shape[0]
        start_t = max(get_t_coord(center[1] - area[1]), 0)
        end_t = min(get_t_coord(center[1] + area[1]), lay_map.shape[1] - 1)

        hits = []
        for t_coord in range(start_t, end_t + 1):
            phi_coord = start_phi
            while phi_coord != end_phi:
                hits += lay_map[phi_coord, t_coord]
                phi_coord = (phi_coord + 1) % lay_map.shape[0]

        return np.array(hits)

    def get_hits_around(self, volume, layer, center, radius):
        """
        Gets all hits around a point defined by a characteristic distance.
        ---
        volume  : int               : volume that contains the layer
        layer   : int               : layer number
        center  : (float, float)    : point around which to collect hits. Of the form (phi, t) where t = z if barrel volume and r if endcap
        radius  : float             : radius around which to collect hits.
        ---
        hits    : List              : list of hits
        """
        area = np.empty(2)
        if volume in Geometry.BARRELS:
            s = np.mean(self.geometry.u_bounds[volume][layer])
            area[0] = radius / s
        else:
            area[0] = radius / center[1]
        area[1] = radius

        return self.get_near_hits(volume, layer, center, area)

class BFieldMap:
    """
    B Field object. Currently only supports a uniform magnetic field in the z direction.
    In principle, rewriting the get method that returns the magnetic field direction at a point should work, but this is untested.
    """
    def get(self, pos):
        return np.array([0, 0, 2])
