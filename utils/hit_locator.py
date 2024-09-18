#!/usr/bin/env python

import h5py
import numpy as np

import utils.geometry
import utils.seeds


class HitLocator:
    hit_map = {} # data structure that stores the hits
    full_layers = {} # arrays of all hits on given layer
    geometry = None # detector geometry

    def __init__(self, resolution, geometry):
        """
        Initialize the data structure with empty cells using Geometry object.
        ---
        resolution      : float     : width of cells in mm
        geometry        : Geometry  : geometry object for the detector
        """
        assert type(geometry) == utils.geometry.Geometry
        self.geometry = geometry

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


    def load_hits(self, hits_file, event_id, hit_type="m", layers_to_save=None):
        """
        Load hits into data structure from hits file
        ---
        hits_file       : h5 file       : file containing hit information
        event_id        : int           : event to store
        hit_type        : char          : type of hit, "m" for measurements and "t" for truth hits
        layers_to_save  : set (tuples)  : layers that should be stored in full arrays
        """
        assert hit_type == "m" or hit_type == "t", "hit_type must be t or m"

        if layers_to_save is None:
            layers_to_save={(8,2), (8,4), (7,14), (9,2), (8,6), (7,12), (9,4)}
        volumes_to_save = {vol for vol, _ in layers_to_save}

        for volume_id in hits_file[str(event_id)].keys():
            volume = int(volume_id)
            vol_range = self.geometry.t_bounds[volume]
            vol_hits = hits_file[str(event_id) + "/" + volume_id + "/hits"]

            if volume in volumes_to_save:
                for layer in self.geometry.u_bounds[volume]:
                    if (volume, layer) in layers_to_save:
                        self.full_layers[(volume, layer)] = vol_hits[vol_hits["layer_id"][:] == layer]

            for hit in vol_hits:
                layer = hit['layer_id']
                lay_map = self.hit_map[volume][layer]

                # TODO: add way to retrieve global measurement x, y, z - always use truth for now
                if hit_type == "m" or hit_type == "t":
                    x, y, z = hit['true_x'], hit['true_y'], hit['true_z']

                phi = utils.math_utils.arctan(y, x)
                phi_coord = round((phi / (2 * np.pi)) * (lay_map.shape[0] - 1))

                t = z if volume in self.geometry.BARRELS else np.sqrt(x**2 + y**2)
                print(x, y, z)
                
                assert vol_range[0] <= t <= vol_range[1], str(t) + " Vol: " + str(volume) + " " + str(vol_range)
                t_coord = round((lay_map.shape[1] - 1) * (t - vol_range[0]) / (vol_range[1] - vol_range[0]))

                lay_map[phi_coord, t_coord].append(hit)


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
        if volume in utils.geometry.Geometry.BARRELS:
            s = np.mean(self.geometry.u_bounds[volume][layer])
            area[0] = radius / s
        else:
            area[0] = radius / center[1]
        area[1] = radius

        return self.get_near_hits(volume, layer, center, area)