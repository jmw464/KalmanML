#!/usr/bin/env python

import sys, h5py
from tqdm import tqdm
import numpy as np

import utils
import helices

def truth_seeds(locator, pcut=None):
    """
    Truth seeding in innermost layers
    ---
    locator	: utils.HitLocator	: HitLocator object that contains the hits
    pcut	: float			    : NOT IMPLEMENTED
    ---
    seeds	: array (n,3,10)	: truth seeds
    """
    layer_graph = {
        (8, 2) : [(8,4), (7,14), (9,2)],
        (7, 14) : [(7,12)],
        (8, 4) : [(8,6), (7,14), (9,2)],
        (9,2) : [(9,4)]
    }
    first_layer = locator.get_layer_hits(8, 2)
    seeds = []
    
    for first_hit in first_layer:
        particle_id = first_hit[9]
        seed = [first_hit]
        seed_layers = [(8, 2)]

        while len(seed) < 3:
            next_layer_tups = layer_graph[seed_layers[-1]]

            found = False
            for layer_tup in next_layer_tups:
                next_layer = locator.get_layer_hits(*layer_tup)
                matches = next_layer[next_layer[:,9] == particle_id]
                if matches.shape[0] > 0:
                    found = True
                    seed.append(matches[0])
                    seed_layers.append(layer_tup)
                    break

            if not found:
                break

        if len(seed) == 3:
            seeds.append(seed)

    return np.array(seeds)