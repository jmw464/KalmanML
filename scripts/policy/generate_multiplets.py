#!/usr/bin/env python

import os, sys, h5py, argparse
from tqdm import tqdm
import numpy as np

from utils import helix
from utils import geometry
from utils import seeds
from utils import hit_locator


def generate_triplets(hits_path, detector_path, event_id, radius, histograms):
    """
    Generates training triplets for policy network
    ---
    hits_path		: string	    	: path to hits file
    detector_path	: string		    : path to detector file
    event_id		: int			    : event to generate triplets from
    radius          : int               : radius around which to search for hits around helix intersections
    ---
    triplets		: array (n,3,10)	: training triplets
    labels		    : array (n)		    : training labels
    finding_stats	: array (m,4)		: first three values are the positions of helix intersections. Last is binary if it successfully found next hit
    ---
    """
    locator_resolution = 5
    stepsize = 5
    
    det_geometry = geometry.Geometry(detector_path, geometry.BFieldMap())
    locator = hit_locator.HitLocator(locator_resolution, det_geometry)
    locator.load_hits(hits_path, event_id, hit_type="t")

    seeds = seeds.truth_seeds(locator)

    triplets = []
    labels = []
    finding_stats = []

    prop_break = 0
    
    for seed in seeds:
        prev_triplet = seed
        track_incomplete = True
        while track_incomplete:
            hit_positions = np.stack((prev_triplet['true_x'], prev_triplet['true_y'], prev_triplet['true_z']), axis=-1)
            new_hits, predicted_pos = helix.get_next_hits(hit_positions, stepsize, radius, locator, det_geometry)

            if new_hits is None or len(new_hits) == 0:
                # Terminate track if no new hits are found
                track_incomplete = False
            else:
                match_ind = new_hits["particle_id"] == prev_triplet["particle_id"][-1]
                labels.append(np.int32(match_ind))
                for hit in new_hits:
                    triplets.append(np.append(prev_triplet, np.array([hit]), 0))

                matches = new_hits[match_ind]
                not_matches = new_hits[~match_ind]
               
                dist_to_predicted = lambda hits: np.sqrt(np.sum((np.stack((hits["true_x"], hits["true_y"], hits["true_z"]), axis=-1) - predicted_pos)**2, axis=1))
                match_dists = dist_to_predicted(matches)
                not_match_dists = dist_to_predicted(not_matches)
                for dist in match_dists:
                    histograms[0][np.int32(dist)] += 1
                for dist in not_match_dists:
                    try:
                        histograms[1][np.int32(dist)] += 1
                    except:
                        print(dist)
                        assert False
                
                if matches.shape[0] == 0:
                    track_incomplete = False
                else:
                    match_coords = np.stack((matches["true_x"], matches["true_y"], matches["true_z"]), axis=-1)
                    prev_triplet_coords = np.stack((prev_triplet["true_x"], prev_triplet["true_y"], prev_triplet["true_z"]), axis=-1)
                    if match_coords[0] in prev_triplet_coords:
                        prop_break += 1
                        break
                    prev_triplet = np.append(prev_triplet[1:], [matches[0]], axis=0)
            
            if predicted_pos is not None:
                finding_stats.append(np.append(predicted_pos, np.array([np.int32(track_incomplete)])))
    
    finding_stats = np.array(finding_stats)
    try:
        print("Event: " + str(event_id) + " Finding rate: " + str(np.sum(finding_stats[:,-1]) / finding_stats.shape[0]))
    except IndexError:
        print("Event: ", event_id)
        print(triplets)
        print(labels)
        assert False
    labels = np.concatenate(labels)
    assert len(labels) == len(triplets)

    return triplets, labels, finding_stats


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, help = 'Path containing pre-processed data (with hits.hdf5 file)')
    parser.add_argument('--out_dir', type=str, help = 'Path to store generated multiplets')
    parser.add_argument('-r', '--radius', type=int, default=40, help = 'Radius around which to search for hits around helix intersections (in mm)')
    args = parser.parse_args()

    hits_file = h5py.File(args.in_dir+"hits.hdf5", "r")
    detector_path = args.in_dir+"detectors.csv"
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    out_file = h5py.File(args.out_dir + "multiplets.hdf5", "w")

    nevents = max(list(map(int, hits_file.keys())))

    histograms = [[0 for _ in range(2*args.radius)], [0 for _ in range(2*args.radius)]]

    for event_id in range(nevents):
        groupname = str(args.radius) + "/" + str(event_id)
        group = out_file.create_group(groupname)
        
        event_triplets, event_labels, event_finding_stats = generate_triplets(hits_file, detector_path, event_id, args.radius, histograms)
        
        group.create_dataset("quadruplets", data=event_triplets)
        group.create_dataset("labels", data=event_labels)

    np.savetxt(args.outdir + "histograms.csv", np.array(histograms), delimiter=",")

    #finding_stats = np.array(finding_stats)
    #print("Total finding rate:", np.sum(finding_stats[:,-1]) / finding_stats.shape[0])

    #np.savetxt(outdir + "finding_stats_testing.csv", finding_stats, delimiter=",")
    #np.savetxt(outdir + "triplets.csv", triplets_flat, delimiter=",")
    #np.savetxt(outdir + "labels.csv", labels, delimiter=",")
    out_file.close()
  
  
if __name__ == '__main__':
    main(sys.argv)