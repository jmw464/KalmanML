import sys
import numpy as np
import utils
import time
import h5py
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
    
    geometry = utils.Geometry(detector_path, utils.BFieldMap())
    locator = utils.HitLocator(locator_resolution, geometry)
    locator.load_hits(hits_path, event_id, hit_type="t")

    seeds = truth_seeds(locator)

    triplets = []
    labels = []
    finding_stats = []

    prop_break = 0
    
    for seed in seeds:
        # Special treatmenet for seed
        if False:
            volume, layer, _  = geometry.nearest_layer(seed[-1,6:9])
            seed_phi = utils.arctan(seed[-1,7], seed[-1,6])
            seed_t = seed[-1,8] if volume in geometry.BARRELS else np.sqrt(seed[-1,7]**2 + seed[-1,6]**2)
            projection = (seed_phi, seed_t)

            new_hits = locator.get_hits_around(volume, layer, projection, radius)
            match_ind = new_hits[:,9] == seed[-1,9]

            for hit in new_hits:
                triplets.append(np.append(seed[:-1], np.array([hit]), 0))
            labels.append(np.int32(match_ind))
            finding_stats.append(np.append(seed[-1,6:9], np.array([1])))

        prev_triplet = seed
        track_incomplete = True
        while track_incomplete:
            new_hits, predicted_pos = helices.get_next_hits(prev_triplet[:,6:9], stepsize, radius, locator, geometry)
            # print(new_hits, predicted_pos)
            # assert False

            if new_hits is None or len(new_hits) == 0:
                # Terminate track if no new hits are found
                track_incomplete = False
            else:
                match_ind = new_hits[:,9] == prev_triplet[-1,9]
                #labels = np.concatenate((labels, np.int32(match_ind)))
                labels.append(np.int32(match_ind))
                for hit in new_hits:
                    triplets.append(np.append(prev_triplet, np.array([hit]), 0))

                matches = new_hits[match_ind]
                not_matches = new_hits[~match_ind]
               
                dist_to_predicted = lambda hits: np.sqrt(np.sum((hits[:,6:9] - predicted_pos)**2, axis=1))
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
                    if matches[0,6:9] in prev_triplet[:,6:9]:
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
    event_set = int(argv[1])
    print("Event set:", event_set)

    # hits_path = "/global/cfs/cdirs/atlas/max_zhao/mlkf/trackml/muons6000_100/processed/measurements.hdf5"
    hits_path = "/global/cfs/cdirs/atlas/max_zhao/mlkf/trackml/500ev_chi15/event_set{}/ttbar200_100/processed/measurements.hdf5".format(event_set)
    detector_path = "/global/homes/m/max_zhao/mlkf/trackml/data/detectors.csv"
    outdir = "/global/cfs/cdirs/atlas/max_zhao/mlkf/trackml/500ev_chi15/event_set{}/ttbar200_100/processed/".format(event_set)
    events = range(100)
    search_radii = [40]

    outfile = h5py.File(outdir + "quadruplets.hdf5", "w")

    for radius in search_radii:
        histograms = [[0 for _ in range(2*radius)], [0 for _ in range(2*radius)]]

        for event_id in events:
            groupname = str(radius) + "/" + str(event_id)
            group = outfile.create_group(groupname)
            
            start_time = time.time()
            event_triplets, event_labels, event_finding_stats = generate_triplets(hits_path, detector_path, event_id, radius, histograms)
            print("Finished event {} in:".format(event_id), time.time() - start_time)
            
            group.create_dataset("quadruplets", data=event_triplets)
            group.create_dataset("labels", data=event_labels)

        np.savetxt(outdir + "histograms.csv", np.array(histograms), delimiter=",")

    #finding_stats = np.array(finding_stats)
    #print("Total finding rate:", np.sum(finding_stats[:,-1]) / finding_stats.shape[0])

    #np.savetxt(outdir + "finding_stats_testing.csv", finding_stats, delimiter=",")
    #np.savetxt(outdir + "triplets.csv", triplets_flat, delimiter=",")
    #np.savetxt(outdir + "labels.csv", labels, delimiter=",")
    outfile.close()
    return 0
  
if __name__ == "__main__":
    print("\nFinished with exit code:", main(sys.argv))
