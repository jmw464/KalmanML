import tensorflow as tf
import numpy as np
import utils
import helices
from mcts import Policy_nn
from scripts.policy.generate_multiplets import truth_seeds
import sys
import h5py

class PolicyTracker:
    HELIX_STEPSIZE = 5
    HELIX_RADIUS = 30
    MAX_DEPTH = 30

    geometry = None
    locator = None
    policy = None
    threshold = None

    def __init__(self, geometry, locator, policy, threshold):
        self.geometry = geometry
        self.locator = locator
        self.policy = policy
        self.threshold = threshold

    def track(self, seed, depth=0):
        track = list(seed)
        last_three_hits = np.array(track[-3:])[:,6:9]
        next_hits, _ = helices.get_next_hits(last_three_hits, self.HELIX_STEPSIZE, self.HELIX_RADIUS, self.locator, self.geometry)

        if next_hits is None or next_hits.shape[0] == 0 or depth > self.MAX_DEPTH:
            return [track]

        hit_probs = self.policy.apply(track, next_hits)
        good_hits = next_hits[np.array(hit_probs >= self.threshold)]
        if good_hits.shape[0] == 0:
            return [track]

        good_tracks = []
        for hit in good_hits:
            good_tracks += self.track(track + [hit], depth+1)
        return good_tracks

def main(argv):
    hits_path = "/global/cfs/cdirs/atlas/max_zhao/mlkf/trackml/hits.hdf5"
    detector_path = "/global/homes/m/max_zhao/mlkf/trackml/data/detectors.csv"
    outdir = "/global/cfs/cdirs/atlas/max_zhao/mlkf/tracks/policy_tracker/"
    model_path = "/global/homes/m/max_zhao/mlkf/trackml/models/policy_80_50/"
    max_hits = 25
    threshold = 0.5
    
    events = range(80, 100)
    policy = Policy_nn(model_path)

    tracks_file = h5py.File(outdir + "tracks{}.hdf5".format(int(threshold*100)), "w")

    for event_id in events:
        geometry = utils.Geometry(detector_path, utils.BFieldMap())
        locator = utils.HitLocator(5, detector_path)
        locator.load_hits(hits_path, event_id)
        seeds = truth_seeds(locator)

        tracker = PolicyTracker(geometry, locator, policy, threshold)

        event_tracks = []
        for seed in seeds:
            event_tracks += tracker.track(seed)

        event_tracks_arr = np.zeros((len(event_tracks), max_hits, 10))
        for i, track in enumerate(event_tracks):
            event_tracks_arr[i,:len(track)] = np.array(track)

        group = tracks_file.create_group(str(event_id))
        group.create_dataset("tracks", data=event_tracks_arr)
        print("Event: {} Seeds: {} Tracks: {}".format(event_id, len(seeds), len(event_tracks)))
    
    tracks_file.close()
    return 0

if __name__ == "__main__":
    print("\nFinished with exit code:", main(sys.argv))