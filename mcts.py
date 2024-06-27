import torch as th
import tensorflow as tf

import os
import sys
import time
import numpy as np
import utils
import helices
import h5py
from train_evaluation.evaluation_model import *
from triplet_generator import truth_seeds

def wrong_way(intersection, track, geometry):
    wrong_way = False
    if intersection is not None:
        inter_vol, inter_lay, _ = geometry.nearest_layer(intersection)
        for elem in track:
            elem_vol = geometry.nearest_layer(elem[10:13], only_volume=True)
            if elem_vol == inter_vol and inter_lay == elem[0]:
                wrong_way = True
    return wrong_way

class Policy:
    def __init__(self):
        pass

    def __call__(self, track, candidates):
        return None

class Policy_nn(Policy):
    model = None

    def __init__(self, tfnn_path):
        self.model = tf.keras.models.load_model(tfnn_path)
        
    def __call__(self, track, candidates, intersection):
        track = np.array(track)[:,10:13]
        try:
            candidates = candidates[:,10:13]
        except:
            print(candidates)
            assert False

        previous_three = np.repeat(np.expand_dims(track[-3:], 0), candidates.shape[0], axis=0)
        expand_inter = np.repeat(np.expand_dims(intersection, axis=0), candidates.shape[0], axis=0)
        previous_three = np.reshape(previous_three, (candidates.shape[0], 9))
        inputs = np.concatenate([previous_three, candidates, expand_inter], axis=1)
        inputs = np.reshape(inputs, (inputs.shape[0], 15))
        inputs = tf.convert_to_tensor(inputs) / 1000

        outputs = self.model(inputs, training=False)
        return outputs[:,0]

class Edge:
    prev_node = None
    next_node = None

    scores = None
    prior = None

    def __init__(self, prior):
        self.prior = prior
        self.scores = []

    def get_action(self):
        return np.mean(self.scores) if self.scores else 0

    def get_sim(self, bonus_value=1):
        action = self.get_action()
        if action:
            return action + bonus_value * self.prior / (1 + len(self.scores))
        return self.prior

class Node:
    prev_edge = None
    next_edges = None
    track = None # list
    is_terminal = None

    def __init__(self, prev_edge, track):
        self.prev_edge = prev_edge
        self.track = track
        self.next_edges = []
        self.is_terminal = False

    def open_new(self, hit, prior):
        new_edge = Edge(prior)
        new_edge.prev_node = self
        new_edge.next_node = Node(new_edge, self.track.copy())
        new_edge.next_node.track.append(hit)

        self.next_edges.append(new_edge)
        return new_edge.next_node

class EvalOutput:
    def __init__(self, raw_output):
        self.raw_output = raw_output.detach().numpy()[:,0]

    def quality(self):
        return np.mean(self.raw_output)

    def update(self, edges):
        relevant = self.raw_output[3:3+len(edges)]
        for i, edge in enumerate(edges):
            edge.scores.append(relevant[i])

class Evaluation:
    MAX_LEN = 30

    def __init__(self, checkpoint_path, geometry):
        self.geometry = geometry
        device = th.device('cpu')
        self.model = EvalNN(7,128,2,1,False,0.1,30).double().to(device)

        checkpoint = th.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = checkpoint['epoch']+1
        # print("Loading previous model. Starting from epoch {}.".format(start_epoch), flush=True)

    def __call__(self, track):
        track = np.array(track)
        volumes = np.empty(track.shape[0])
        for i, hit in enumerate(track):
            volumes[i], _, _ = self.geometry.nearest_layer(hit[10:13])

        inputs = np.zeros((self.MAX_LEN, 7))
        inputs[:track.shape[0],0] = volumes
        inputs[:track.shape[0],1:] = track[:,:6]
        raw_output = self.model(th.from_numpy(inputs))
        
        return EvalOutput(raw_output[:track.shape[0]])

class Tree:
    root = None

    def __init__(self, seed):
        self.root = Node(None, list(seed))

    def traverse(self):
        curr_node = self.root
        visited_edges = []

        while curr_node.next_edges:
            max_edge = max(curr_node.next_edges, key=lambda x: x.get_sim())
            curr_node = max_edge.next_node
            visited_edges.append(max_edge)

        return curr_node, visited_edges

class SingleTracker:
    HELIX_STEPSIZE = 5
    SEARCH_RADIUS = 30

    def __init__(self, seed, geometry, locator, policy, evaluation, rnd=None):
        self.tree = Tree(seed)
        self.geometry = geometry
        self.locator = locator
        self.policy = policy
        self.evaluation = evaluation
        self.rnd = rnd if rnd is not None else np.random.default_rng(seed=42)

    def playout(self, stub):
        incomplete = True
        curr_track = stub.copy()
        while incomplete:
            try:
                next_hits, intersection = helices.get_next_hits(
                    np.array(curr_track)[-3:,10:13],
                    self.HELIX_STEPSIZE, 
                    self.SEARCH_RADIUS,
                    self.locator,
                    self.geometry
                )
            except ValueError:
                # print(stub)
                # print(curr_track)
                print(np.array(curr_track)[-3:,10:13])

            if next_hits is None or wrong_way(intersection, curr_track, self.geometry) or len(next_hits) == 0 or len(curr_track) >= 25:
                incomplete = False
            else:
                raw_probs = self.policy(curr_track.copy(), next_hits, intersection)

                # Hack
                stop_prob = 0.1 if len(curr_track) >= 7 else 0
                norm_probs = raw_probs * (1 - stop_prob) / np.sum(raw_probs)
                total_probs = np.append(norm_probs, stop_prob)
                assert np.abs(np.sum(total_probs) - 1) < 1e-4
                choice = self.rnd.choice(np.arange(len(next_hits) + 1), p=total_probs/np.sum(total_probs))
                if choice == len(next_hits):
                    incomplete = False
                else:
                    for elem in curr_track:
                        if np.all(elem == next_hits[choice]):
                            print("BRUH")
                            print(inter_vol, inter_lay)
                            print(self.geometry.nearest_layer(elem[10:13]))
                            assert False
                    # for elem in curr_track:
                    #     elem_vol, elem_lay, _ = self.geometry.nearest_layer(elem[10:13])
                    #     if elem_vol == inter_vol and inter_lay == elem_lay:
                    #         print("BRUH")
                    #         assert False
                    curr_track.append(next_hits[choice])

        if len(curr_track) > 25:
            print(len(curr_track))
            print(len(stub))
            assert False
        return curr_track

    def open_node(self, node):
        try:
            next_hits, intersection = helices.get_next_hits(
                np.array(node.track)[-3:,10:13],
                self.HELIX_STEPSIZE, 
                self.SEARCH_RADIUS,
                self.locator,
                self.geometry
            )
        except ValueError:
            print("open_node diff")
            assert False

        if next_hits is None or len(next_hits) == 0 or wrong_way(intersection, node.track, self.geometry):
            node.is_terminal = True
        else:
            priors = self.policy(node.track, next_hits, intersection)
            for i, hit in enumerate(next_hits):
                node.open_new(hit, priors[i])

    def step(self):
        leaf, visited_edges = self.tree.traverse()
        if not leaf.is_terminal:
            self.open_node(leaf)
            playout_track = self.playout(leaf.track)
        else:
            playout_track = leaf.track.copy()
        
        eval_output = self.evaluation(playout_track)
        eval_output.update(visited_edges)

        return playout_track, eval_output.quality()

class EventTracker:
    def __init__(self, geometry, locator, policy, evaluation, rnd=None):
        self.st_args = [geometry, locator, policy, evaluation]
        self.rnd = rnd if rnd is not None else np.random.default_rng(seed=42)

    def __call__(self, seeds, niterations, threshold, outpath=None):
        # debug_target = 1210
        # seeds = seeds[debug_target:debug_target+1]

        niterations = set(niterations)
        single_trackers = [SingleTracker(s, *self.st_args, self.rnd) for s in seeds]

        track_sets = []
        tracks = []
        for step_no in range(1, max(niterations)+1):
            sys.stdout.write("Step number: {}".format(step_no))
            sys.stdout.flush()

            for tracker_idx, tracker in enumerate(single_trackers):
                try:
                    playout_track, score = tracker.step()
                except:
                    print("Seed idx:",tracker_idx)
                    assert False, "Intentional debug error"
                if score >= threshold:
                    tracks.append(playout_track)

            if step_no in niterations:
                track_sets.append(tracks.copy())

                if outpath is not None:
                    outfile = h5py.File(outpath + "{}.hdf5".format(step_no), "w")
                    event_tracks = tracks.copy()
                    track_arr = np.empty((len(event_tracks),25,13))
                    for i, track in enumerate(event_tracks):
                        track_arr[i,:len(track),:] = np.array(track)
                    group = outfile.create_group("tracks")
                    group.create_dataset("tracks", data=track_arr)
                    outfile.close()

        assert len(track_sets) == len(niterations)
        return track_sets

def main(argv):
    hits_path = "/global/cfs/cdirs/atlas/max_zhao/mlkf/trackml/500ev_chi15/event_set0/ttbar200_100/processed/measurements.hdf5"
    detector_path = "/global/homes/m/max_zhao/mlkf/trackml/data/detectors.csv"
    policy_path = "/global/homes/m/max_zhao/mlkf/trackml/models/inter_policy/"
    evaluation_path = "/global/homes/m/max_zhao/mlkf/trackml/models/evaluation/11/transformer_11_model.pt"
    outdir = "/global/cfs/cdirs/atlas/max_zhao/mlkf/tracks/"

    runnumber = 0
    test_events = [80] #range(80, 100)
    niterations = range(1, 11)

    outdir = outdir + "run{}/".format(runnumber)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    print("Starting!")

    rnd = np.random.default_rng(seed=42)
    geometry = utils.Geometry(detector_path, utils.BFieldMap())

    policy = Policy_nn(policy_path)
    evaluation = Evaluation(evaluation_path, geometry)

    locator_resolution = 5
    for event in test_events:
        locator = utils.HitLocator(locator_resolution, geometry)
        locator.load_hits(hits_path, event, hit_type="t")

        seeds = truth_seeds(locator)
        print("{} seeds:".format(event), seeds.shape)

        event_tracker = EventTracker(geometry, locator, policy, evaluation, rnd)

        start_time = time.time()
        tracking_result = event_tracker(seeds, niterations, 0.7, outpath=outdir+"iter")

        print("Time per iteration:", (time.time() - start_time) / (niterations[-1]*seeds.shape[0]))
        print("Iterations:", niterations[-1]*seeds.shape[0])

        # outfile = h5py.File(outdir + "event{}".format(event), "w")
        # for iter_idx, niter in enumerate(niterations):
        #     event_tracks = tracking_result[iter_idx]
        #     track_arr = np.empty((len(event_tracks),25,13))
        #     for i, track in enumerate(event_tracks):
        #         track_arr[i,:len(track),:] = np.array(track)
        #     group = outfile.create_group(str(niter))
        #     group.create_dataset("tracks", data=tracks_arr)
        # outfile.close()

    return 0

if __name__ == "__main__":
    print("\nFinished with exit code:", main(sys.argv))