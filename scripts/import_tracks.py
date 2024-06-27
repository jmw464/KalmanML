#!/usr/bin/env python

import os,sys,math,glob,ROOT,h5py,time
import numpy as np
# import matplotlib.pyplot as plt
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas, TTree, gDirectory, TChain, TH2D


def plot_candidates(track_candidates, projection, max_candidates):

    def projection_index(coordinate):
        if coordinate == "x":
            return 1
        elif coordinate == "y":
            return 2
        elif coordinate =="z":
            return 3
        else:
            raise Exception("Invalid projection. Please use a permutation of xy, xz or yz.")

    index1, index2 = (projection_index(projection[0]), projection_index(projection[1]))
    max_candidates = min(max_candidates, track_candidates.shape[0])
    plt.figure()
    for i in range(max_candidates):
        track_candidate = track_candidates[i,:][track_candidates[i,:,0] > 0]
        plt.plot(track_candidate[:,index1],track_candidate[:,index2],alpha=0.5)
    plt.xlabel(projection[0])
    plt.ylabel(projection[1])
    plt.savefig("tracks.pdf")

def plot_match_distribution(track_truth, path):
    match_counts = np.empty(track_truth.shape[0])
    total_counts = np.empty(track_truth.shape[0])
    for i in range(track_truth.shape[0]):
        assert np.all(np.equal(track_truth[i,:,0], (track_truth[i,:,1] == track_truth[i,0,1]).astype(float)))
        nonpad = track_truth[i,:,0][track_truth[i,:,1] > 1]
        match_counts[i] = np.sum(nonpad) # np.sum(nonpad) / nonpad.shape[0]
        total_counts[i] = nonpad.shape[0]

    plt.figure(figsize=(8,6))
    plt.hist(match_counts, bins=20, color="blue", label="nmatch", histtype="step")
    plt.hist(total_counts, bins=20, color="orange", label="ntotal", histtype="step")
    plt.legend()
    plt.savefig(path)
    plt.close() 


def main(argv):
    gROOT.SetBatch(True)

    # BEGIN INPUT

    maxhits = 30
    minhits = 3

    indir = "/global/cfs/cdirs/atlas/max_zhao/mlkf/trackml/test_events/eval_test/ttbar200_5/"
    outdir = indir+"processed/"

    # END INPUT

    tracks_file = TFile(indir+"trackstates_ckf.root")
    tracks_tree = tracks_file.Get("trackstates")

    hits_file = h5py.File(outdir+"hits.hdf5", "r")
    outfile = h5py.File(outdir+"tracks.hdf5","w")
 
    nentries = tracks_tree.GetEntries()
    track_meas = np.zeros((nentries,maxhits,7))
    track_truth = np.zeros((nentries,maxhits,5))

    percents_match = []
    bad_tracks = [] #index of tracks to remove from final sample

    start_time = time.time()

    for ientry, track_entry in enumerate(tracks_tree):
        nhits = track_entry.nMeasurements
        event_id = track_entry.event_nr
        if nhits > minhits:
            for i in range(nhits):
                volume_id = track_entry.volume_id[i]
                layer_id = track_entry.layer_id[i]
                module_id = track_entry.module_id[i]

                rel_hits = hits_file[str(event_id)+"/"+str(volume_id)]["hits"]
                true_x = track_entry.t_x[i]
                true_y = track_entry.t_y[i]
                true_z = track_entry.t_z[i]

                hitmatch = np.logical_and.reduce((rel_hits[:,0] == layer_id, rel_hits[:,1] == module_id, np.abs(rel_hits[:,6] - true_x) < 10e-4, np.abs(rel_hits[:,7] - true_y) < 10e-4, np.abs(rel_hits[:,8] - true_z) < 10e-4))

                #check if hit match was found, if not add to bad tracks
                if np.where(hitmatch)[0].size == 0:
                    bad_tracks.append(ientry)
                    break
                else:
                    hitindex = np.where(hitmatch)[0][0]
                    
                    try:
                        track_truth[ientry,nhits-i-1] = [0, rel_hits[hitindex,-1], true_x, true_y, true_z]
                        track_meas[ientry,nhits-i-1] = [volume_id, layer_id, module_id, rel_hits[hitindex,2], rel_hits[hitindex,3], rel_hits[hitindex,4], rel_hits[hitindex,5]]
                    except IndexError:
                        print("IndexError:", nhits-i-1)

            if ientry not in bad_tracks:
                track_truth[ientry,:,0] = np.logical_and(track_truth[ientry,:,1] == track_truth[ientry,0,1], track_truth[ientry,:,1] > 0) #calculate NN labels by checking which hits are from same particle as seed
                track_candidate = track_truth[ientry,:,1][track_truth[ientry,:,1] > 0] #isolate track candidate (useful for plotting)
                try:
                    percent_from_seed_particle = track_candidate[track_candidate == track_candidate[0]].shape[0]/track_candidate.shape[0] #calculate percentage of hits in track that are from same particle as seed
                    percents_match.append(percent_from_seed_particle)
                except IndexError:
                    print("Empty track")
                    bad_tracks.append(ientry)

            # if ientry % 1000 == 0:
            #     plt.figure(figsize=(8,6))
            #     plt.hist(percents_match, bins=40, label="{:.3f}%".format(np.mean(percents_match)*100))
            #     plt.title("CKF match percentage with first hit")
            #     plt.legend()
            #     plt.savefig(outdir + "ckf_match_rate.png")
            #     plt.close()

            if ientry % 1000 == 0:
                duration = time.time() - start_time
                eta = (1/3600) * duration * nentries / (ientry + 1)
                sys.stdout.write("\rElapsed: {} Projected hours: {}".format(duration, eta))
                sys.stdout.flush()

    #remove bad tracks
    track_meas = np.delete(track_meas, bad_tracks, axis=0)
    track_truth = np.delete(track_truth, bad_tracks, axis=0)
    print("\nRemoved {} bad tracks".format(len(bad_tracks)))

    track_meas = track_meas[track_truth[:,0,1] > 0]
    track_truth = track_truth[track_truth[:,0,1] > 0]
    # plot_match_distribution(track_truth, "./pre_match_dist.png")
    track_truth, unique_indices = np.unique(track_truth,return_index=True,axis=0)   
    # plot_match_distribution(track_truth, "./unique_match_dist.png")
    track_meas = track_meas[unique_indices]

    shuffle_array = np.random.permutation(track_truth.shape[0])
    track_truth = track_truth[shuffle_array]
    track_meas = track_meas[shuffle_array]

    #calculate what percent tracks are from unique seeds
    percent_from_unique_seed = np.unique(track_truth[:,0,1]).shape[0]/track_truth.shape[0]

    # plot_candidates(track_truth,"xy",10000)

    group = outfile.create_group("tracks")
    group.create_dataset("measurements",data=track_meas)
    group.create_dataset("truth",data=track_truth)

    hits_file.close()
    outfile.close()


if __name__ == '__main__':
    main(sys.argv)
