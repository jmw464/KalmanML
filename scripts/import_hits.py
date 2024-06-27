#!/usr/bin/env python

import os,sys,math,glob,ROOT,h5py
import numpy as np
import awkward as ak
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas, TTree, gDirectory, TChain, TH2D
import uproot
import time

def main(argv):
    gROOT.SetBatch(True)
    
    # BEGIN INPUT

    nvar = 12 #number of stored variables for each hit (includes event, volume, layer and module number)

    indir = "/global/cfs/cdirs/atlas/max_zhao/mlkf/trackml/test_events/eval_test/ttbar200_5/"
    outdir = indir+"processed/"

    # END INPUT

    if not os.path.exists(outdir): os.mkdir(outdir)

    hits_file = uproot.open(indir+"hits.root")
    hits_tree = hits_file["hits"]
    measurements_file = uproot.open(indir+"measurements.root")

    outfile = h5py.File(outdir+"hits.hdf5","w")

    data = np.zeros((0,nvar))
    keylist = measurements_file.keys()
    
    for i, key in enumerate(keylist):
        keylist[i] = key.split(";")[0]
    keylist = list(set(keylist))
    keylist.sort()

    start_time = time.time()
    print("Beginning data processing")

    #read each folder in measurements file (corresponding to volume, layer and modules number)
    measurement_entries = 0
    for key in keylist:
        cur_time = time.time()
        vol = key.split("_")[0]
        layer_tree = measurements_file[key]

        measurement_entries += layer_tree.num_entries
        layer_data = np.zeros((layer_tree.num_entries,nvar))
        
        layer_data[:,0] = layer_tree["event_nr"].array(library="np")
        layer_data[:,1] = layer_tree["volume_id"].array(library="np")
        layer_data[:,2] = layer_tree["layer_id"].array(library="np")
        layer_data[:,3] = layer_tree["surface_id"].array(library="np")
        layer_data[:,4] = layer_tree["rec_loc0"].array(library="np")
        layer_data[:,5] = layer_tree["rec_loc1"].array(library="np")
        layer_data[:,6] = ak.sum(layer_tree["channel_value"].array(library="ak"),axis=1)
        layer_data[:,7] = layer_tree["clus_size"].array(library="np")
        layer_data[:,8] = layer_tree["true_x"].array(library="np")
        layer_data[:,9] = layer_tree["true_y"].array(library="np")
        layer_data[:,10] = layer_tree["true_z"].array(library="np")

        data = np.append(data,layer_data,axis=0)

        print("Processed key {} with {} hits in {} seconds".format(key, layer_tree.num_entries, time.time()-cur_time))

    #sort data for each volume based on event, volume, layer and module number (to match truth particles)
    data = data[data[:,3].argsort(kind='mergesort')] #sort by module
    data = data[data[:,2].argsort(kind='mergesort')] #sort by layer
    data = data[data[:,1].argsort(kind='mergesort')] #sort by volume
    data = data[data[:,0].argsort(kind='mergesort')] #sort by event

    print("Adding truth particle info. Time elapsed: {} seconds".format(time.time()-start_time))
    
    #read truth particle information from hits file
    truth_data = np.zeros((hits_tree.num_entries,8))
    truth_data[:,0] = hits_tree["event_id"].array(library="np")
    truth_data[:,1] = hits_tree["tx"].array(library="np")
    truth_data[:,2] = hits_tree["ty"].array(library="np")
    truth_data[:,3] = hits_tree["tz"].array(library="np")
    truth_data[:,4] = hits_tree["volume_id"].array(library="np")
    truth_data[:,5] = hits_tree["layer_id"].array(library="np")
    truth_data[:,6] = hits_tree["sensitive_id"].array(library="np")
    truth_data[:,7] = hits_tree["particle_id"].array(library="np")
    truth_data = truth_data[truth_data[:,0].argsort(kind="mergesort")] #sort by event

    start_time = time.time()
    curr_event = -1
    for i in range(data.shape[0]):
        if i % 10000 == 0:
            duration = time.time() - start_time
            eta = (1/3600) * duration * data.shape[0] / (i + 1)
            sys.stdout.write("\rElapsed: {} Projected hours: {}".format(duration, eta))
            sys.stdout.flush()
        if curr_event != data[i,0]:
            curr_event = data[i,0]
            event_truth = truth_data[data[i,0] == truth_data[:,0]]
        match_index = np.argwhere(np.logical_and.reduce(
            [data[i,8] == event_truth[:,1],
            data[i,9] == event_truth[:,2]]
        ))[0]
        data[i,-1] = truth_data[match_index,-1]

    print("Writing file. Time elapsed: {} seconds".format(time.time()-start_time))

    #write hdf5 file
    cur_time = time.time()
    for i in range(data.shape[0]):
        event_id = int(data[i,0])
        volume_id = int(data[i,1])

        groupname = str(event_id)+"/"+str(volume_id) #group structure is event/volume/layer/module

        if groupname not in outfile.keys():
            group = outfile.create_group(groupname)
            data_slice = data[np.logical_and(data[:,0]==event_id,data[:,1]==volume_id)] #slice data corresponding to event, volume, layer and module
            group.create_dataset("hits",data=data_slice[:,2:]) #remove event, volume and layer information from array
            print("Writing {} in {} seconds".format(groupname, time.time()-cur_time))
            cur_time = time.time()

    print("Finished. Time elapsed: {} seconds".format(time.time()-start_time))

    outfile.close()


if __name__ == '__main__':
    main(sys.argv)
