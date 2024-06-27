from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas, TTree, gDirectory, TChain, TH2D
import sys
import h5py
import numpy as np

def main(argv):
    gROOT.SetBatch(True)

    # BEGIN INPUT

    indir = "/global/cfs/cdirs/atlas/max_zhao/mlkf/trackml/500ev_chi15/event_set0/ttbar200_100/"
    inpath = indir + "particles_initial.root"
    outpath = indir + "processed/"

    # END INPUT
    
    infile = TFile(inpath)
    part_tree = infile.Get("particles")
    outfile = h5py.File(outpath + "particles.hdf5", "w")

    for event_id, event_particles in enumerate(part_tree):
        data = np.empty((len(event_particles.particle_id), 4))
        data[:,0] = event_particles.particle_id
        data[:,1] = event_particles.eta
        data[:,2] = event_particles.phi
        data[:,3] = event_particles.pt
        
        group = outfile.create_group(str(event_id))
        group.create_dataset("particles", data=data)

    outfile.close()

    return 0

if __name__ == "__main__":
    print("\nFinished with exit code:", main(sys.argv))