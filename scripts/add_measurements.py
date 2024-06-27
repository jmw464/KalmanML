import utils
import sys
import h5py
import numpy as np
import time
import matplotlib.pyplot as plt

def main(argv):
    event_set = int(argv[1])
    indir = "/global/cfs/cdirs/atlas/max_zhao/mlkf/trackml/500ev_chi15/event_set{}/ttbar200_100/".format(event_set)
    detector_path = "/global/homes/m/max_zhao/mlkf/trackml/data/detectors.csv"

    hits_path = indir + "processed/hits.hdf5"
    measurements_path = indir + "processed/measurements.hdf5"
    geometry = utils.Geometry(detector_path, utils.BFieldMap())

    hits_file = h5py.File(hits_path, "r")
    meas_file = h5py.File(measurements_path, "w")

    differences = {vol:[] for vol in utils.Geometry.VOLUMES}
    overlaps = 0

    for event_id in hits_file.keys():
        event_id = str(event_id)

        start_time = time.time()
        for vol_id in hits_file[event_id].keys():

            event = int(event_id)
            vol = int(vol_id)
            
            groupname = event_id + "/" + vol_id
            vol_hits = hits_file[groupname + "/hits"]

            new_vol_hits = np.zeros((vol_hits.shape[0], vol_hits.shape[1] + 3))
            new_vol_hits[:,:10] = vol_hits

            for ihit, hit in enumerate(new_vol_hits):
                # transform_xyz = geometry.get_transformation(vol, hit[0], hit[1])

                # module = int(geometry.nearest_modules(vol, hit[0], hit[6:9], 1).module_id)
                possible_modules = geometry.nearest_modules(vol, hit[0], hit[6:9])
                consistent_meas = []

                for i in range(possible_modules.shape[0]):
                    module = possible_modules.iloc[i]

                    rotation_matrix, center = geometry.get_transformation(vol, hit[0], module.module_id)
                    true_local = np.transpose(rotation_matrix) @ (hit[6:9] - center)

                    if utils.check_module_boundary(true_local[0], true_local[1], module):
                        meas_local = np.array([hit[2], hit[3], 0])
                        consistent_meas.append(rotation_matrix @ meas_local + center)

                if not consistent_meas:
                    # Cheat and use the truth hit instead
                    print(hit[6:9])
                    new_vol_hits[ihit,10:] = hit[6:9]
                else:
                    if len(consistent_meas) > 1:
                        overlaps += 1
                    
                    consistent_diff = [np.sqrt(np.sum((meas - hit[6:9])**2)) for meas in consistent_meas]
                    differences[vol].append(min(consistent_diff))
                    new_vol_hits[ihit,10:] = consistent_meas[np.argmin(consistent_diff)]

                #differences.append(np.sqrt(np.sum((meas_global - hit[6:9])**2)))

                # differences[vol].append(np.sqrt(np.sum((meas_global - hit[6:9])**2)))
                # new_vol_hits[i,10:] = meas_global

            group = meas_file.create_group(groupname)
            group.create_dataset("hits", data=new_vol_hits)

        print("Event {} time: {:.3f}".format(event_id, time.time() - start_time))
        start_time = time.time()

    print("Total overlaps:"s, overlaps)

    # for vol in differences:
    #     plt.figure(figsize=(8,6))
    #     plt.yscale("log")
    #     plt.hist(differences[vol], bins=40)
    #     plt.xlabel("Distance (mm)")
    #     plt.savefig("/global/homes/m/max_zhao/bin/new_meas_diff{}.png".format(vol))

    hits_file.close()
    meas_file.close()
    return 0
    
if __name__ == "__main__":
    print("\nFinished with exit code:", main(sys.argv))