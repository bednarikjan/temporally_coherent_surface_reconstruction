# This script generates the point clouds from the DFAUST dataset [1]. It takes
# the original sequences and for each sample it uniformly randomly resamples
# 10k points from the original mesh. Note that the clean registered meshes are
# used here. the script saves the data for all the subjects and sequences in
# one common file. It saves the following:
#
# - sampled points
# - cummulative index ranges for the individual sequences
# - GT mesh areas
#
# [1] F. Bogo et al. Dynamic FAUST: Registering Human Bodies in Motion. CVPR'17.
#

# 3rd party.
import h5py
import torch
import numpy as np
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

# Project files.
import externals.jblib.file_sys as jbfs
import externals.jblib.mesh as jbm

################################################################################
# Settings.
################################################################################

path_meshes = ''  # <-- Path to the root directory with the *.hdf5 files.
path_ds_out = ''  # <-- Path to the root directory to save the dataset.
name_pts_out = '/cvlabdata2/cvlab/datasets_jan/dfaust/registrations_all_10k.npy'
name_seq2inds_out = '/cvlabdata2/cvlab/datasets_jan/dfaust/registrations_all_10k_seq2ind.npz'
name_areas_out = 'registrations_areas_all.npy'

dev = torch.device('cuda')

num_pts = 10000

################################################################################
# Main script.
################################################################################

path_reg_m = jbfs.jn(path_meshes, 'registrations_m.hdf5')
path_reg_f = jbfs.jn(path_meshes, 'registrations_f.hdf5')

# Load data.
file_regs_m = h5py.File(path_reg_m, 'r')
file_regs_f = h5py.File(path_reg_f, 'r')

seq_ids = \
    [k for k in file_regs_m.keys() if k != 'faces'] + \
    [k for k in file_regs_f.keys() if k != 'faces']
seq_ids = np.sort(seq_ids)

faces_m = np.asarray(file_regs_m['faces']).astype(np.int32)
faces_f = np.asarray(file_regs_f['faces']).astype(np.int32)

# Sample the pclouds.
proc_smpls = 0
s2i = {}
pts = []
areas = []
for seq_i, s in enumerate(seq_ids):
    verts = np.asarray(
        (file_regs_m, file_regs_f)[s in file_regs_f][s]). \
        astype(np.float32).transpose((2, 0, 1))
    verts_t = torch.from_numpy(verts).to(dev)
    faces = (faces_m, faces_f)[s in file_regs_f]
    faces_t = torch.from_numpy(faces).to(dev)

    s2i[s] = (proc_smpls, proc_smpls + verts.shape[0])
    proc_smpls += verts.shape[0]

    for smpl_i, smpl in enumerate(verts_t):
        print(f"\rProcessing sequence {seq_i + 1}/{len(seq_ids)}, "
              f"sample {smpl_i + 1}/{verts.shape[0]}", end='')

        m = Meshes(verts=[smpl], faces=[faces_t])
        pts.append(sample_points_from_meshes(
            m, num_samples=num_pts)[0].cpu().numpy())
        areas.append(jbm.area(verts[smpl_i], faces))
pts_all = np.stack(pts, axis=0)

# Save resulting pclouds and inds.
np.save(jbfs.jn(path_ds_out, name_pts_out), pts_all)
np.savez(jbfs.jn(path_ds_out, name_seq2inds_out), **s2i)
np.save(jbfs.jn(path_ds_out, name_areas_out), areas)
