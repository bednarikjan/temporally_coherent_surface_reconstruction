# This script generates the point clouds from the Articulated Mesh Animation
# dataset [1]. It takes the original sequences and for each sample it
# uniformly randomly resamples 10k points from the original mesh. It saves
# the data for all the sequences in one common file. It saves the following:
#
# - sampled points
# - cummulative index ranges for the individual sequences
# - GT mesh vertices
# - boolean indices of valid GT indices
# - GT mesh areas
# - GT mesh faces
#
# NOTE: Not all the sequences have the same number of vertices, however, we
# need to store all the GT vertices in one 3d structure of shape (B, N, 3).
# Therefore, we store the same numbe of vertices for all the sequences, where
# some of them are just dummy values, and we indicate the valid indices in
# the `pts_gt_valid.npy` file.
#
#
# [1] D. Vlasic et al. Articulated mesh animation from multi-view silhouettes.
# TOG 2008.
#

# 3rd party.
import torch
import numpy as np
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
import matplotlib.pyplot as plt

# Python std.
from timeit import default_timer as timer

# Project files.
import externals.jblib.file_sys as jbfs
import externals.jblib.mesh as jbm

################################################################################
# Settings.
################################################################################

# Paths.
path_objs = ''  # <-- Path to the root directory with objs for each sequence.
path_ds_out = ''  # <-- Path to the root directory to save the dataset.
name_pts_out = 'pts_10k.npy'
name_seq2inds_out = 'pts_10k_seq2ind.npz'
name_gt_pts_out = 'pts_gt.npy'
name_gt_pts_num_valid_out = 'pts_gt_valid.npy'
name_gt_faces_out = 'faces_gt.npz'
name_areas_out = 'areas_gt.npy'

dev = torch.device('cuda')

num_verts_orig = 10002
num_pts = 10000

################################################################################
# Main script.
################################################################################

# Sample the pclouds.
s2i = {}
pts = []
pts_gt = []
faces_gt = {}
areas = []
proc_smpls = 0
seqs = sorted(jbfs.lsd(path_objs))
for seqi, seq in enumerate(seqs):
    path_seq = jbfs.jn(path_objs, seq, 'meshes')
    smpls = jbfs.ls(path_seq, exts='obj')
    s2i[seq] = (proc_smpls, proc_smpls + len(smpls))
    proc_smpls += len(smpls)
    for smpli, smpl in enumerate(smpls):
        print(f"\rProcessing seq {seqi + 1}/{len(seqs)}, "
              f"sample {smpli + 1}/{len(smpls)}.", end='')
        path_smpl = jbfs.jn(path_seq, smpl)
        verts, faces = load_obj(path_smpl, device=dev)[:2]
        faces = faces.verts_idx

        # Save faces.
        if smpli == 0:
            faces_gt[seq] = faces.cpu().numpy()

        m = Meshes(verts=[verts], faces=[faces])
        pts.append(sample_points_from_meshes(
            m, num_samples=num_pts)[0].cpu().numpy())
        pts_gt.append(verts.cpu().numpy())

        # Save area.
        areas.append(jbm.area(
            verts.cpu().numpy(), faces.cpu().numpy()))

pts_all = np.stack(pts, axis=0)
areas_all = np.array(areas)

# Get a single np.array of GT pts, some samples padded with zeros.
pts_gt_valid = np.array([x.shape[0] for x in pts_gt], dtype=np.int32)
max_valid = np.max(pts_gt_valid)
pts_gt_all = np.stack([np.concatenate(
    [x, np.zeros((max_valid - x.shape[0], 3),
                 dtype=np.float32)], axis=0) for x in pts_gt], axis=0)

# Save resulting pclouds, inds, areas.
assert pts_all.dtype == np.float32
assert pts_gt_all.dtype == np.float32
assert pts_gt_valid.dtype == np.int32
assert areas_all.dtype == np.float32

np.save(jbfs.jn(path_ds_out, name_pts_out), pts_all)
np.save(jbfs.jn(path_ds_out, name_gt_pts_out), pts_gt_all)
np.save(jbfs.jn(path_ds_out, name_gt_pts_num_valid_out), pts_gt_valid)
np.save(jbfs.jn(path_ds_out, name_areas_out), areas_all)
np.savez(jbfs.jn(path_ds_out, name_seq2inds_out), **s2i)
np.savez(jbfs.jn(path_ds_out, name_gt_faces_out), **faces_gt)
