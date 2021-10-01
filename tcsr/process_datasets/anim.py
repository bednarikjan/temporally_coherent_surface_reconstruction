# This script generates the pointclouds from various datasets containing
# animals in motion. It takes the original sequences and for each sample it
# uniformly randomly resamples 10k points from the original mesh. The data can
# be optionally centered and scaled. It saves the data for every sequence in
# its own directory. It saves the following:
# - sampled points
# - GT mesh vertices
# - GT mesh areas
# - GT mesh faces
#
# Supported datasets:
# [1] G. Aujay et al. Harmonic Skeleton for Realistic Character Animation.
# SGIGGRAPH 2007.
# [2] R. Sumner. Deformation Transfer for Triangle Meshes. SIGGRAPH 2004
#

# 3rd party.
import torch
import numpy as np
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj

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
name_gt_pts_out = 'pts_gt.npy'
name_gt_faces_out = 'faces_gt.npy'
name_areas_out = 'areas_gt.npy'

# Data selection.
sequences_selection = []  # All sequences if empty.

# Sampling.
num_pts = 10000

# Transformation.
center = True
scale = True
scale_max_size = 1.

dev = torch.device('cuda')

################################################################################
# Main script.
################################################################################

# Get sequences.
sequences = sequences_selection if len(sequences_selection) > 0 \
    else jbfs.lsd(path_objs)
assert all([s in jbfs.lsd(path_objs) for s in sequences])

# Process all sequences.
for si, s in enumerate(sequences):
    # Sample the pclouds.
    pts = []
    pts_gt = []
    faces_gt = None
    areas = []
    path_seq = jbfs.jn(path_objs, s)
    smpls = jbfs.ls(path_seq, exts='obj')
    centroid = None
    max_size = None
    for smpli, smpl in enumerate(smpls):
        print(f"\rProcessing seq {si + 1}/{len(sequences)}, "
              f"sample {smpli + 1}/{len(smpls)}.", end='')
        path_smpl = jbfs.jn(path_seq, smpl)
        verts, faces = load_obj(path_smpl, device=dev)[:2]
        faces = faces.verts_idx

        # Save faces.
        if smpli == 0:
            faces_gt = faces.cpu().numpy()

        # Sample the mesh.
        m = Meshes(verts=[verts], faces=[faces])
        pts_smpl = sample_points_from_meshes(
            m, num_samples=num_pts)[0].cpu().numpy(). \
            astype(np.float32)
        pts_gt_smpl = verts.cpu().numpy().astype(np.float32)

        # Get centroid and scale of the first sample in seq.
        if smpli == 0:
            if center or scale:
                centroid = np.mean(pts_gt_smpl, axis=0)
            if scale:
                max_size = np.max(
                    np.max(pts_gt_smpl, axis=0) -
                    np.min(pts_gt_smpl, axis=0))

        # Center and scale.
        if center or scale:
            pts_smpl = pts_smpl - centroid
            pts_gt_smpl = pts_gt_smpl - centroid
        if scale:
            pts_smpl = pts_smpl / max_size * scale_max_size
            pts_gt_smpl = pts_gt_smpl / max_size * scale_max_size
            if not center:
                pts_smpl = pts_smpl + centroid
                pts_gt_smpl = pts_gt_smpl + centroid

        # Store pts.
        pts.append(pts_smpl)
        pts_gt.append(pts_gt_smpl)

        # Save area.
        areas.append(jbm.area(
            pts_gt_smpl, faces.cpu().numpy()))

    # Assemble all data samples.
    pts_all = np.stack(pts, axis=0)
    pts_gt_all = np.stack(pts_gt, axis=0)
    areas_all = np.array(areas)

    # Save resulting pclouds, inds, areas.
    assert pts_all.dtype == np.float32
    assert pts_gt_all.dtype == np.float32
    assert areas_all.dtype == np.float32
    path_out = jbfs.jn(path_ds_out, s)
    jbfs.make_dir(path_out)
    for d, n in zip([pts_all, pts_gt_all, areas_all, faces_gt],
                    [name_pts_out, name_gt_pts_out, name_areas_out,
                     name_gt_faces_out]):
        np.save(jbfs.jn(path_out, n), d)
