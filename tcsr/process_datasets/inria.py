# Generates the point cloud dataset form the INRIA Clothed Human [1] dataset
# (raw human scans) by sampling the 10k points uniformly from the scanned
# surfaces represented as meshes. Besides point clouds, the sequence-to-indices
# file is generated. It saves the data for all the sequences in one common
# file. It also parses the mocap files and extracts the GT sparse keypoints.
# It saves the following:
#
# - sampled points
# - cummulative index ranges for the individual sequences
# - GT sparse keypoints
# - GT mesh areas
# - Scales used to scale each sequence.
#
# [1] J. Yang, S. Wuhrer et al. Estimation of Human Body Shape in Motion
# with Wide Clothing. ECCV 2016.
#

# 3rd party.
import numpy as np
import torch
from pytorch3d.io import load_ply
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

# Project files.
import externals.jblib.file_sys as jbfs
import externals.jblib.mesh as jbm

# Python std
import re

################################################################################
# Settings.
################################################################################

# Paths.
path_data = ''  # <-- Path to the dataset root.
path_ds_out = ''  # <-- Path to the root dir. where to store the generated data.
name_pts_out = 'pts_10k.npy'
name_reg_out = 'regs.npy'
name_areas_out = 'areas.npy'
name_s2i_out = 'seq2inds.npz'
name_scales_out = 'scales.txt'

num_pts = 10000

save_pts = True
save_seq2inds = True
save_reg = True
save_areas = True
save_scales = True

dev = torch.device('cuda')

################################################################################
# Helpers.
################################################################################

# Regex to parse the mocap files.
r = re.compile('(\s+\S+){4}\s+(\S+)\s+(\S+)\s+(\S+)\s+(.+$)$')

# Tf to the OpenGL coord. sys. (x-right, y-up, -z-forward).
Rnp = np.array(
    [[0., 1., 0.],
     [0., 0., 1.],
     [1., 0., 0.]], dtype=np.float32)
Rt = torch.from_numpy(Rnp)


def get_regs(pth):
    """ Parses the sparse GT keypooints from the text files.

    Args:
        pth (str): Path to the file.

    Returns:
        np.array: Keypoints, shape (N, 3).
    """
    regs = []
    with open(pth, 'r') as f:
        coords = f.readlines()[8:19]
    for ln in coords:
        x, y, z, nm = r.findall(ln)[0][1:]
        regs.append(np.array([x, y, z], dtype=np.float32))
    return np.stack(regs, axis=0) * 1e-3


################################################################################
# Main script.
################################################################################

# Get the sequences.
sequences = jbfs.lsd(path_data)
num_seqs = len(sequences)

# Process all sequences.
s2i = {}
pclouds_all = []
regs_all = []
areas_all = []
scales_all = []
processed_samples = 0
for seqi, seq in enumerate(sequences):
    path_seq_scan = jbfs.jn(path_data, seq, 'mesh')
    path_seq_reg = jbfs.jn(path_data, seq, 'lnd')
    samples_scan = jbfs.ls(path_seq_scan, exts='ply')
    samples_reg = jbfs.ls(path_seq_reg, exts='lnd')
    num_samples = len(samples_scan)
    assert len(samples_reg) == num_samples
    s2i[seq] = [processed_samples,
                processed_samples + num_samples]
    processed_samples += num_samples

    # Sample pclouds and get regs.
    pclouds = []
    regs = []
    areas = []
    for i in range(len(samples_scan)):
        print(f"\rProcessing seq {seqi + 1}/{num_seqs}, "
              f"sample {i + 1}/{num_samples}", end='')
        verts, faces = load_ply(jbfs.jn(
            path_seq_scan, samples_scan[i]))
        verts = (Rt @ verts.T).T
        m = Meshes([verts.to(dev)], faces=[faces.to(dev)])
        pclouds.append(sample_points_from_meshes(
            m, num_samples=num_pts)[0])
        regs.append((Rnp @ get_regs(jbfs.jn(
            path_seq_reg, samples_reg[i])).T).T)

        # Save area.
        areas.append(jbm.area(
            verts.cpu().numpy(), faces.cpu().numpy()))

    pclouds = torch.stack(pclouds, dim=0)
    regs = np.stack(regs, axis=0).astype(np.float32)
    areas = np.array(areas, dtype=np.float32)

    # Compute an average scale and scale the pclouds.
    scale = (pclouds - pclouds.mean(dim=1, keepdim=True)).norm(
        dim=2).max(dim=1)[0].mean() * 2.
    pclouds_all.append(
        (pclouds / scale).cpu().numpy().astype(np.float32))
    regs_all.append(regs / scale.item())
    areas_all.append(areas / (scale.item() ** 2))
    scales_all.append(scale.item())

pclouds_all = np.concatenate(pclouds_all, axis=0)
regs_all = np.concatenate(regs_all, axis=0)
areas_all = np.concatenate(areas_all, axis=0)

# Store the pclouds and seq2inds.
if save_pts:
    np.save(jbfs.jn(path_ds_out, name_pts_out), pclouds_all)
if save_seq2inds:
    np.savez_compressed(jbfs.jn(path_ds_out, name_s2i_out), **s2i)
if save_reg:
    np.save(jbfs.jn(path_ds_out, name_reg_out), regs_all)
if save_areas:
    np.save(jbfs.jn(path_ds_out, name_areas_out), areas_all)
if save_scales:
    with open(jbfs.jn(path_ds_out, name_scales_out), 'w') as f:
        for seq, s in zip(sequences, scales_all):
            f.writelines(f"{seq}: {s:.6f}\n")
