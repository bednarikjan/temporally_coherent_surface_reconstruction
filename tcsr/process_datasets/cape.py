# This script generates the CAPE dataset [1] from the RAW scans. It takes the
# original raw scans (noisy non-watertight meshes) and for each sample it
# uniformly randomly resamples 10k points from the surface of the mesh itself.
# It saves the data for all the sequences in one common file. It saves the
# following:
#
# - sampled points
# - cummulative index ranges for the individual sequences
# - GT mesh vertices
# - GT mesh areas
# - GT mesh faces
#
# NOTE: Sampling from the vertices only is not safe as there might
# still be (and there are) spurious vertices far from the true surface.
# Only the faces are reliable at this point (due to how the mesh is
# preprocessed here).
#
# [1] Q. Ma, M. Black et al. Learning to Dress 3D People in Generative Clothing.
# CVPR 2020.
#

# 3rd party.
import torch
import trimesh
import numpy as np

try:
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes
    from pytorch3d.io import load_ply
except:
    print('[ERROR]: Could not import pytorch3d.')

# Python std.
from timeit import default_timer as timer

# Project files.
import externals.jblib.file_sys as jbfs
import externals.jblib.mesh as jbm

################################################################################
# Settings.
################################################################################

# Paths.
path_scans = ''  # <-- Path to cape/raw_scans
path_regs = ''  # <-- Path to cape/registrations
path_ds_out = ''  # <-- Path to the root directory to save the dataset.
name_out_pts_raw = 'pts_raw_from_mesh_10k.npy'
name_out_regs = 'regs.npy'
name_seq2inds = 'seq2ind_raw.npz'
name_out_areas = 'areas.npy'

dev = torch.device('cuda')
num_pts = 10000
verbose = True

process_pts_raw = True
process_seq2inds = True
process_regs = True
process_areas = True

################################################################################
# Helpers.
################################################################################


def preprocess_mesh(verts, faces, th=-600., sc=0.001):
    """ Cuts away the segment of the mesh representing the floor,
    removes spurious chunks of meshes which do not belong to the
    human surface.

    Args:
        verts (np.array[float32]): Vertices, shape (N, 3).
        faces (np.array[float32]): Faces, shape (F, 3).
        th (float): the vertices/faces below `th` aliong the vertical (y) axis
            are discarded.
        sc (float): Scale of the target mesh.
    Returns:
        np.array[float32]: Processed vertices, shape (N', 3), N' <= N.
        np.array[float32]: Processed faces, shape (F', 3), F' <= F.
    """
    # Build mesh.
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Threshold floor vertices.
    m = m.slice_plane((0., th, 0.), (0., 1., 0.))

    # Find con. components and extract the biggest one.
    cc = trimesh.graph.connected_components(
        m.face_adjacency, min_len=3)
    cbig = cc[np.argmax(np.array([c.shape[0]] for c in cc))]

    # Discard the faces of the smaller components.
    msk_f_valid = np.zeros((m.faces.shape[0],), dtype=np.bool)
    msk_f_valid[cbig] = True
    m.update_faces(msk_f_valid)

    return (np.asarray(m.vertices) * sc).astype(np.float32), \
           np.asarray(m.faces).astype(np.int32)


################################################################################
# Main script.
################################################################################

# Outuput data.
pts_all = []
regs_all = []
areas_all = []
seq2inds = {}

# Processing stats.
ts = timer()
elapsed = 0
remaining = float('inf')
processed_samples = 0
num_smpls_tot = 0
for subj in jbfs.lsd(path_scans):
    for seq in jbfs.lsd(jbfs.jn(path_scans, subj)):
        num_smpls_tot += len(jbfs.ls(jbfs.jn(
            path_scans, subj, seq), exts='ply'))

# Process samples.
subjects = jbfs.lsd(path_scans)
for subji, subj in enumerate(subjects):
    path_subj_raw = jbfs.jn(path_scans, subj)
    path_subj_reg = jbfs.jn(path_regs, subj)
    sequences = jbfs.ls(path_subj_raw)
    for seqi, seq in enumerate(sequences):
        path_seq_raw = jbfs.jn(path_subj_raw, seq)
        path_seq_reg = jbfs.jn(path_subj_reg, seq)
        smpls_raw = jbfs.ls(path_seq_raw, exts='ply')
        smpls_reg = jbfs.ls(path_seq_reg, exts='npz')
        assert len(smpls_raw) == len(smpls_reg)

        # Save inds.
        if process_seq2inds:
            seq2inds[f"{subj}_{seq}"] = \
                [processed_samples, processed_samples + len(smpls_raw)]

        for i in range(len(smpls_raw)):
            print(f"\r Processing subj {subji + 1}/{len(subjects)} "
                  f"seq {seqi + 1}/{len(sequences)}, "
                  f"smpl {i + 1}/{len(smpls_raw)}, "
                  f"elpsd: {elapsed / 3600.:.2f} h, "
                  f"rem: {remaining / 3600.:.2f} h", end='')
            # Get pts from raw scans.
            if process_pts_raw or process_areas:
                # Load.
                verts, faces = [x.numpy() for x in load_ply(
                    jbfs.jn(path_seq_raw, smpls_raw[i]))]

                # Preprocess mesh (remove floor and spurious mesh chunks).
                verts, faces = preprocess_mesh(verts, faces)

                # Sample points from the mesh.
                if process_pts_raw:
                    verts_t, faces_t = \
                        [torch.from_numpy(x).to(dev) for x in [verts, faces]]
                    m = Meshes([verts_t], faces=[faces_t])
                    pts_all.append(sample_points_from_meshes(
                        m, num_samples=num_pts)[0].cpu().numpy().
                                   astype(np.float32))
            # Get areas.
            if process_areas:
                areas_all.append(jbm.area(verts, faces))

            # Get GT registrations.
            if process_regs:
                regs_all.append(np.load(jbfs.jn(
                    path_seq_reg, smpls_reg[i]))['v_posed'].astype(np.float32))

            processed_samples += 1

            # Timing.
            elapsed = timer() - ts
            speed = processed_samples / elapsed
            remaining = (num_smpls_tot - processed_samples) / speed

# Store data.
if process_pts_raw:
    pth_out_raw = jbfs.jn(path_ds_out, name_out_pts_raw)
    pts_raw = np.stack(pts_all, axis=0)
    np.save(pth_out_raw, pts_raw)
if process_seq2inds:
    np.savez_compressed(jbfs.jn(path_ds_out, name_seq2inds), **seq2inds)
if process_regs:
    pth_out_regs = jbfs.jn(path_ds_out, name_out_regs)
    pts_reg = np.stack(regs_all, axis=0)
    np.save(pth_out_regs, pts_reg)
if process_areas:
    pth_out_areas = jbfs.jn(path_ds_out, name_out_areas)
    areas = np.array(areas_all)
    np.save(pth_out_areas, areas)
