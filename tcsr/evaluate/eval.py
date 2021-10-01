### Computes the metrics related to the task of (dense) keypoint transfer and
# stores them in .yaml file. Given a ds D of shapes, it draws N random pairs
# (Ai, Bi), transfers the kpts from Ai to Bi and from Bi to Ai and computes the
# metrics.
#
# NOTE: This differs fro 'eval.py' in the fact that the kpts are transferred
# from A_GT all the way to B_GT and then the metrics are computed
# as opposed to 'eval.py' where the kpts are only transferred to B_pred and then
# the metrics are evaluated w.r.t. B_GT. Moreover, 'eval.py' is symmetric in
# the sense that for a pair (A, B) it computes the metrics A->B and B->A, but
# this script is assymetric and only computes A->B to be consistent with how we
# visualize the errors.
#
# - CD
#   Chamfer Distance of the reconstructed shapes. This is not immediately
#   relevant for the task of kpts transfer, but it is a good metric to
#   see how precise the reconstructions are.
#
# - mean L2 distance
#   L2 distance between the predicted and GT kpt averaged over the number
#   of kpts.
#
# - mean rank
#   Rank of every predicted kpt (i.e. how many over predicted pts are closer to
#   the GT kpt tha the predicted kpt) averaged over the number of kpts.
#
# - PCK
# - PCK area under curve
#

# Pythod std.
import math
import gc
import ctypes
libc = ctypes.CDLL("libc.so.6")  # Used to clear the GPU memory.

# Project files.
import externals.jblib.helpers as helpers
import externals.jblib.deep_learning.torch_helpers as dlt_helpers
import tcsr.evaluate.metrics as metrics
import tcsr.train.helpers as tr_helpers
from tcsr.data.data_loader import DatasetClasses

# 3rd party
import torch
import numpy as np
from externals.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import \
    chamfer_3DDist


def eval_trrun(dataset, path_trrun, n_iters, n_pts, subjects=[],
               pck_rng=(0., 0.1), pck_steps=100, dev=torch.device('cuda'),
               uv_pts_mode='uniform_floor', verbose=True, print_results=True):
    """
    """
    assert isinstance(pck_rng, (list, tuple)) and len(pck_rng) == 2
    assert pck_rng[1] > pck_rng[0]
    assert uv_pts_mode in (
        'grid_floor', 'grid_ceil', 'random_floor', 'random_ceil',
        'regular_floor', 'regular_ceil')

    chamfer_distance = chamfer_3DDist()

    # Load trrun config.
    path_conf, path_trstate = dlt_helpers.get_path_conf_tr_state(path_trrun)
    conf = helpers.load_conf(path_conf)
    P_orig = conf['num_patches']

    # Load data.
    subjects = subjects if len(subjects) > 0 else conf.get('subjects', None)
    ds = DatasetClasses[dataset](
        num_pts=conf['N'], subjects=subjects, sequences=conf['sequences'],
        mode='within_seq', center=conf['center'],
        align_rot=conf['align_rotation'], resample_pts=True, with_reg=True,
        synth_rot=conf['synth_rot'],
        synth_rot_ang_per_frame=conf['synth_rot_ang_per_frame'],
        synth_rot_up=conf['synth_rot_up'], noise=conf['noise'],
        ds_type=conf.get('ds_type', 'clean'))

    # Load model.
    model = tr_helpers.create_model_train(conf)
    model.load_state_dict(torch.load(path_trstate)['weights'])
    _ = model.eval()

    # Get number of uv pts.
    grid_size = None
    n_pts_new = None
    if uv_pts_mode in ('grid_floor', 'grid_ceil'):
        grid_size = (math.floor, math.ceil)['ceil' in uv_pts_mode](
            math.sqrt(n_pts / P_orig))
        if P_orig * grid_size ** 2 != n_pts and verbose:
            print(f"[WARNING]: Requested number of points {n_pts} cannot be "
                  f"split into {P_orig} square grids for initial feedforward. "
                  f"Using {P_orig * grid_size ** 2} pts instead.")
    elif uv_pts_mode in (
            'random_floor', 'random_ceil', 'regular_floor', 'regular_ceil'):
        n_pts_new = (math.floor, math.ceil)['ceil' in uv_pts_mode](
            n_pts / P_orig) * P_orig
        if n_pts_new != n_pts and verbose:
            print(f"[WARNING]: Requested number of points {n_pts} cannot be "
                  f"split into {P_orig} patches for initial feedforward, "
                  f"using {n_pts_new} instead.")

    # Prepare dict for storing reg. sampled pts.
    reg_pts_2d = {}

    # Process all random pairs.
    msl2_all = []
    mrankn_all = []
    cd_all = []
    pcks_all = []
    aucs_all = []
    for it in range(n_iters):
        print(f"\rProcessing iter {it + 1}/{n_iters}.", end='')

        # Get data.
        smpl = ds[np.random.randint(len(ds))]
        pts = smpl['pts'].to(dev)
        pts_reg = smpl['pts_reg'].to(dev)

        # Get collapsed patches.
        if uv_pts_mode in ('grid_floor', 'grid_ceil'):
            model.predict_mesh(pts, mesh_edge_verts=grid_size)
        elif uv_pts_mode in ('random_floor', 'random_ceil'):
            uv = np.tile(np.random.uniform(
                0., 1., (n_pts_new // P_orig, 2)).astype(np.float32),
                         (P_orig, 1))
            model.predict(pts, uv=uv)
        elif uv_pts_mode in ('regular_floor', 'regular_ceil'):
            m = n_pts_new // P_orig
            if m in reg_pts_2d:
                uv = reg_pts_2d[m]
            else:
                uv = np.tile(tr_helpers.regular_spacing(
                    m, (0., 1.), 250, 0.994, dev=dev, verbose=verbose).
                             cpu().numpy(), (P_orig, 1))
                reg_pts_2d[m] = np.copy(uv)
            model.predict(pts, uv=uv)

        inds_nclpsd = [inc.detach().cpu().numpy() for
                       inc in model.collapsed_patches_A(collapsed=False)]
        inds_nclpsd_uni = np.sort(np.unique(np.concatenate(inds_nclpsd)))
        if np.any([(inc.shape[0] != inds_nclpsd_uni.shape[0]) or
                   (not np.allclose(inc, inds_nclpsd_uni))
                   for inc in inds_nclpsd]):
            if verbose:
                print(f"[WARNING]: Not all src samples have the same collapsed "
                      f"patches, taking the union of non-collapsed.")

        # Get new vals for grid size and tot. num. of predicted pts.
        P_new = inds_nclpsd_uni.shape[0]
        if P_new == 0:
            if verbose:
                print(f"[WARNING]: All the patches are collapsed, "
                      f"therefore using all the patches.")
            P_new = P_orig
            inds_nclpsd_uni = np.arange(P_orig)

        if uv_pts_mode in ('grid_floor', 'grid_ceil'):
            M_orig = P_orig * (grid_size ** 2)
            grid_size_new = (math.floor, math.ceil)['ceil' in uv_pts_mode](
                math.sqrt(M_orig / P_new))
            M_new = (grid_size_new ** 2) * P_new
            if M_new != M_orig and verbose:
                print(f"[WARNING]: Collapsed patches using {M_new} pts instead "
                      f"of {M_orig} (P: {P_new}).")
            # Feedforward
            with torch.no_grad():
                model.predict_mesh(
                    pts, mesh_edge_verts=grid_size_new, patches=inds_nclpsd_uni,
                    compute_geom_props=False)
        elif uv_pts_mode in (
                'random_floor', 'random_ceil', 'regular_floor', 'regular_ceil'):
            m_curr = (math.floor, math.ceil)['ceil' in uv_pts_mode](
                n_pts / P_new)
            M_new = m_curr * P_new
            if M_new != n_pts and verbose:
                print(f"[WARNING]: Collapsed patches, using {M_new} instead "
                      f"of {n_pts} (P: {P_new}).")
            if 'random' in uv_pts_mode:
                uv = np.tile(np.random.uniform(
                    0., 1., (m_curr, 2)).astype(np.float32), (P_new, 1))
            else:
                if m_curr in reg_pts_2d:
                    uv = reg_pts_2d[m_curr]
                else:
                    uv = tr_helpers.regular_spacing(
                        m_curr, (0., 1.), 250, 0.994, dev=dev, verbose=True).\
                        cpu().numpy()
                    reg_pts_2d[m_curr] = np.copy(uv)
                uv = np.tile(uv, (P_new, 1))
            # Feedforward
            with torch.no_grad():
                model.predict(pts, uv=uv, patches=inds_nclpsd_uni,
                              compute_geom_props=False)

        # Extract predicted points.
        pp_a, pp_b = model.pc_pred
        kpt_gt_a, kpt_gt_b = pts_reg

        # Mean sq. L2 and Mean normalized rank.
        i_kpt2pp_a = tr_helpers.closest_point(
            kpt_gt_a[None], pp_a[None])[0][0]  # (K, )
        i_pp2kpt_b = tr_helpers.closest_point(
            pp_b[None], kpt_gt_b[None])[0][0]  # (M, )
        kpt_p_b = kpt_gt_b[i_pp2kpt_b[i_kpt2pp_a]]  # (K, 3)

        ranks, dists = metrics.rank_dist_kpts(
            kpt_gt_b[None], kpt_gt_b[None], kpt_p_b[None])
        mrankn_all.append(ranks[0].detach().cpu().numpy())
        msl2_all.append(dists[0].detach().cpu().numpy())

        # PCK and auc.
        pck, auc = metrics.mean_pck_auc(
            kpt_gt_b[None], kpt_p_b[None], d_range=pck_rng, steps=pck_steps)
        pcks_all.append(pck.detach().cpu().numpy())
        aucs_all.append(auc.detach().cpu().numpy())

        # CD.
        dp2gt, dgt2p = chamfer_distance(model.pc_pred, pts_reg)[:2]
        cd_all.append((dp2gt.mean() + dgt2p.mean()).detach().cpu().item())

    # Process metrics and export to .yaml.
    msl2_all = np.concatenate(msl2_all, axis=0)
    mrankn_all = np.concatenate(mrankn_all, axis=0)
    res = {'sl2_mu': np.mean(msl2_all).item(), 'sl2_std': np.std(msl2_all).item(),
           'rank_mu': np.mean(mrankn_all).item(), 'rank_std': np.std(mrankn_all).item(),
           'auc_mu': np.mean(aucs_all).item(), 'auc_std': np.std(aucs_all).item(),
           'cd_mu': np.mean(cd_all).item(), 'cd_std': np.std(cd_all).item(),
           'pck': np.mean(np.stack(pcks_all, axis=0), axis=0).tolist()}

    # Clean GPU mem.
    del [ds, model, smpl, pts, pts_reg, pp_a, pp_b, kpt_gt_a, kpt_gt_b,
         i_kpt2pp_a, i_pp2kpt_b, kpt_p_b, ranks, dists, pck, auc, dp2gt, dgt2p]
    gc.collect()
    libc.malloc_trim(0)
    torch.cuda.empty_cache()

    if print_results:
        print(f"Final results - sequence {conf['sequences']}\n=============")
        print(f"sl2 (x1e3): {res['sl2_mu'] * 1000.:.2f}+-{res['sl2_std'] * 1000.:.2f}, "
              f"rank: {res['rank_mu'] * 100.:.2f}+-{res['rank_std'] * 100.:.2f}, "
              f"PCK-auc: {res['auc_mu'] * 100.:.2f}+-{res['auc_std'] * 100.:.2f}, "
              f"cd (x1e3): {res['cd_mu'] * 1000.:.3f}+-{res['cd_std'] * 1000.:.3f}")
        print(f"EXCEL friendly: "
              f"{res['sl2_mu'] * 1000.:.2f}+-{res['sl2_std'] * 1000.:.2f},"
              f"{res['rank_mu'] * 100.:.2f}+-{res['rank_std'] * 100.:.2f},"
              f"{res['auc_mu'] * 100.:.2f}+-{res['auc_std'] * 100.:.2f},"
              f"{res['cd_mu'] * 1000.:.3f}+-{res['cd_std'] * 1000.:.3f}")


    return res
