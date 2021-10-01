# Python std.
import copy
import math

# 3rd party.
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# Project files.
import externals.jblib.vis3d as jbv3
import tcsr.evaluate.metrics as metrics
import tcsr.train.helpers as tr_helpers


def deep_update(d, u):
    if u is not None:
        for k, v in u.items():
            d[k] = deep_update(d.get(k, {}), v) \
                if isinstance(v, dict) else v
    return d


def get_rend_config(conf, ds, ds_spec=None, seq=None, seq_spec=None):
    d = copy.deepcopy(conf['default'])
    d = deep_update(d, conf[ds]['all'][(
        ds_spec, 'default')[ds_spec is None]])

    # Sequence specific.
    if seq is not None and conf[ds]['sequence_specific'] is not None and \
            seq in conf[ds]['sequence_specific']:
        seq_spec = (seq_spec, 'default')[seq_spec is None]

        d = deep_update(
            d, conf[ds]['sequence_specific'][seq][seq_spec])
    return d


def name_from_config(ds_spec=None, seq_spec=None):
    return f"{(ds_spec, 'default')[ds_spec is None]}_" \
           f"{(seq_spec, 'default')[seq_spec is None]}"


def compute_dist_rank_errors(
        ds, model, model_name, conf, confr, dev=torch.device('cuda'),
        compute_dist=True, compute_rank=True):
    """
    Args:
        ds (torch.utils.data.Dataset): Dataset.
        model (torch.nn.Module): Model
        model_name (str): Type of model.
        conf (dict): Model config.
        confr (dict): Rendring config.
        dev: Device
    """
    assert model_name in ('an', 'dsr', 'our', 'cc', 'mc')

    # Process samples
    dists_all = []
    ranks_all = []

    if confr['heatmap']['mode'] == 'fixed':
        smpl1 = ds[confr['heatmap']['mode_params']['fixed_frame']]
        num_smpls = len(ds)
    elif confr['heatmap']['mode'] == 'consecutive':
        num_smpls = len(ds) - 1
        num_verts = ds[0]['pts_reg'].shape[1]
        dists_all.append(np.zeros((num_verts,), dtype=np.float32))
        ranks_all.append(np.zeros((num_verts,), dtype=np.float32))

    if model_name in ('an', 'dsr', 'our', 'mc'):
        # Get number of uv pts.
        P_orig = conf['num_patches']
        n_pts = conf['N']
        grid_size = round(math.sqrt(n_pts / P_orig))

    for idx in range(num_smpls):
        print(f"\rProcessing sample {idx + 1}/{num_smpls}.", end='')

        # Get data.
        if confr['heatmap']['mode'] == 'consecutive':
            smpl1 = ds[idx]
            smpl2 = ds[idx + 1]
        else:
            smpl2 = ds[idx]

        pts = torch.stack(
            [smpl1['pts'][0], smpl2['pts'][0]], dim=0).to(dev)
        kpt_gt_a, kpt_gt_b = torch.stack(
            [smpl1['pts_reg'][0], smpl2['pts_reg'][0]], dim=0).to(dev)

        # Feedforward
        if model_name in ('an', 'dsr', 'our', 'mc'):
            # Get collapsed patches.
            model.predict_mesh(pts, mesh_edge_verts=grid_size)
            inds_nclpsd = [inc.detach().cpu().numpy() for
                           inc in model.collapsed_patches_A(collapsed=False)]
            inds_nclpsd_uni = np.sort(np.unique(np.concatenate(inds_nclpsd)))

            # Get new vals for grid size and tot. num. of predicted pts.
            P_new = inds_nclpsd_uni.shape[0]
            M_orig = P_orig * (grid_size ** 2)
            grid_size_new = math.ceil(math.sqrt(M_orig / P_new))
            M_new = (grid_size_new ** 2) * P_new

            # Feedforward
            with torch.no_grad():
                model.predict_mesh(
                    pts, mesh_edge_verts=grid_size_new, patches=inds_nclpsd_uni,
                    compute_geom_props=False)
            pp_a, pp_b = model.pc_pred
        else:
            # Feedforward.
            pts_a, pts_b = pts
            with torch.no_grad():
                pp_b = model(pts_a.T[None], pts_b.T[None])[0].T
            pp_a = pts_a

        # Compute errors.
        i_kpt2pp_a = tr_helpers.closest_point(kpt_gt_a[None], pp_a[None])[0][0]  # (K, )
        i_pp2kpt_b = tr_helpers.closest_point(pp_b[None], kpt_gt_b[None])[0][0]  # (M, )
        kpt_p_b = kpt_gt_b[i_pp2kpt_b[i_kpt2pp_a]]  # (K, 3)

        ranks, dists = metrics.rank_dist_kpts(
            kpt_gt_b[None], kpt_gt_b[None], kpt_p_b[None],
            compute_dist=compute_dist, compute_rank=compute_rank)

        if compute_rank:
            ranks_all.append(ranks[0].detach().cpu().numpy())
        if compute_dist:
            dists_all.append(dists[0].detach().cpu().numpy())

    if compute_dist:
        dists_all = np.stack(dists_all, axis=0)
    if compute_rank:
        ranks_all = np.stack(ranks_all, axis=0)

    return dists_all, ranks_all


def compute_heatmap_colors(err, th):
    clrs = np.stack([jbv3.get_colors_cmap(
        x, cmap='Reds', mn=0., mx=th) for x in err], axis=0)
    assert clrs.dtype == np.float32
    return clrs


def render_pointclouds(
        renderer, ds, confr, pts_clrs=None, bs=16, dev=torch.device('cuda')):
    """
    Args:
        ds (torch.utils.data.Dataset): Dataset.
        pts_clrs (np.array): Per point color.
        bs (int): Batch size used for rendering.
    """
    # Offset data.
    offs = torch.tensor(
        confr['data']['offset'], dtype=torch.float32, device=dev)

    n_batch = math.ceil(len(ds) / bs)
    images_all = []
    for bi in range(n_batch):
        print(f"\rProcessing batch {bi + 1}/{n_batch}.", end='')
        fr, to = bi * bs, min((bi + 1) * bs, len(ds))

        # Get batch of points.
        pts = []
        centroids = []
        rots = []
        for i in range(fr, to):
            smpl = ds[i]
            pts.append(smpl['pts'][0])
            if 'centroids' in smpl:
                centroids.append(smpl['centroids'][0])
            if 'canonical_rotations' in smpl:
                rots.append(smpl['canonical_rotations'][0])
        pts = torch.stack(pts, dim=0).to(dev)
        if len(centroids) > 0:
            centroids = torch.stack(centroids, dim=0).to(dev)
        if len(rots) > 0:
            rots = torch.stack(rots, dim=0).to(dev)

        # Uncenter and unrotate pts.
        if confr['data']['align_rot']:
            assert len(rots) > 0
            rinv = torch.inverse(rots)
            pts = torch.bmm(rinv, pts.transpose(1, 2)).transpose(1, 2)
        if not confr['data']['center'] and len(centroids) > 0:
            pts = pts + centroids[:, None]

        # Offset points.
        pts = pts + offs

        # Get points colors.
        pclrs = None if pts_clrs is None else pts_clrs[fr:to]
        images_all.append(renderer.render(
            pts, pts_colors=pclrs, keep_alpha=True))
    return np.concatenate(images_all, axis=0)


def render_heatmap(
        renderer, ds, clrs, confr, bs=16, dev=torch.device('cuda')):
    """
    Args:
        ds (torch.utils.data.Dataset): Dataset.
        clrs (np.array): Per vertex colors.
        bs (int): Batch size used for rendering.
    """
    # Offset data.
    offs = torch.tensor(
        confr['data']['offset'], dtype=torch.float32, device=dev)

    n_batch = math.ceil(len(ds) / bs)
    images_all = []
    for bi in range(n_batch):
        print(f"\rProcessing batch {bi + 1}/{n_batch}.", end='')
        fr, to = bi * bs, min((bi + 1) * bs, len(ds))
        curr_bs = to - fr

        pts_reg = []
        rots = []
        cents = []
        for i in range(fr, to):
            smpl = ds[i]
            pts_reg.append(smpl['pts_reg'][0])
            if 'rotations' in smpl:
                rots.append(smpl['rotations'][0])
            if 'centroids' in smpl:
                cents.append(smpl['centroids'][0])
        pts_reg = torch.stack(pts_reg, dim=0).to(dev)
        if len(rots) > 0:
            rots = torch.stack(rots, dim=0).to(dev)
        if len(cents) > 0:
            cents = torch.stack(cents, dim=0).to(dev)

        # TODO: This might not work, remove if it causes troubles
        # Uncenter and unrotate predictions.
        if not confr['data']['align_rot'] and len(rots) > 0:
            rinv = torch.inverse(rots)
            pts_reg = torch.bmm(rinv, pts_reg.transpose(1, 2)).transpose(1, 2)
        if not confr['data']['center'] and len(cents) > 0:
            pts_reg = pts_reg + cents[:, None]
        ###

        # TODO: maybe use this instead? But it only works for some cases.
        # # Uncenter and unrotate predictions.
        # if confr['data']['align_rot']:
        #     if len(rots) > 0:
        #         rinv = torch.inverse(rots)
        #         pts_reg = torch.bmm(
        #             rinv, pts_reg.transpose(1, 2)).transpose(1, 2)
        # if confr['data']['center']:
        #     if len(cents) > 0:
        #         pts_reg = pts_reg + cents[:, None]

        # Offset data.
        pts_reg = pts_reg + offs

        vclrs = torch.from_numpy(clrs[fr:to]).to(dev)
        images_all.append(renderer.render(
            pts_reg, verts_colors=vclrs, keep_alpha=True))
    return np.concatenate(images_all, axis=0)


def render_uv_patches(
        model, renderer, dl, confr, mesh_edge_verts, model_type='mmcl_withenc',
        rot=None):
    images_all = []

    # Offset data.
    offs = torch.tensor(
        confr['data']['offset'], dtype=torch.float32, device=model.device)

    if rot is not None:
        assert rot.shape == (3, 3)
        rot = rot.to(model.device)

    for bi, batch in enumerate(dl):
        print(f"\rProcessing batch {bi + 1}/{len(dl)}.", end='')

        curr_bs = batch['inds'].shape[0] // 2
        assert curr_bs * 2 == batch['inds'].shape[0]

        if model_type in (
                'mmcl_withenc', 'mmcl_noenc', 'mc_globrot_pred',
                'mc_patchtf_pred'):
            x = batch['pts'].reshape((curr_bs, 2, -1, 3))[:, 0]
        elif model_type in ('mc_onehot',):
            x = batch['inds'].reshape((curr_bs, 2))[:, 0]
        else:
            raise Exception(f"Unknown model type {model_type}")

        # Predict
        vp, fp = model.predict_mesh(x, mesh_edge_verts=mesh_edge_verts)
        verts = vp.reshape((curr_bs, -1, 3))

        if rot is not None:
            verts = (rot @ verts.reshape((-1, 3)).T).T.reshape((curr_bs, -1, 3))

        # Uncenter and unrotate predictions.
        if confr['data']['align_rot']:
            assert 'canonical_rotations' in batch
            rinv = torch.inverse(batch['canonical_rotations'].reshape(
                (curr_bs, 2, 3, 3))[:, 0])
            verts = torch.bmm(rinv, verts.transpose(1, 2)).transpose(1, 2)
        if not confr['data']['center'] and 'centroids' in batch:
            verts = verts + batch['centroids'].reshape((curr_bs, 2, 3))[:, 0][:, None]

        # Offset data.
        verts = verts + offs

        images_all.append(renderer.render(verts, keep_alpha=True))
    return np.concatenate(images_all, axis=0)


def feedforward_noncollapsed(model, pts, grid_size, P, inds_nclpsd=None):
    if inds_nclpsd is None:
        # Get collapsed patches.
        model.predict_mesh(pts, mesh_edge_verts=grid_size)
        inds_nclpsd = [inc.detach().cpu().numpy() for
                       inc in model.collapsed_patches_A(collapsed=False)]
        inds_nclpsd = np.sort(np.unique(np.concatenate(inds_nclpsd)))

    # Get new vals for grid size and tot. num. of predicted pts.
    P_new = inds_nclpsd.shape[0]
    M_orig = P * (grid_size ** 2)
    grid_size_new = math.ceil(math.sqrt(M_orig / P_new))

    # Feedforward
    model.predict_mesh(
        pts, mesh_edge_verts=grid_size_new, patches=inds_nclpsd)
    pc_pred = model.pc_pred
    return pc_pred, inds_nclpsd


def compute_colormap_transfer(
        path_cmap, ds, model, model_name, conf, confr,
        dev=torch.device('cuda'), mode='precomputed'):
    assert mode in ('precomputed', 'camera_specific')

    # Get reference colormap.
    colors_template = None
    if mode == 'precomputed':
        colors_template = torch.from_numpy(np.load(path_cmap))
        ref_idx = confr['heatmap']['mode_params']['fixed_frame']
    elif mode == 'camera_specific':
        cmap = plt.imread(path_cmap)[..., :3].astype(np.float32)
        assert np.max(cmap) <= 1.
        ref_idx = confr['texture']['style_args']['camera_plane']['ref_idx']
        pts_ref = ds[ref_idx]['pts_reg'][0].cpu().numpy()
        pts_ref_r = to_camera_frame(
            pts_ref, confr['camera']['azi'], confr['camera']['ele'])
        pts_ref_rn = (pts_ref_r - np.min(pts_ref_r, axis=0)) / \
                     (np.max(pts_ref_r, axis=0) - np.min(pts_ref_r, axis=0))
        uv = pts_ref_rn[:, :2].astype(np.float32)  # (N, 2)
        uvn = np.clip(np.round(
            uv * (np.array(cmap.shape[:2]) - 1)).astype(np.int32),
                      np.array([0, 0]), np.array(cmap.shape[:2]) - 1)
        colors_template = torch.from_numpy(cmap[uvn[:, 0], uvn[:, 1]])

    # Get initial color assignment.
    smpl = ds[ref_idx]
    pts = smpl['pts'][0][None].to(dev)
    kpt_gt = smpl['pts_reg'][0].to(dev)

    pp = pts[0]
    if model_name in ('an', 'dsr', 'our', 'mc'):
        # Get number of uv pts.
        P_orig = conf['num_patches']
        n_pts = conf['N']
        grid_size = round(math.sqrt(n_pts / P_orig))
        pp, inds_nclpsd = \
            feedforward_noncollapsed(model, pts, grid_size, P_orig)
        pp = pp[0]

    i_pp2kpt = tr_helpers.closest_point(pp[None], kpt_gt[None])[0][0]  # (M, )
    clrs_templ_pred = colors_template[i_pp2kpt.cpu()]  # (M, 3)

    # Get colors for all frames.
    colors_pred_all = []  # Each (N, 3)
    # smpl1 = ds[fixed_frame]
    smpl1 = smpl
    num_smpls = len(ds)
    for idx in range(num_smpls):
        print(f"\rProcessing sample {idx + 1}/{num_smpls}.", end='')

        if idx == ref_idx:
            clrs = colors_template  # (N, 3)
        else:
            # Get data.
            smpl2 = ds[idx]
            pts = torch.stack(
                [smpl1['pts'][0], smpl2['pts'][0]], dim=0).to(dev)
            pts_reg = torch.stack(
                [smpl1['pts_reg'][0], smpl2['pts_reg'][0]], dim=0).to(dev)
            kpt_gt_a, kpt_gt_b = pts_reg

            # Get predicted and GT pair.
            if model_name in ('an', 'dsr', 'our', 'mc'):
                pp_ab, _ = feedforward_noncollapsed(
                    model, pts, grid_size, P_orig, inds_nclpsd=inds_nclpsd)
                pp_a, pp_b = pp_ab
            else:
                # Feedforward.
                pts_a, pts_b = pts
                with torch.no_grad():
                    pp_b = model(pts_a.T[None], pts_b.T[None])[0].T

            # Get pred colors.
            i_kpt2pp = tr_helpers.closest_point(
                kpt_gt_b[None], pp_b[None])[0][0]  # (N, )
            clrs = clrs_templ_pred[i_kpt2pp]  # (N, 3)
        colors_pred_all.append(clrs)
    return torch.stack(colors_pred_all, dim=0). \
        cpu().numpy().astype(np.float32)  # (S, N, 3


def to_camera_frame(pts, azi, ele):
    # Get rotation of the object.
    Ra = Rotation.from_euler('xyz', [0., -azi, 0.], degrees=True).as_matrix()
    Re = Rotation.from_euler('xyz', [ele, 0., 0.], degrees=True).as_matrix()
    R = Re @ Ra

    # Rotate the object.
    return (R @ pts.T).T


def compute_uv_transfer(
        ds, model, model_name, conf, confr, dev=torch.device('cuda')):
    # Get reference vertices.
    ref_idx = confr['texture']['style_args']['camera_plane']['ref_idx']
    pts_ref = ds[ref_idx]['pts_reg'][0].cpu().numpy()

    # Hack
    mn = -1.25
    mx = 1.25

    # Compute the reference UV given the camera orientation.
    pts_ref_r = to_camera_frame(
        pts_ref, confr['camera']['azi'], confr['camera']['ele'])

    pts_ref_rn = ((pts_ref_r - np.min(pts_ref_r, axis=0)) /
                  (np.max(pts_ref_r, axis=0) - np.min(pts_ref_r, axis=0))) * \
                 (mx - mn) + mn

    uv_ref = pts_ref_rn[:, :2].astype(np.float32)  # (N, 2)
    uv_ref = torch.from_numpy(uv_ref)

    # Get initial ref. UV assignment.
    smpl = ds[ref_idx]

    # debug
    pts_tmp = smpl['pts'][0]
    if model_name in ('an', 'dsr', 'mc', 'our'):
        pts_our = pts_tmp[:conf['N']][None].to(dev)
    else:
        pts_cc = pts_tmp[None].to(dev)
        pp = pts_cc[0]

    kpt_gt = smpl['pts_reg'][0].to(dev)

    if model_name in ('an', 'dsr', 'mc', 'our'):
        P_orig = conf['num_patches']
        n_pts = conf['M']
        grid_size = round(math.sqrt(n_pts / P_orig))
        pp, inds_nclpsd = \
            feedforward_noncollapsed(model, pts_our, grid_size, P_orig)
        pp = pp[0]

    i_pp2kpt = tr_helpers.closest_point(pp[None], kpt_gt[None])[0][0]  # (M, )
    uv_ref_pred = uv_ref[i_pp2kpt.cpu()]  # (M, 2)

    # Get UVs for all frames.
    uvs_pred_all = []  # Each (N, 2)
    smpl1 = smpl
    num_smpls = len(ds)
    for idx in range(num_smpls):
        print(f"\rProcessing sample {idx + 1}/{num_smpls}.", end='')

        if idx == ref_idx:
            uvs = uv_ref  # (N, 2)
        else:
            # Get data.
            smpl2 = ds[idx]
            pts = torch.stack(
                [smpl1['pts'][0], smpl2['pts'][0]], dim=0).to(dev)
            pts_reg = torch.stack(
                [smpl1['pts_reg'][0], smpl2['pts_reg'][0]], dim=0).to(dev)
            kpt_gt_a, kpt_gt_b = pts_reg

            # Get predicted and GT pair.
            if model_name in ('an', 'dsr', 'mc', 'our'):
                pp_ab, _ = feedforward_noncollapsed(
                    model, pts, grid_size, P_orig, inds_nclpsd=inds_nclpsd)
                pp_a, pp_b = pp_ab
            else:
                # Feedforward.
                pts_a, pts_b = pts
                with torch.no_grad():
                    pp_b = model(pts_a.T[None], pts_b.T[None])[0].T

            # Get pred colors.
            i_kpt2pp = tr_helpers.closest_point(
                kpt_gt_b[None], pp_b[None])[0][0]  # (N, )
            uvs = uv_ref_pred[i_kpt2pp]  # (N, 2)
        uvs_pred_all.append(uvs)
    return torch.stack(uvs_pred_all, dim=0). \
        cpu().numpy().astype(np.float32)  # (S, N, 3


def render_uv_transfer(
        renderer, ds, uvs, confr, bs=16, dev=torch.device('cuda')):
    """
    Args:
        ds (torch.utils.data.Dataset): Dataset.
        uvs (np.array): Per vertex UVs.
        bs (int): Batch size used for rendering.
    """
    # Offset data.
    offs = torch.tensor(
        confr['data']['offset'], dtype=torch.float32, device=dev)

    n_batch = math.ceil(len(ds) / bs)
    images_all = []
    for bi in range(n_batch):
        print(f"\rProcessing batch {bi + 1}/{n_batch}.", end='')
        fr, to = bi * bs, min((bi + 1) * bs, len(ds))

        pts_reg = []
        for i in range(fr, to):
            pts_reg.append(ds[i]['pts_reg'][0])
        pts_reg = torch.stack(pts_reg, dim=0).to(dev)

        # Offset data.
        pts_reg = pts_reg + offs
        uvs_batch = torch.from_numpy(uvs[fr:to]).to(dev)

        images_all.append(renderer.render(pts_reg, uvs_batch, keep_alpha=True))
    return np.concatenate(images_all, axis=0)


def render_reference_GT_UVs(
        renderer, ds, confr, bs=16, dev=torch.device('cuda')):
    # Hack
    mn = -1.25
    mx = 1.25

    # Get reference vertices.
    ref_idx = confr['texture']['style_args']['camera_plane']['ref_idx']
    pts_ref = ds[ref_idx]['pts_reg'][0].cpu().numpy()

    # Compute the reference UV given the camera orientation.
    pts_ref_r = to_camera_frame(
        pts_ref, confr['camera']['azi'], confr['camera']['ele'])
    pts_ref_rn = ((pts_ref_r - np.min(pts_ref_r, axis=0)) /
                  (np.max(pts_ref_r, axis=0) - np.min(pts_ref_r, axis=0))) * \
                 (mx - mn) + mn

    uv_ref = pts_ref_rn[:, :2].astype(np.float32)  # (N, 2)
    uv_ref = np.tile(uv_ref, (len(ds), 1, 1))

    return render_uv_transfer(
        renderer, ds, uv_ref, confr, bs=bs, dev=dev)
