# Python std
import os
import math
import shutil
from itertools import cycle

# 3rd party
import torch
import numpy as np
from scipy.spatial.transform import Rotation

# Project files.
import externals.jblib.vis3d as jbv3
import externals.jblib.helpers as helpers
import externals.jblib.file_sys as jbfs
from tcsr.models.models_mc import ModelMetricConsistency


class Device:
    """ Creates the computation device.

    Args:
        gpu (bool): Whether to use gpu.
    """
    def __init__(self, gpu=True):
        if gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            if gpu:
                print('[WARNING] cuda not available, using CPU.')


def create_trrun_save_conf(path_conf, key_path_trrun='path_train_run',
                           force_base_dir_perm=False, ds_specific_path=False):
    """ Loads the configuration file (.yaml), creates a new training run output
    dir, saves the config. file into the output dir, returns the config and the
    out path.

    Args:
        path_conf (str): Absolute path to the configuration file.
        key_run (str): Dict key to value storing path to the dir holding
            trianing data.

    Returns:
        conf (dict): Loaded config file.
        out_path (str): Path to new output dir.
    """

    # Load conf.
    conf = helpers.load_conf(path_conf)

    # Get train run path.
    trrun_subdir = create_trrun_name(conf)
    out_path = jbfs.jn(conf[key_path_trrun], trrun_subdir)
    if ds_specific_path:
        seq = conf['sequences']
        assert isinstance(seq, (str, list, tuple))
        if isinstance(seq, (list, tuple)):
            assert len(seq) == 1
            seq_str = seq[0]
        elif isinstance(seq, str):
            assert seq == 'all'
            seq_str = 'all'
        out_path = jbfs.jn(conf[key_path_trrun], seq_str, trrun_subdir)

    # Create train run dir.
    base_dir_exists = os.path.exists(conf[key_path_trrun])
    if os.path.exists(out_path):
        out_path_old = out_path
        out_path = jbfs.unique_dir_name(out_path)
        print('WARNING: The output path {} already exists, creating new dir {}'.
              format(out_path_old, out_path))
    jbfs.make_dir(out_path)
    if force_base_dir_perm:
        os.chmod(out_path, 0o0777)
        if not base_dir_exists:
            os.chmod(conf[key_path_trrun], 0o0777)

    # Save config.
    shutil.copy(path_conf, jbfs.jn(out_path, os.path.basename(path_conf)))

    return conf, out_path


def ds2str(c):
    """ Creates a string from the dataset-specific params.

    Args:
        c (dict): Config.

    Returns:
        str: String representation.
    """
    s = '_DS_'

    # Dataset pairing mode.
    s += f"mode-{c['ds_mode']}"
    if c['ds_mode'] == 'neighbors':
        s += f"{c['ds_mode_params']['max_frames']}"

    # Synth. noise.
    noise = c.get('noise', None)
    if isinstance(noise, float) and noise > 0.:
        s += f"_nstd{noise:.3f}"

    # Augmentation.
    rar = c.get('rand_ax_rot', None)
    if rar is not None:
        rarnm = {'single_axis': 'sax', '3_ortho_axes': '3ax'}[rar]
        s += f"_raxr-{rarnm}_{c['rand_ax_up']}-{c['rand_ax_steps']}-" \
             f"{c['rand_ax_mode']}"
    rtr = c.get('rand_transl', None)
    s += f"_rtr{rtr:.1f}" if rtr is not None else ''

    # Dataset sampling type.
    dss = c.get('ds_sampling', 'standard')
    if dss == 'grow':
        s += f"_smpl_grow-{c['ds_grow_start']}" \
             f"{c['ds_grow_window_init']}"
        dssc = c['ds_grow_conf']
        s += f"-lin{dssc['it_start']:.2e}-{dssc['it_end']:.2e}-" \
             f"st{dssc['step']}"

        dsgs = c['ds_grow_sampling']
        if dsgs['type'] == 'uniform':
            dsg_str = f"-uni-ignh" \
                      f"{('F', 'T')[dsgs['ignore_hist_before_start']]}" \
                      f"-m{dsgs['max_hist_val_mult']:.1f}"
        elif dsgs['type'] == 'rand':
            dsg_str = f"-random"
        s += dsg_str

    # Dataset type.
    dst = c.get('ds_type', '')
    s += ('', f"type-{dst}")[len(dst) > 0]

    # Alignment.
    s += f"_cent{('F', 'T')[c['center']]}_" \
         f"alignrot{('F', 'T')[c['align_rotation']]}"

    return s


def create_trrun_name(c):
    """ Generates a trianing run name given the params in the conf. file.

    Args:
        c (dict): Config file.

    Returns:
        str: Training run name.
    """
    name_base = f"{c['name_base']}_" if len(c.get('name_base', '')) > 0 else ''
    augm = ('', '_augm')[c.get('augmentation', False)]
    alph_mc = ('', f"_mc{c['alpha_mc']:.1e}")[c['loss_mc']]
    loss_ssc = (
        '', f"_ssc{c.get('alpha_ssc', 0.):.1e}-"                    
            f"-cd{c.get('loss_ssc_cd', 'orig')}"
            f"-mc{c.get('loss_ssc_mc', 'orig')}")[c.get('loss_ssc', False)]
    alph_sciso = ('', f"_sciso{c['alpha_scaled_isometry']:.1e}")[
        c['loss_scaled_isometry']]
    dsstr = ds2str(c)

    name = f"{name_base}" + f"{augm}" + f"_p{c['num_patches']}" + \
           f"{alph_mc}" + f"{loss_ssc}" + f"{alph_sciso}" + f"_bs{c['bs']}" + \
           f"{dsstr}"

    if name.startswith('_'):
        name = name[1:]

    return name


def create_model_train(conf):
    """ Creates a model given config file `conf`.

    Args:
        conf (dict): Config. file.

    Returns:
        nn.Module: Model.
    """
    return ModelMetricConsistency(
        M=conf['M'], code=conf['code'], num_patches=conf['num_patches'],
        enc_batch_norm=conf['enc_batch_norm'],
        dec_batch_norm=conf['dec_batch_norm'],
        loss_scaled_isometry=conf['loss_scaled_isometry'],
        alpha_scaled_isometry=conf['alpha_scaled_isometry'],
        alphas_sciso=conf['alphas_sciso'], loss_mmcl=conf['loss_mmcl'],
        loss_mc=conf['loss_mc'], alpha_mc=conf['alpha_mc'],
        loss_ssc=conf['loss_ssc'], alpha_ssc=conf['alpha_ssc'],
        loss_ssc_cd=conf['loss_ssc_cd'], loss_ssc_mc=conf['loss_ssc_mc'],
        gpu=True)


def prepare_uv(num_pts, num_patches):
    """ Generates points spaced in a regular grid in 2D space. If
    `num_pts` cannot be divided into `num_patches` P so that each patch
    would have E x E pts, `num_pts` is adjusted to the closest number
    E ** 2 * num_patches. Every patch thus gets exactly the same set
    of 2D point coordinates.

    Args:
        num_pts (int): # points to generate.
        num_patches (int): # patches the model uses.

    Returns:
        np.array[float32]: Points, (N, 2), N = E ** 2 * P.
        int: Adjusted # sampled points.
    """
    ppp = num_pts / num_patches
    ev = int(round(math.sqrt(ppp)))
    M = int(ev ** 2 * num_patches)
    if M != num_pts:
        print(f"[WARNING]: Cannot split {num_pts} among {num_patches} patches "
              f"regularly, using {M} instead ({ev ** 2} = {ev} * {ev} "
              f"pts per patch).")
    return np.tile(helpers.grid_verts_2d(ev, ev, 1., 1.), (num_patches, 1)), M


def get_patches_colors(mode='patches', conf=None, M=None):
    """ Returns the per-point color.

    Args:
        mode (str): One of:
            'same': Constant green color.
            'patches': Each patch has different unifomr color.
        conf (dict): Config file.
        M (int): # points.

    Returns:
        np.array of float32: Per-point color, shape (N, 3).
    """
    assert mode in ('same', 'patches')

    if mode == 'same':
        clrs = 'green'
    elif mode == 'patches':
        n_patches = conf['num_patches']
        spp = M // n_patches
        assert(np.isclose(spp, M / n_patches))
        clrs_cycle = cycle(list(jbv3.get_contrast_colors().values()))
        clrs = np.ones((M, 3), dtype=np.float32)
        for i in range(n_patches):
            clr = np.array(next(clrs_cycle))
            clrs[i * spp:(i + 1) * spp] *= clr[None]
    else:
        raise Exception('Unsupported mode "{}"'.format(mode))

    return clrs


def pclouds2vis(pcgt, pcp, num_disp, conf, rot=[0., 0., 0.]):
    """ Converts the GT and predicted pclouds to the format suitable for
    Tensorboard visualization - For every sample the GT and predicted pclouds
    are visualized separately, GT is gray, predicted is colored by patches.

    Args:
        pcgt (torch.Tensor): GT pcloud, shape (B, N, 3)
        pcp (torch.Tensor): Pred. pcloud, shape (B, M, 3)

    Returns:
        pcs (torch.Tensor of float32): Pclouds to visualize (num_disp, 2, P, 3),
            P is max(M, N).
        clrs (torch.Tensor of uint8): Per-point colors (num_disp, 2, P, 3)
    """
    B, N = pcgt.shape[:2]
    M = pcp.shape[1]
    assert(pcp.shape[0] == B)
    P = np.maximum(N, M)

    pcgt = torch.cat([pcgt, torch.zeros(
        (B, P - N, 3), dtype=torch.float32)], dim=1)  # (B, P, 3)
    pcp = torch.cat([pcp, torch.zeros(
        (B, P - M, 3), dtype=torch.float32)], dim=1)  # (B, P, 3)
    assert pcgt.shape == (B, P, 3)
    assert pcp.shape == (B, P, 3)

    # Rotate pclouds.
    assert isinstance(rot, (tuple, list)) and len(rot) == 3
    R = torch.from_numpy(Rotation.from_euler(
        'xyz', rot, degrees=True).as_matrix().astype(np.float32))[None] #(1,3,3)
    pcgt = (R @ pcgt.transpose(1, 2)).transpose(1, 2)
    pcp = (R @ pcp.transpose(1, 2)).transpose(1, 2)

    clrs_gt = torch.ones((B, P, 3), dtype=torch.uint8) * 127  # (B, P, 3)
    clrs_pred = torch.from_numpy(np.tile(
        (get_patches_colors(mode='patches', conf=conf, M=M) * 255.0).
            astype(np.uint8), (B, 1, 1)))  # (B, M, 3)
    clrs_pred = torch.cat([clrs_pred, torch.zeros(
        (B, P - M, 3), dtype=torch.uint8)], dim=1)  # (B, P, 3)
    assert clrs_gt.shape == pcgt.shape
    assert clrs_pred.shape == pcp.shape
    assert clrs_gt.dtype == torch.uint8
    assert clrs_pred.dtype == torch.uint8

    pcs = torch.cat([pcgt[:, None], pcp[:, None]], dim=1)[:num_disp] * \
          torch.tensor([1., -1., -1.])
    clrs = torch.cat([clrs_gt[:, None], clrs_pred[:, None]], dim=1)[:num_disp]

    return pcs, clrs


class LRSchedulerFixed:
    def __init__(self, opt, iters, lrfr, verbose=True):
        assert isinstance(iters, (int, list, tuple))
        assert isinstance(lrfr, (float, list, tuple))

        if isinstance(iters, int):
            iters = [iters]
        if isinstance(lrfr, float):
            lrfr = [lrfr] * len(iters)
        assert len(iters) == len(lrfr)

        self._step = 0
        self._opt = opt
        self._iters = np.array(iters)
        self._lrfr = lrfr
        self._verbose = verbose

    def step(self):
        self._step += 1
        it = np.where(self._step == self._iters)[0]
        assert len(it) <= 1
        if len(it) == 1:
            lf = self._lrfr[it[0]]
            lrold = self._opt.param_groups[0]['lr']
            lrnew = lrold * lf
            if self._verbose:
                print(f"[INFO] Reached iter {self._step} and "
                      f"changing lr from {lrold} to {lrnew}.")
            self._opt.param_groups[0]['lr'] = lrnew

    def state_dict(self):
        return {'step': self._step}

    def load_state_dict(self, d):
        self._step = d['step']


def identity(x):
    """ Implements the identity function.
    """
    return x


def distance_matrix_squared(X, Y, chunked=None):
    """ Computes a squared distance matrix between two pclouds.

    Args:
        X (torch.Tensor): Pcloud X, shape (B, N, D).
        Y (torch.Tensor): Pcloud Y, shape (B, M, D).
        chunked (int): If not None, the distance matrix will be computed
            iteratively over chunks of rows (to prevent OOM on a GPU).

    Returns:
        Squared distance matrix, shape (B, N, M).
    """
    # Check shapes.
    assert X.ndim == 3 and Y.ndim == 3
    assert Y.shape[::2] == X.shape[::2]

    if chunked is None:
        dm = (X[:, :, None] - Y[:, None]).square().sum(dim=3)  # (B, N, M)
    else:
        B, N = X.shape[:2]
        M = Y.shape[1]
        iters = math.ceil(N / chunked)
        dm = torch.zeros((B, N, M), dtype=torch.float16).to(X.device)
        for i in range(iters):
            fr, to = i * chunked, (i + 1) * chunked
            dm[:, fr:to] = (X[:, fr:to, None] - Y[:, None]).square().sum(dim=3).type(torch.float16)

    # Get per-point distance matrix.
    return dm


def distance_matrix(X, Y):
    """ Computes a distance matrix between two pclouds.

    Args:
        X (torch.Tensor): Pcloud X, shape (B, N, D).
        Y (torch.Tensor): Pcloud Y, shape (B, M, D).

    Returns:
        Distance matrix, shape (B, N, M).
    """
    return distance_matrix_squared(X, Y).sqrt()


def closest_point(X, Y, distm=None):
    """

    Args:
        X (torch.Tensor): Pcloud X, shape (B, N, D).
        Y (torch.Tensor): Pcloud Y, shape (B, M, D).
        distm (torch.Tensor): Distance matrix, shape (B, N, M).

    Returns:
        inds_X2Y (torch.Tensor[int32]): Forall pt in X, idx of closest pt in Y,
            shape (B, N).
        inds_Y2X (torch.Tensor[int32]): Forall pt in Y, idx of closest pt in X,
            shape (B, M).
    """
    # Check and get shapes.
    assert X.ndim == 3 and Y.ndim == 3
    assert Y.shape[::2] == X.shape[::2]
    B, N = X.shape[:2]
    M = Y.shape[1]

    # Get the distance matrix.
    distm = distance_matrix_squared(X, Y) if distm is None else distm
    assert distm.shape == (B, N, M)

    # Get closest points indices.
    inds_X2Y = distm.argmin(dim=2)  # (B, N)
    inds_Y2X = distm.argmin(dim=1)  # (B, M)
    assert inds_X2Y.shape == (B, N) and inds_Y2X.shape == (B, M)

    return inds_X2Y, inds_Y2X


def regular_spacing(
        num_pts, rng, iters, decay, dev=torch.device('cpu'), verbose=False):
    # Generate inital random pts.
    x_init = torch.empty(
        (num_pts, 2), dtype=torch.float32, device=dev).uniform_(*rng)

    # Helper vars.
    dist_inf = (rng[1] - rng[0]) * 100.
    eye_inf = torch.eye(
        num_pts, dtype=torch.float32, device=dev) * dist_inf

    # Get max initial step.
    step_max = 0.25 * (1. / math.sqrt(num_pts))

    # Process all iters.
    x = x_init.detach().clone()
    for i in range(iters):
        if verbose:
            print(f"\rProcessing iter {i + 1}/{iters}", end='')

        # Get nn distance for all pts.
        dm = ((x[None] - x[:, None]) ** 2.).sum(dim=2)
        dists_min = torch.min(dm + eye_inf, dim=1)[0]

        # Generate random step directions.
        angs = torch.empty(
            (num_pts,), dtype=torch.float32, device=dev). \
            uniform_(0., 2. * math.pi)
        dirs = torch.stack(
            [torch.cos(angs), torch.sin(angs)], dim=1) * step_max

        # Get candidate new positions of points and nn dist.
        x_cand = torch.clip(x + dirs, *rng)
        dm_cand = ((x_cand[None] - x_cand[:, None]) ** 2.).sum(dim=2)
        dists_min_cand = torch.min(dm_cand + eye_inf, dim=1)[0]

        # Move the points which increase the nn distance.
        msk = dists_min_cand > dists_min
        x[msk] = x_cand[msk]

        # Decay the step.
        step_max *= decay

    return x
