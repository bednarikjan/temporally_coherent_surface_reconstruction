### Trains the model on a dataset given by parameter 'ds' in the config .yaml file.

# Python std
import argparse
from timeit import default_timer as timer

# project files
import externals.jblib.file_sys as jbfs
import externals.jblib.helpers as helpers
import externals.jblib.deep_learning.torch_helpers as dlt_helpers
import tcsr.train.helpers as tr_helpers
from tcsr.data.data_loader import DatasetClasses, DataLoaderDevicePairs
from tcsr.data.sampler_gradual_growth import BatchSamplerGradualGrow

# 3rd party
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Settings.
max_mesh_vis_pts = 2500
vis_rot = [180., 0., 0.]

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--conf', help='Path to the main config file of the model.')
parser.add_argument('--cont', help='Path to existing training run.')
args = parser.parse_args()

# Load the config file, create training run output path.
if args.cont is None:
    conf, path_trrun = tr_helpers.create_trrun_save_conf(
        args.conf, force_base_dir_perm=True, ds_specific_path=True)
else:
    path_trrun = args.cont
    path_conf, path_trstate = dlt_helpers.get_path_conf_tr_state(path_trrun)
    conf = helpers.load_conf(path_conf)

# Prepare TB writers.
writer_tr = SummaryWriter(jbfs.jn(path_trrun, 'tr'))

# Get data
ds = DatasetClasses[conf['ds']](
    num_pts=conf['N'], subjects=conf.get('subjects', None),
    sequences=conf['sequences'], mode=conf['ds_mode'],
    mode_params=conf['ds_mode_params'], center=conf['center'],
    align_rot=conf['align_rotation'], resample_pts=True, with_area=True,
    rand_ax_rot=conf['rand_ax_rot'], rand_ax_up=conf['rand_ax_up'],
    rand_ax_steps=conf['rand_ax_steps'], rand_ax_mode=conf['rand_ax_mode'],
    rand_transl=conf['rand_transl'], synth_rot=conf['synth_rot'],
    synth_rot_ang_per_frame=conf['synth_rot_ang_per_frame'],
    synth_rot_up=conf['synth_rot_up'], noise=conf['noise'],
    pairing_mode=conf['ds_pairing_mode'],
    pairing_mode_kwargs=conf['ds_pairing_mode_params'],
    ds_type=conf.get('ds_type', 'clean'))
bs = conf['bs'] // 2
assert bs * 2 == conf['bs']
ds_sampling = conf.get('ds_sampling', 'standard')
sampler = None
if ds_sampling == 'standard':
    dl = DataLoaderDevicePairs(DataLoader(
        ds, batch_size=bs, shuffle=True, num_workers=4, drop_last=True),
        gpu=True)
elif ds_sampling == 'grow':
    sampler = BatchSamplerGradualGrow(
        bs, len(ds), start=conf['ds_grow_start'],
        window_init=conf['ds_grow_window_init'],
        window_growth=conf['ds_grow_conf'], unit_abs=True,
        sampling=conf['ds_grow_sampling'])
    dl = DataLoaderDevicePairs(DataLoader(
        ds, batch_sampler=sampler, num_workers=4), gpu=True)
else:
    raise Exception(f"Unknown dataset sampling mode '{ds_sampling}'")

# Build a model.
model = tr_helpers.create_model_train(conf, num_cws=len(ds))
model.train()

# Prepare training.
opt = torch.optim.Adam(model.parameters(), lr=conf['lr'])
lrsched = None
if conf.get('lr_scheduler_fixed'):
    lrsched = tr_helpers.LRSchedulerFixed(
        opt, conf['lr_sched_iters'], conf['lr_sched_fracs'], verbose=True)
it_start = 1

# Continue already existing training.
if args.cont is not None:
    trstate = torch.load(path_trstate)
    model.load_state_dict(trstate['weights'])
    opt.load_state_dict(trstate['optimizer'])
    if 'scheduler' in trstate and conf['lr_scheduler_fixed']:
        lrsched.load_state_dict(trstate['scheduler'])
    it_start = trstate['iterations']
    if 'batch_sampler' in trstate and sampler is not None:
        sampler.load_state_dict(trstate['batch_sampler'])

    del trstate
    torch.cuda.empty_cache()

# Prepare savers.
saver = dlt_helpers.TrainStateSaver(
    jbfs.jn(path_trrun, 'chkpt.tar'), model=model, optimizer=opt,
    scheduler=lrsched, batch_sampler=sampler)

# Training loop.
tstart = timer()
tst_ips = 0.
losses_tr = dlt_helpers.RunningLoss()
dlit = iter(dl)
max_iters = conf.get('max_train_iters', float('inf'))
chkpt_iters = conf.get('checkpoint_iters', None)
B = conf['bs']
for it in range(it_start, max_iters + 1):
    # Iters per s timing.
    if it == it_start + 1:
        tst_ips = timer()

    # LR scheduler.
    if lrsched is not None:
        lrsched.step()

    ### Train.
    try:
        batch = next(dlit)
    except:
        dlit = iter(dl)
        batch = next(dlit)
    if conf.get('rand_ax_rot', False):
        assert batch['pts'].shape[0] == B * conf['rand_ax_steps']
    else:
        assert batch['pts'].shape[0] == B
    pts_gt = batch['pts']
    inds = batch['inds']
    A_gt = batch.get('areas', None)
    x = {'mmcl_withenc': pts_gt, 'mmcl_noenc': inds}[conf['model']]

    # Feedforward.
    model.forward(x, B)
    losses = model.loss(
        pts_gt, B, loss_distort=conf['loss_scaled_isometry'], A_gt=A_gt)

    # Backward.
    opt.zero_grad()
    losses['loss_tot'].backward()
    opt.step()

    losses_tr.update(**{k: v.item() for k, v in losses.items()})
    if it % conf['print_period'] == 0:
        losses_avg = losses_tr.get_losses()
        for k, v in losses_avg.items():
            writer_tr.add_scalar(k, v, it)
        losses_tr.reset()
        writer_tr.add_scalar('lr', opt.param_groups[0]['lr'], it)

        speed = (it - it_start) / (timer() - tst_ips)
        rem = (max_iters - it) / max(speed, 1e-6)
        strh = '\rit {}/{}, {:.0f} m, {:.1f} its/min, rem {:.1f} h - '.format(
            it, max_iters, (timer() - tstart) / 60., speed * 60., rem / 3600.)
        strl = ', '.join(['{}: {:.4f}'.format(k, v)
                          for k, v in losses_avg.items() if '_raw' not in k])
        print(strh + strl, end='')

    # Save number of collapsed patches.
    if (it % conf['save_num_collapsed_patches_period'] == 0
            and 'fff' in model.geom_props) or it >= max_iters:
        num_collpased = np.sum(
            [inds.shape[0] for inds in model.collapsed_patches_A()]) / \
                        model.pc_pred.shape[0]
        writer_tr.add_scalar('collapsed_patches', num_collpased,
                             global_step=it)

    # Save pclouds.
    if conf['pcloud_save_period'] != 0 and \
            (it % conf['pcloud_save_period'] == 0 or it >= max_iters):
        nv = min(conf['num_vis'], B)
        if conf.get('f_chd', 'pow') == 'pcloud_mesh':
            pg = pts_gt.verts_padded().detach().cpu()
            pg = pg[:, torch.from_numpy(
                np.random.permutation(
                    pg.shape[1])[:min(pg.shape[1], max_mesh_vis_pts)])]
        else:
            pg = pts_gt.cpu()
        pcs_vis, clrs_vis = tr_helpers.pclouds2vis(
            pg, model.pc_pred.detach().cpu(), nv, conf, rot=vis_rot)
        assert pcs_vis.shape == (nv, 2, np.maximum(conf['N'], conf['M']), 3)
        assert clrs_vis.shape == pcs_vis.shape
        for idx, (pc, clr) in enumerate(zip(pcs_vis, clrs_vis)):  # (2, P, 3)
            writer_tr.add_mesh('pc_{}'.format(idx), vertices=pc, colors=clr,
                               global_step=it)

    # Save train state.
    if it % conf['train_state_save_period'] == 0 or it >= max_iters:
        saver(iterations=it)

    # Save train state on a special iteration.
    if chkpt_iters is not None:
        if it in chkpt_iters:
            nm_override = f"chkpt_{it}.tar"
            pth_override = jbfs.jn(saver.get_file_dir(), nm_override)
            saver(iterations=it, file_path_override=pth_override)

    # Terminate
    if it >= max_iters:
        print(f"[INFO] Reached preset max iters. {max_iters}, terminating.")
        break
