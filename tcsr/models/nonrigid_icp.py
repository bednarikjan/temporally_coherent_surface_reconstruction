# Python std.
import os
import subprocess
import shutil

# Project files.
import externals.jblib.file_sys as jbfs
import tcsr.train.helpers as tr_helpers

# 3rd party
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


class TheaRegister:
    def __init__(self, rounds=25, smooth=3, smooth_last=1, 
                 path_out_dir_tmp='/tmp/register', cleanup=False):
        self._rounds = rounds
        self._smooth = smooth
        self._smooth_last = smooth_last
        self._path_out_dir_tmp = path_out_dir_tmp
        self._cleanup = cleanup
        
        jbfs.make_dir(path_out_dir_tmp)
    
    def _build_cmd(
        self, path_a, path_b, path_file_out, rounds=None, 
        smooth=None, smooth_last=None):
        
        rounds = (rounds, self._rounds)[rounds is None]
        smooth = (smooth, self._smooth)[smooth is None]
        smooth_last = (
            smooth_last, self._smooth_last)[smooth_last is None]
        
        args = f"--rounds {rounds} --smooth {smooth} " \
               f"--smooth-last {smooth_last} " \
               f"{path_a} {path_b} {path_file_out}"
                
        return f"Register {args}"
        
    def register(
        self, path_a, path_b, path_file_out=None, 
        rounds=None, smooth=None, smooth_last=None, 
        cleanup=None):
        
        # Get output path.
        path_file_out = jbfs.jn(self._path_out_dir_tmp, 'out.pts') \
            if path_file_out is None else path_file_out
        path_dir_out = os.path.dirname(path_file_out)
        
        # Build command and run.
        cmd = self._build_cmd(
            path_a, path_b, path_file_out, rounds=rounds, 
            smooth=smooth, smooth_last=smooth_last)
        subprocess.run(cmd.split(), capture_output=True)
        
        # Parse output pts.
        fname_a = jbfs.split_name_ext(
            os.path.basename(path_a))[0]
        pts_def, normals_def = np.transpose(np.loadtxt(
            jbfs.jn(path_dir_out, f"{fname_a}_deformed.pts"), 
            dtype=np.float32).reshape((-1, 2, 3)), (1, 0, 2))
        
        # Remove tmp files.
        if cleanup or cleanup is None and self._cleanup:
            fname_b = jbfs.split_name_ext(
                os.path.basename(path_b))[0]
            os.remove(path_file_out)
            os.remove(jbfs.jn(
                path_dir_out, f"{fname_a}___corr___{fname_b}.pts"))
            os.remove(jbfs.jn(path_dir_out, f"{fname_a}_colored.pts"))
            os.remove(jbfs.jn(path_dir_out, f"{fname_a}_deformed.pts"))
            os.remove(jbfs.jn(path_dir_out, f"{fname_a}_with_offsets.pts"))
            os.remove(jbfs.jn(path_dir_out, f"{fname_b}_colored.pts"))
        
        return pts_def, normals_def


class NonrigidICP:
    modes = ('random', 'propagate_many2one', 'propagate_hungarian',
             'propagate_noproj')

    def __init__(self, ds, rounds=25, smooth=3, smooth_last=1, cleanup=True,
                 mode='random'):
        # Check and save properties.
        self._ds = ds
        assert mode in NonrigidICP.modes
        self._corresp_all = None
        self._mode = mode
        self._projection = None
        self._path_dir_tmp = None
        if mode in ('propagate_many2one', 'propagate_hungarian'):
            self._projection = mode.split('_')[1]
        elif mode == 'propagate_noproj':
            self._path_dir_tmp = jbfs.unique_dir_name(
                '/tmp/nricp_propagate_noproj')
            jbfs.make_dir(self._path_dir_tmp)

        # Create nonrigid ICP object.
        self._threg = TheaRegister(
            rounds=rounds, smooth=smooth, smooth_last=smooth_last,
            cleanup=cleanup,
            path_out_dir_tmp=jbfs.unique_dir_name('/tmp/register'))

    def _precompute_propagation(self):
        # Set the ds pairing mode to consecutive.
        old_pair_mode, old_pair_mode_kwargs = \
            self._ds.set_pairing_mode(mode='consecutive')

        self._corresp_all = []
        for si in range(len(self._ds._seq_start_inds) - 1):
            fr, to = self._ds._seq_start_inds[si:si + 2]
            for i in range(fr, to - 1):
                print(f"\rPrecomputing propagation, "
                      f"seq {si + 1}/{len(self._ds._seq_start_inds) - 1}, "
                      f"smpl {i - fr + 1}/{to - fr - 1}", end='')

                # Get samples.
                smpl = self._ds[i]
                pts_tgt = smpl['pts'][1]
                pts_def = torch.from_numpy(self._threg.register(
                    smpl['paths'][0], smpl['paths'][1])[0])

                # Get predicted correspondences.
                if self._projection == 'many2one':
                    i_def2tgt = tr_helpers.closest_point(
                        pts_def[None], pts_tgt[None])[0][0].numpy()  # (N, )
                elif self._projection == 'hungarian':
                    cm = tr_helpers.distance_matrix_squared(
                        pts_def[None], pts_tgt[None])[0]  # (N, N)
                    i_def2tgt = linear_sum_assignment(cm.cpu().numpy())[1] #(N,)
                else:
                    raise Exception(f"Unknown projection {self._projection}.")
                assert i_def2tgt.shape == (pts_tgt.shape[0], )
                self._corresp_all.append(i_def2tgt)
            self._corresp_all.append(None)  # Sequences separator.
        assert len(self._corresp_all) == self._ds._seq_start_inds[-1]

        # Reset the ds pairing mode to consecutive.
        self._ds.set_pairing_mode(
            mode=old_pair_mode, mode_kwargs=old_pair_mode_kwargs)

    def _compute_propagation_no_projection(self, fr, to):
        # Checks.
        assert self._path_dir_tmp is not None \
               and os.path.exists(self._path_dir_tmp)

        # Set the ds pairing mode to consecutive, turn off point/reg loading.
        old_pair_mode, old_pair_mode_kwargs = \
            self._ds.set_pairing_mode(mode='consecutive')
        old_load_pts, old_load_reg = self._ds.set_loading(
            load_pts=False, load_reg=False)

        # Get tmp path and save the source sample.
        pth_tmp = jbfs.jn(self._path_dir_tmp, 'pts_def.pts')
        shutil.copy(self._ds[fr]['paths'][0], pth_tmp)

        # Process the sample range.
        for i in range(fr, to):
            pts_def, normals_def = self._threg.register(
                pth_tmp, self._ds[i + 1]['paths'][0])
            np.savetxt(pth_tmp, np.concatenate(
                [pts_def, normals_def], axis=1), fmt='%.6f')

        # Reset points loading and ds pairing mode
        self._ds.set_loading(load_pts=old_load_pts, load_reg=old_load_reg)
        self._ds.set_pairing_mode(
            mode=old_pair_mode, mode_kwargs=old_pair_mode_kwargs)

        return pts_def

    def draw_align_pair(self, idx, mode=None):
        # Checks and save args.
        mode = self._mode if mode is None else mode
        assert mode in NonrigidICP.modes

        # Draw smpl pair.
        pair = self._ds[idx]

        if mode == 'random':
            pts_def = torch.from_numpy(self._threg.register(
                pair['paths'][0], pair['paths'][1])[0])
        elif mode in ('propagate_many2one', 'propagate_hungarian'):
            if self._corresp_all is None:
                self._precompute_propagation()
            inds = np.arange(self._corresp_all[0].shape[0])
            for i in range(*pair['inds'].numpy()):
                inds = self._corresp_all[i][inds]
            pts_def = pair['pts'][1][torch.from_numpy(inds).type(torch.int64)]
        elif mode == 'propagate_noproj':
            pts_def = torch.from_numpy(self._compute_propagation_no_projection(
                *pair['inds'].numpy()))

        pair['pts_def'] = pts_def
        return pair
