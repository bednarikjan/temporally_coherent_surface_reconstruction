# 3rd party
import torch
import numpy as np
import torch.nn as nn

# Project files.
from tcsr.models.encoder import EncoderPointNet
from tcsr.models.sampler import FNSamplerRandUniform
from tcsr.models.decoder import DecoderMultiPatch, DecoderAtlasNet
from tcsr.models.diff_props import DiffGeomProps
from tcsr.train.helpers import Device
import externals.jblib.mesh as jbm


class ModelMetricConsistency(nn.Module, Device):
    """ As DSRMMCL with PointNet encoder.
    """
    def __init__(self, M=2500, code=1024, num_patches=10, enc_batch_norm=False,
                 dec_batch_norm=False, dec_actf='softplus',
                 loss_scaled_isometry=False, alpha_scaled_isometry=0.,
                 alphas_sciso=None, loss_mc=False, alpha_mc=1.0,
                 loss_ssc=False, alpha_ssc=1.0, loss_ssc_cd='orig',
                 loss_ssc_mc='orig', gpu=True, **kwargs):
        # Checks.
        assert loss_ssc_cd in ('orig', 'all')
        assert loss_ssc_mc in ('orig', 'all')

        # Initialize parents.
        nn.Module.__init__(self)
        Device.__init__(self, gpu=gpu)

        # Save arguments.
        self._num_patches = num_patches
        self._spp = M // num_patches  # Num. of samples per patch.
        self._M = self._spp * num_patches
        self._code = code
        self._num_decoders = num_patches
        self._loss_mc = loss_mc
        self._alpha_mc = alpha_mc
        self._loss_distort = loss_scaled_isometry
        self._alpha_distort = alpha_scaled_isometry
        self._loss_ssc = loss_ssc
        self._alpha_ssc = alpha_ssc
        self._loss_ssc_cd = ('all', loss_ssc_cd)[loss_ssc]
        self._loss_ssc_mc = ('all', loss_ssc_mc)[loss_ssc]
        if loss_scaled_isometry:
            self._alphas_si = {k: torch.tensor(float(v)).to(self.device)
                               for k, v in alphas_sciso.items()}

        # Constants.
        self._zero = torch.tensor(0.).to(self.device)
        self._eps = torch.tensor(1e-20).to(self.device)

        # Declaration of the vars computed in the forward pass.
        self.pc_pred = None
        self.geom_props = None

        # Diff. geom. props object.
        self.dgp = DiffGeomProps(curv_mean=False, curv_gauss=False, fff=True)

        # Build the network.
        self.enc = EncoderPointNet(
            nlatent=code, dim_input=3, batch_norm=enc_batch_norm)
        self.sampler = FNSamplerRandUniform((0., 1.), (0., 1.), M, gpu=gpu)
        self.dec = DecoderMultiPatch(
            num_patches, DecoderAtlasNet, code=code, sample_dim=2,
            batch_norm=dec_batch_norm, activ_fns=dec_actf,
            use_tanh=False, gpu=gpu, **kwargs)

    def collapsed_patches_A(self, max_ratio=1e-3, collapsed=True):
        """ Detects the collapsed patches by inspecting the ratios of their
        areas which are computed analytically.

        Returns:
            list of torch.Tensor of int32: Per within-batch sample indices
                corresponding to the collapsed patches (or non-collapsed if
                `collapsed=False`), shape (P, ) for evey item in list of
                length B, P is # patches.
        """
        E, F, G = self.geom_props['fff']. \
            reshape((-1, self._num_patches, self._spp, 3)). \
            permute(3, 0, 1, 2)  # Each (B, P, spp)
        Ap = (E * G - F ** 2).mean(dim=2).detach()  # (B, P)
        mu_Ap = Ap.mean(dim=1, keepdim=True)  # (B, 1)
        inds = (Ap / (mu_Ap + 1e-30)) < max_ratio  # (B, P), uint8
        inds = inds if collapsed else ~inds
        return [s.nonzero(as_tuple=False).reshape((-1,)).
                    type(torch.int32) for s in inds]

    def _get_fff(self):
        B = self.pc_pred.shape[0]
        spp = self.pc_pred.shape[1] // self._num_patches
        assert spp * self._num_patches == self.pc_pred.shape[1]
        return self.geom_props['fff'].\
            reshape((B, self._num_patches, spp, 3)).\
            permute(3, 0, 1, 2)  # (3, B, P, spp)

    def _get_area_squared(self, EFG=None):
        E, F, G = self._get_fff() if EFG is None else EFG
        return torch.max(E * G - F.pow(2), self._zero)  # (B, P, spp)

    def _get_area(self, EFG=None):
        EFG = self._get_fff() if EFG is None else EFG
        return self._get_area_squared(EFG=EFG).sqrt()  # (B, P, spp)

    def _get_area_and_area_squared(self, EFG=None):
        EFG = self._get_fff() if EFG is None else EFG
        A2 = self._get_area_squared(EFG=EFG)  # (B, P, spp)
        A = A2.sqrt()  # (B, P, spp)
        return A, A2

    def loss_collapse(self, EFG=None, A2_pred=None):
        # Get per-point first fundamental form and local squared area.
        EFG = self._get_fff() if EFG is None else EFG
        E, F, G = EFG
        A2_pred =self._get_area_squared(EFG=EFG) if A2_pred is None else A2_pred

        # Get mean values of E, G fff components.
        muE = E.mean()
        muG = G.mean()

        # Get losses.
        L_stretch = ((E - G).pow(2) / (A2_pred + self._eps)).mean() * \
                    self._alphas_si['stretch']
        L_E = ((E - muE).pow(2) / (A2_pred + self._eps)).mean() * \
              self._alphas_si['E']
        L_G = ((G - muG).pow(2) / (A2_pred + self._eps)).mean() * \
              self._alphas_si['G']
        L_F = (F.pow(2) / (A2_pred + self._eps)).mean() *self._alphas_si['skew']
        return {'L_skew': L_F, 'L_E': L_E, 'L_G': L_G, 'L_stretch': L_stretch}

    def loss_overlap(self, A_gt, A_pred=None):
        A_pred = self._get_area() if A_pred is None else A_pred
        return torch.max(self._zero, A_pred.mean(dim=2).sum(dim=1) - A_gt).\
                   pow(2).mean() * self._alphas_si['total_area']

    def loss_clps_olap(self, A_gt=None, EFG=None, A=None, A2=None):
        """
        """
        # Get per-point first fundamental form, area and squared area.
        if EFG == None:
            EFG = self._get_fff()  # Each (B, P, spp)

        if A is None or A2 is None:
            A, A2 = self._get_area_and_area_squared(EFG=EFG)

        # Loss collapse.
        L_clps = self.loss_collapse(EFG=EFG, A2_pred=A2)

        # Loss total area.
        L_olap = self.loss_overlap(
            A_gt * self._alphas_si['total_area_mult'], A_pred=A) \
            if A_gt is not None else self._zero

        return {'L_skew': L_clps['L_skew'], 'L_E': L_clps['L_E'],
                'L_G': L_clps['L_G'], 'L_stretch': L_clps['L_stretch'],
                'L_Atot': L_olap,
                'loss_sciso': L_clps['L_skew'] + L_clps['L_E'] +
                              L_clps['L_G'] + L_clps['L_stretch'] + L_olap}

    def _distance_matrix(self, pc_N, pc_M):
        """ Computes a distance matrix between two pclouds.

        Args:
            pc_N (torch.Tensor): GT pcloud, shape (B, N, 3)
            pc_M (torch.Tensor): Predicted pcloud, shape (B, M, 3)

        Returns:
            Distance matrix, shape (B, M, N).
        """
        B, M, D = pc_M.shape
        B2, N, D2 = pc_N.shape
        assert B == B2 and D == D2 and D == 3

        x = pc_M.reshape((B, M, 1, D))
        y = pc_N.reshape((B, 1, N, D))
        return (x - y).pow(2).sum(dim=3).sqrt()  # (B, M, N, 3) -> (B, M, N)

    def _register_pts(self, pc_gt, pc_p):
        """

        Args:
            pc_gt:
            pc_p:

        Returns:

        """
        distm = self._distance_matrix(pc_gt, pc_p)  # (B, M, N)
        inds_p2gt = distm.argmin(dim=2)  # (B, M)
        inds_gt2p = distm.argmin(dim=1)  # (B, N)
        return inds_p2gt, inds_gt2p

    def _cd(self, pc_gt, pc_p, inds_p2gt, inds_gt2p):
        """ Extended Chamfer distance.

        Args:
            pc_gt: (B, N, 3)
            pc_p: (B, M, 3)
            inds_p2gt: (B, M)
            inds_gt2p: (B, N)

        Returns:

        """
        # Reshape inds.
        inds_p2gt = inds_p2gt.unsqueeze(2).expand(-1, -1, 3)
        inds_gt2p = inds_gt2p.unsqueeze(2).expand(-1, -1, 3)

        # Get registered points.
        pc_gt_reg = pc_gt.gather(1, inds_p2gt)  # (B, M, 3)
        pc_p_reg = pc_p.gather(1, inds_gt2p)  # (B, N, 3)

        # Compute per-point-pair differences.
        d_p2gt = torch.pow((pc_p - pc_gt_reg), 2).sum(dim=2)  # (B, M)
        d_gt2p = torch.pow((pc_gt - pc_p_reg), 2).sum(dim=2)  # (B, N)

        # Compute scalar loss.
        return d_p2gt.mean() + d_gt2p.mean()

    def _loss_chamfer_distance(self, pc_gt, pc_pred):
        """ Loss functions computing Chamfer distance.

        Args:
            pc_gt (torch.Tensor): GT pcloud, shape (B, N, 3).
            pc_pred (torch.Tensor): Predicted pcloud, shape (B, M, 3).

        Returns:
            torch.Tensor: Scalar loss.
        """
        # Get registrations, get loss.
        inds_p2gt, inds_gt2p = self._register_pts(pc_gt, pc_pred)
        return self._cd(pc_gt, pc_pred, inds_p2gt, inds_gt2p)

    def _loss_metric_consistency(self, EFG=None):
        """ TODO: no interpolation version of loss_mc.
        """
        EFG = self._get_fff() if EFG is None else EFG  # (3, B, P, spp)
        P, spp = EFG.shape[2:]

        B = EFG.shape[1]
        assert B % 2 == 0

        E, F, G = EFG.reshape((3, B // 2, 2, P, spp))  # Each (B', 2, P, spp)
        return ((E[:, 0] - E[:, 1]).pow(2.) + 2. * (F[:, 0] - F[:, 1]).pow(2.) +
                (G[:, 0] - G[:, 1]).pow(2.)).mean()

    def _loss_self_supervised_correspondences(self, pc_gt, pc_p):
        """ Self-supervised correspondence loss.

        Args:
            pc_gt: Shape (B, S, N, 3), B is orig. batch size, S is number
                of rotation steps.
            pc_p: Shape (B, S, M, 3), B is orig. batch size, S is number
                of rotation steps.

        Returns:

        """
        # Get # samples per patch.
        S = pc_gt.shape[1]

        # Get points assignment for the first predicted sample.
        inds_p2gt = self._reg_func_impl(pc_gt[:, 0], pc_p[:, 0])[0]  # (B, M)

        # Extract corresp. GT pts for the predictions of the augmented samples.
        pcg = pc_gt[:, 1:].gather(2, inds_p2gt[:, None, :, None].expand(
            -1, S - 1, -1, 3))  # (B, S - 1, M, 3)

        # Loss.
        return (pcg - pc_p[:, 1:]).square().sum(dim=3).mean()

    def loss(self, pc_gt, Bo, loss_mc=True, loss_ssc=True, loss_distort=False,
             A_gt=None):
        """ TODO: Temporary loss for a model which does not use interpolation.
        """
        # Get # pts, # patches, # samples per patch.
        N = pc_gt.shape[1]
        M = self.pc_pred.shape[1]
        S = pc_gt.shape[0] // Bo
        assert Bo * S == pc_gt.shape[0]
        P = self._num_patches
        spp = self.pc_pred.shape[1] // self._num_patches
        assert spp * P == self.pc_pred.shape[1]

        losses = {'loss_tot': 0.}

        # Chamfer distance.
        if not self._loss_ssc or self._loss_ssc_cd == 'all':
            pcp = self.pc_pred
            pcgt = pc_gt
        elif self._loss_ssc_cd == 'orig':
            pcp = self.pc_pred.reshape((Bo, S, M, 3))[:, 0]
            pcgt = pc_gt.reshape((Bo, S, N, 3))[:, 0]
        L_cd = self._loss_chamfer_distance(pcgt, pcp)
        losses['L_chd'] = L_cd
        losses['loss_tot'] += L_cd

        # Metric consistency.
        if loss_mc and self._loss_mc:
            if not self._loss_ssc or self._loss_ssc_mc == 'orig':
                efg = self.geom_props['fff'].reshape((Bo, P, spp, 3)).\
                    permute(3, 0, 1, 2)
            elif self._loss_ssc_mc == 'all':
                EFG = self._get_fff()
                efg = EFG.reshape((3, Bo, S, P, spp)).transpose(1, 2).\
                    reshape((3, S * Bo, P, spp))
            L_mc = self.loss_mc_no_interp(EFG=efg)
            losses['L_mc_raw'] = L_mc
            losses['L_mc'] = self._alpha_mc * L_mc
            losses['loss_tot'] += losses['L_mc']

        # Self-supervised correspondences.
        if loss_ssc and self._loss_ssc:
            L_ssc = self._loss_self_supervised_correspondences(
                pc_gt.reshape((Bo, S, N, 3)),
                self.pc_pred.reshape((Bo, S, M, 3)))
            losses['L_ssc_raw'] = L_ssc
            losses['L_ssc'] = self._alpha_ssc * L_ssc
            losses['loss_tot'] += losses['L_ssc']

        # Distortion loss (collapse + overlap)
        if loss_distort and self._loss_distort:
            efg = self.geom_props['fff'].reshape((Bo, P, spp, 3)).\
                permute(3, 0, 1, 2)
            A, A2 = self._get_area_and_area_squared(EFG=efg)
            losses_sciso = self.loss_clps_olap(A_gt=A_gt, EFG=efg, A=A, A2=A2)
            lsi = losses_sciso['loss_sciso']
            for k in ['L_skew',  'L_E', 'L_G', 'L_stretch', 'L_Atot']:
                losses[k] = losses_sciso[k]
            losses['L_sciso_raw'] = lsi
            losses['L_sciso'] = lsi * self._alpha_distort
            losses['loss_tot'] += losses['L_sciso']

        return losses

    def _standardize_patches_inds(self, patches=None):
        """ Extracts the list of unique patch indices from an array-like object
        in an ascending order.

        Args:
            patches (array-like): Indices.

        Returns:
            list: Standardized list of indices.
        """
        if patches is not None:
            assert isinstance(patches, (list, tuple, np.ndarray))
            if isinstance(patches, np.ndarray):
                assert patches.ndim == 1
                patches = patches.tolist()
            assert min(patches) >= 0 and \
                   max(patches) < self._num_decoders
            patches = np.sort(np.unique(patches)).tolist()
        else:
            patches = list(np.arange(self._num_decoders))

        return patches

    def _decode(self, uvs, cws, spp, patches):
        """ Decodes the UV samples, given the codewords, to 3D pts.

        Args:
            uvs (torch.Tensor): UV samples, shape (B, M, 2).
            cws (torch.Tensor): Codewords, shape (B, C).
            spp (int): # samples per patch.
            patches (list): List of inds of patches to predict.

        Returns:
            torch.Tensor: Predicted pts, shape (B, M, 3).
        """
        pts_pred = []  # Each (B, S, 3)
        for i, pi in enumerate(patches):
            patch_uv = uvs[:, i * spp:(i + 1) * spp]  # (B, S, 2)
            patch_cw = cws.unsqueeze(1).expand(-1, spp, -1).\
                contiguous()  # (B, 1, C)
            x = torch.cat([patch_uv, patch_cw], 2).\
                contiguous()  # (B, S, C + 2)
            pts_pred.append(self.dec[pi](x))
        return torch.cat(pts_pred, 1).contiguous()  # (B, M, 3)

    def forward(self, x, Bo, uv=None):
        """ The same as `forward_no_interp` but the implementation is
        potentially faster in case of Lssc is used with `loss_ssc_mc`='orig'.
        In that case it only computes the diff. geom. props. for the original
        (non-augmented) samples.

        Args:
            Bo (int): The original batch size (before augmentation).
        """
        # Get batch size, # pts, # samples per patch.
        B = x.shape[0]
        S = B // Bo
        assert Bo * S == B
        C = self._code
        M = self._M if uv is None else uv.shape[0]
        spp = self._spp if uv is None else M // self._num_patches
        assert spp * self._num_patches == M

        # Get UV samples.
        uv = self.sampler(1) if uv is None else torch.from_numpy(
            uv[None].astype(np.float32)).to(self.device)  # (1, M, 2)
        uv_orig = torch.cat([uv] * Bo, dim=1).reshape(Bo, M, 2)  # (Bo, M, 2)
        uv_orig.requires_grad = True

        if self._loss_ssc:
            uv_augm = torch.cat([uv] * (B - Bo), dim=1).\
                reshape((B - Bo), M, 2)  # (B - Bo, M, 2)

            if self._loss_ssc_mc == 'all':
                uv_augm.requires_grad = True
                uv_all = torch.cat([uv_orig, uv_augm], dim=0)

        # Get CWs and reshuffle to keep the orig. samples (not augm.) first.
        cws = self.enc(x.permute((0, 2, 1)))  # (B, C)
        cws = cws.reshape((Bo, S, C)).transpose(0, 1).\
            reshape((S * Bo, C))  # (B, C)

        # Get per-patch pcloud prediction for the orig. and augm. samples.
        patches = self._standardize_patches_inds(None)
        pcp_orig = self._decode(uv_orig, cws[:Bo], spp, patches)  # (Bo, M, 3)
        if self._loss_ssc:
            pcp_augm = self._decode(
                uv_augm, cws[Bo:], spp, patches)  # (B-Bo, M, 3)

        # Reshuffle the predicted points back to the original order.
        self._pc_pred = pcp_orig
        if self._loss_ssc:
            self.pc_pred = torch.cat([pcp_orig, pcp_augm], dim=0).\
                reshape((S, Bo, M, 3)).transpose(0, 1).\
                reshape((Bo * S, M, 3))  # (B, M, 3)

        # Get diff. geom. props.
        if not self._loss_ssc or self._loss_ssc_mc == 'orig':
            self.geom_props = self.dgp(pcp_orig, uv_orig)
        else:
            self.geom_props = self.dgp(self.pc_pred, uv_all)

    def predict(self, x, uv=None, patches=None, compute_geom_props=True):
        """
        Args:
            uv (np.array[float32]): Force UV coordinates, shape (M, 2).

        Returns:

        """
        # Get inds of patches to predict.
        patches = self._standardize_patches_inds(patches)
        self._num_patches = len(patches)

        # Get batch size, # pts, # pts per patch.
        B = x.shape[0]
        M = self._M if uv is None else uv.shape[0]
        spp = self._spp if uv is None else M // self._num_patches
        self._spp = spp
        assert spp * self._num_patches == M

        # Get UV samples.
        uv = self.sampler(1) if uv is None else torch.from_numpy(
            uv[None].astype(np.float32)).to(self.device)  # (1, M, 2)
        self.uv = torch.cat([uv] * B, dim=1).reshape(B, M, 2)  # (B, M, 2)
        self.uv.requires_grad = True

        # Get CWs.
        cws = self.enc(x.permute((0, 2, 1)))  # (B, C)

        # Decode.
        self.pc_pred = self._decode(
            self.uv, cws, self._spp, patches)  # (B, M, 3)

        # Get diff. geom. props.
        if compute_geom_props:
            self.geom_props = self.dgp(self.pc_pred, self.uv)

    def predict_mesh(self, x, mesh_edge_verts=15, patches=None,
                     compute_geom_props=True):
        """ Predicts B samples. Each patch consists of `mesh_edge_verts`^2
        points regularly sampled from the UV space representing a triangulated
        mesh. Generated faces are returned togther with predicted pts.

        Args:
            x (torch.Tensor[float32]): Input pclouds, shape (B, N, 2).
            mesh_edge_verts (int): Num. vertices per patch edge.
            patches (list[int]): Indices of patches to use. If None, all patches
                are used.

        Returns:
            verts (torch.Tensor[float32]): Predicted vertices,
                shape (B, P, spp, 3), where K is # interp. steps, P
                is # patches, spp is # samples per patch,
                spp = `mesh_edge_verts`^2.
            faces (np.array[float32]): Generated mesh faces (the same for
                each patch), shape (F, 3), F is # faces.
        """
        # Get inds of patches to predict.
        patches = self._standardize_patches_inds(patches)
        self._num_patches = len(patches)

        # Get batch size, # pts per patch.
        B = x.shape[0]
        self._spp = mesh_edge_verts * mesh_edge_verts

        # Generate UV samples and faces.
        uv = np.concatenate([jbm.grid_verts_2d(
            mesh_edge_verts, mesh_edge_verts, 1., 1.)] *
                            self._num_patches, axis=0)  # (P * S, 2)
        self.uv = torch.from_numpy(uv.astype(np.float32)).to(self.device).\
            unsqueeze(0).expand(B, -1, -1)  # (B, P * S, 2)
        self.uv.requires_grad = True
        faces = jbm.grid_faces(mesh_edge_verts, mesh_edge_verts)  # (F, 3)

        # Get CWs.
        self.cws = self.enc(x.permute((0, 2, 1)))  # (B, C)

        # Decode.
        self.pc_pred = self._decode(
            self.uv, self.cws, self._spp, patches)  # (B, M, 3)
        verts = self.pc_pred.reshape(
            (B, self._num_patches, self._spp, 3))  # (B, P, S, 3)

        # Get diff. geom. props.
        if compute_geom_props:
            self.geom_props = self.dgp(self.pc_pred, self.uv)

        return verts, faces
