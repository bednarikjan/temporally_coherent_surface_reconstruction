""" Differential geometry properties. Implements the computation of the 1st and
2nd order differential quantities of the 3D points given a UV coordinates and a
mapping f: R^{2} -> R^{3}, which takes a UV 2D point and maps it to a xyz 3D
point. The differential quantities are computed using analytical formulas
involving derivatives d_f/d_uv which are practically computed using Torch's
autograd mechanism. The computation graph is still built and it is possible to
backprop through the diff. quantities computation. The computed per-point
quantities are the following: normals, mean curvature, gauss. curvature.

Author: Jan Bednarik, jan.bednarik@epfl.ch
"""

# 3rd party
import torch
import torch.nn as nn
import torch.autograd as ag
import torch.nn.functional as F

# Project files.
from tcsr.models.common import Device


class DiffGeomProps(nn.Module, Device):
    """ Computes the differential geometry properties including normals,
    mean curvature, gaussian curvature, first fundamental form (fff). Works
    for 2 types of mappings, 2D -> 2D or 2D -> 3D. In the first case, only
    fff is available.

    Args:
        normals (bool): Whether to compute normals (2D -> 3D only).
        curv_mean (bool): Whether to compute mean curvature (2D -> 3D only).
        curv_gauss (bool): Whether to compute gauss. curvature (2D -> 3D only).
        fff (bool): Whether to compute first fundamental form.
        gpu (bool): Whether to use GPU.
    """
    def __init__(self, normals=True, curv_mean=True, curv_gauss=True, fff=False,
                 gpu=True):
        nn.Module.__init__(self)
        Device.__init__(self, gpu=gpu)

        self._comp_normals = normals
        self._comp_cmean = curv_mean
        self._comp_cgauss = curv_gauss
        self._comp_fff = fff

    def forward(self, X, uv):
        """ Computes the 1st and 2nd order derivative quantities, namely
        normals, mean curvature, gaussian curvature, first fundamental form.

        Args:
            X (torch.Tensor): 2D or 3D points in output space (B, M, 2 or 3).
            uv (torch.Tensor): 2D points, parameter space, shape (B, M, 2).

        Returns:
            dict: Depending on `normals`, `curv_mean`, `curv_gauss`, `fff`
                includes normals, mean curvature, gauss. curvature and first
                fundamental form as torch.Tensor.
        """

        # Return values.
        ret = {}

        if not (self._comp_normals or self._comp_cmean or self._comp_cgauss or
                self._comp_fff):
            return ret

        # Data shape.
        B, M, D = X.shape

        # 1st order derivatives d_fx/d_uv, d_fy/d_uv, (d_fz/d_uv).
        dX_duv = []
        for o in range(D):
            derivs = self.df(X[:, :, o], uv)  # (B, M, 2)
            assert(derivs.shape == (B, M, 2))
            dX_duv.append(derivs)

        # Jacobian, d_X / d_uv.
        J_f_uv = torch.cat(dX_duv, dim=2).reshape((B, M, D, 2))
        ret['tangents'] = J_f_uv.transpose(2, 3)  # (B, M, 2, D)

        # normals (only makes sense if D = 3).
        normals = None
        if D == 3:
            normals = F.normalize(
                torch.cross(J_f_uv[..., 0],
                            J_f_uv[..., 1], dim=2), p=2, dim=2)  # (B, M, 3)
            assert (normals.shape == (B, M, 3))

        # Save normals.
        if self._comp_normals:
            if D == 2:
                raise Exception('Normals can only be comupted for '
                                '2D -> 3D mapping.')
            ret['normals'] = normals

        if self._comp_fff or self._comp_cmean or self._comp_cgauss:
            # 1st fundamental form (g)
            g = torch.matmul(J_f_uv.transpose(2, 3), J_f_uv)
            assert (g.shape == (B, M, 2, 2))

            # Save first fundamental form, only E, F, G terms, instead of
            # the whole matrix [E F; F G].
            if self._comp_fff:
                ret['fff'] = g.reshape((B, M, 4))[:, :, [0, 1, 3]]  # (B, M, 3)

        if self._comp_cmean or self._comp_cgauss:
            if D == 2:
                raise Exception('Curvature can only be computed for '
                                '2D -> 3D mapping')

            # determinant of g.
            detg = g[:, :, 0, 0] * g[:, :, 1, 1] - g[:, :, 0, 1] * g[:, :, 1, 0]
            assert (detg.shape == (B, M))

            # 2nd order derivatives, d^2f/du^2, d^2f/dudv, d^2f/dv^2
            d2xyz_duv2 = []
            for o in range(3):
                for i in range(2):
                    deriv = self.df(dX_duv[o][:, :, i], uv)  # (B, M, 2)
                    assert(deriv.shape == (B, M, 2))
                    d2xyz_duv2.append(deriv)

            d2xyz_du2 = torch.stack(
                [d2xyz_duv2[0][..., 0], d2xyz_duv2[2][..., 0],
                 d2xyz_duv2[4][..., 0]], dim=2)  # (B, M, 3)
            d2xyz_dudv = torch.stack(
                [d2xyz_duv2[0][..., 1], d2xyz_duv2[2][..., 1],
                 d2xyz_duv2[4][..., 1]], dim=2)  # (B, M, 3)
            d2xyz_dv2 = torch.stack(
                [d2xyz_duv2[1][..., 1], d2xyz_duv2[3][..., 1],
                 d2xyz_duv2[5][..., 1]], dim=2)  # (B, M, 3)
            assert(d2xyz_du2.shape == (B, M, 3))
            assert(d2xyz_dudv.shape == (B, M, 3))
            assert(d2xyz_dv2.shape == (B, M, 3))

            # Each (B, M)
            gE, gF, _, gG = g.reshape((B, M, 4)).permute(2, 0, 1)
            assert (gE.shape == (B, M))

        # Compute mean curvature.
        if self._comp_cmean:
            cmean = torch.sum((-normals / detg[..., None]) *
                (d2xyz_du2 * gG[..., None] - 2. * d2xyz_dudv * gF[..., None] +
                 d2xyz_dv2 * gE[..., None]), dim=2) * 0.5
            ret['cmean'] = cmean

        # Compute gaussian curvature.
        if self._comp_cgauss:
            iiL = torch.sum(d2xyz_du2 * normals, dim=2)
            iiM = torch.sum(d2xyz_dudv * normals, dim=2)
            iiN = torch.sum(d2xyz_dv2 * normals, dim=2)
            cgauss = (iiL * iiN - iiM.pow(2)) / (gE * gG - gF.pow(2))
            ret['cgauss'] = cgauss

        return ret

    def df(self, x, wrt):
        B, M = x.shape
        return ag.grad(x.flatten(), wrt,
                       grad_outputs=torch.ones(B * M, dtype=torch.float32).
                       to(self.device), create_graph=True)[0]
