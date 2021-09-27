# 3rd party
import torch

# Project files.
import tcsr.train.helpers as tr_helpers


def point_correspondences(pred_a, pred_b, gt_a, gt_b, gt_msk_a, gt_msk_b):
    """ Point correspondences metrics mean squared L2 and mean rank.

    - Mean squared L2:
    Given predicted pclouds `pred_a`, `pred_b` coming from 2D UV samples `uv`
    and given GT points in correspondence `gt_a`, `gt_b`, it computes the
    following. Each point gt_a_i in `gt_a` is assigned an index j of the closest
    predicted point pred_a_j in `pred_a`. That is, a mapping (i, j) is
    constructed. Then for each gt_b_i the L2 distance to the pred_b_j is
    computed following the established correspondences (i, j). The same is then
    done in the opposite direction (from `pred_b` to `pred_a`). Finally, the
    mean of all the measured L2 distances is returned.

    - Mean rank:
    The correspondences are found the same way but instead of mean squared L2,
    the mean rank of the pred. pt. to GT pt. is returned.

    Note:
        - It is assumed that both `pred_a`, `pred_b` come from the same `uv`
        samples.
        - It is assumed that the points in `gt_a`, `gt_b` are ordered so that
        they are in semantic correspondence. Not all the points in `gt_a` and
        `gt_b` might be valid. The validity is given by `gt_msk_[a|b]` and only
        the valid intersection of both is considered for the metric computation.
        - The implementation is differentiable and can be used for training.

    Args:
        pred_a (torch.Tensor): Predicted points for sample A, shape (B, N, 3).
        pred_b (torch.Tensor): Predicted points for sample B, shape (B, N, 3).
        gt_a (torch.Tensor): GT annotations for sample A, shape (B, K, 3).
        gt_b (torch.Tensor): GT annotations for sample B, shape (B, K, 3).
        gt_msk_a (torch.Tensor[bool]): Mask of the valid annotations for
            sample A, shape (B, K).
        gt_msk_b (torch.Tensor[bool]): Mask of the valid annotations for
            sample B, shape (B, K).

    Returns:
        torch.Tensor[float32]: Scalar metric, mean squared L2 distance.
        torch.Tensor[float32]: Scalar metric, mean rank.
        torch.Tensor[float32]: Scalar metric, mean rank normalized by N,
            in [0, 1].
    """
    # Shape and dtype checks.
    B, N = pred_a.shape[:2]
    K = gt_a.shape[1]
    dev = pred_a.device

    assert pred_a.shape == (B, N, 3) and pred_a.shape == pred_b.shape
    assert gt_a.shape == (B, K, 3) and gt_a.shape == gt_b.shape
    assert gt_msk_a.shape == (B, K) and gt_msk_a.shape == gt_msk_b.shape
    assert gt_msk_a.dtype == torch.bool and gt_msk_b.dtype == torch.bool

    # Get valid GT annotations.
    msk = gt_msk_a * gt_msk_b  # (B, K)

    # Process each sample pair in the batch.
    dists_all = []
    ranks_all = []
    for bi in range(B):
        # Check whether any common annotations exist.
        Ki = msk[bi].sum().item()
        if Ki == 0:
            continue

        # Get GT annotations and pred. pts.
        gtai = gt_a[bi][msk[bi]]  # (Ki, 3)
        gtbi = gt_b[bi][msk[bi]]  # (Ki, 3)
        pai = pred_a[bi]  # (N, 3)
        pbi = pred_b[bi]  # (N, 3)

        # Get GT to prediction distance matrices for samples A, B.
        dma = (gtai[:, None] - pai[None]).square().sum(dim=2)  # (Ki, N)
        dmb = (gtbi[:, None] - pbi[None]).square().sum(dim=2)  # (Ki, N)

        # For each GT pt. get an index of the closest pred. pt. in A, B.
        gt2pai = dma.argmin(dim=1)  # (Ki, )
        gt2pbi = dmb.argmin(dim=1)  # (Ki, )

        # Extract dists. of pred. to GT pts in the opposite pcloud in the pair.
        dists_all.append(dmb[range(Ki), gt2pai])  # (Ki, )
        dists_all.append(dma[range(Ki), gt2pbi])  # (Ki, )

        # Get the rannks of the pred. pts closest to the GT.
        rankbi = torch.nonzero(dmb.argsort(dim=1) == gt2pai[:, None])  # (Ki, 2)
        rankai = torch.nonzero(dma.argsort(dim=1) == gt2pbi[:, None])  # (Ki, 2)
        assert torch.allclose(rankbi[:, 0], torch.arange(Ki).to(dev))
        assert torch.allclose(rankai[:, 0], torch.arange(Ki).to(dev))
        ranks_all.append(rankbi[:, 1])  # (Ki, )
        ranks_all.append(rankai[:, 1])  # (Ki, )

    msl2 = torch.cat(dists_all, dim=0).mean().item()
    mr = torch.cat(ranks_all, dim=0).type(torch.float32).mean().item()
    mrn = mr / N

    # Final metrics.
    return msl2, mr, mrn


def chamfer_distance(pc_gt, pc_p):
    """ Chamfer distance.

    Args:
        pc_gt (torch.Tensor): GT pcloud, shape (B, N, 3).
        pc_p (torch.Tensor): Pred. pcloud, shape (B, M, 3).

    Returns:
        float: Scalar metric.
    """
    assert pc_gt.ndim == 3 and pc_p.ndim == 3
    assert pc_gt.shape[::2] == pc_p.shape[::2]

    dm = (pc_gt[:, :, None] - pc_p[:, None]).square().sum(dim=3)  # (B, N, M)
    return (dm.min(dim=2)[0].mean(dim=1) + dm.min(dim=1)[0].
            mean(dim=1)).mean().item()


def rank_dist_kpts(
        pc_gt, kpts_gt, kpts_p, compute_dist=True, compute_rank=True):
    """ Rank and distance of the predicted to GT keypoints.

    Args:
        pc_gt (torch.Tensor): GT pcloud, shape (B, N, D).
        kpts_gt (torch.Tensor): GT kpts, shape (B, K, D).
        kpts_p (torch.Tensor): Predicted kpts, shape (B, K, D).

    Returns:
        rank (torch.Tensor): Normalized rank, shape (B, K).
        dist (torch.Tensor): Eucl. distances, shape (B, K).
    """
    rank = None
    dist_kpts_sqrt = None

    # Check and get shapes.
    assert pc_gt.ndim == 3 and kpts_gt.ndim == 3 and kpts_p.ndim == 3
    B, N, D = pc_gt.shape
    K = kpts_gt.shape[1]
    assert kpts_gt.shape == (B, K, D)
    assert kpts_p.shape == (B, K, D)

    if compute_dist or compute_rank:
        # Distance between GT and pred. kpts.
        dist_kpts = (kpts_gt - kpts_p).square().sum(dim=2)  # (B, K)
        dist_kpts_sqrt = dist_kpts.sqrt()

    if compute_rank:
        # Distance from GT kpts to GT pc.
        chunked = (None, 5000)[kpts_gt.shape[1] > 25000]
        # chunked = (None, 2500)[kpts_gt.shape[1] > 25000]
        dm_kgt2pc = tr_helpers.distance_matrix_squared(
            kpts_gt, pc_gt, chunked=chunked)  # (B, K, N)

        # Find rank.
        rank = (dist_kpts[..., None] > dm_kgt2pc).sum(dim=2) / N  # (B, K)

    return rank, dist_kpts_sqrt


def mean_rank_dist_kpts(pc_gt, kpts_gt, kpts_p):
    """ Averages rank and distance over points and batch. See `rank_dist_kpts`.
    """
    rank, dist_kpts = rank_dist_kpts(pc_gt, kpts_gt, kpts_p)
    return rank.mean(), dist_kpts.mean()


def mean_pck_auc(kpts_gt, kpts_p, d_range=(0., 1.), steps=100):
    """ Percentage of correct keypoints and area under PCK curve.

    Args:
        kpts_gt (torch.Tensor): GT kpts, shape (B, K, D).
        kpts_p (torch.Tensor): Predicted kpts, shape (B, K, D).
        d_range (2-tuple): Minimum and maximum distance.
        steps (int): Number of steps.

    Returns:
        pck (torch.Tensor): Mean PCK, shape (S, ).
        auc (torch.Tensor): Scalar mean area under PCK curve.
    """
    # Check arguments, get shapes.
    assert kpts_gt.ndim == 3 and kpts_p.ndim == 3
    assert kpts_gt.shape == kpts_p.shape
    assert len(d_range) == 2
    assert d_range[1] > d_range[0]
    K = kpts_gt.shape[1]

    # Distance between GT and pred. kpts.
    dists = (kpts_gt - kpts_p).square().sum(dim=2)  # (B, K)

    # Distance below given threshold.
    ths = torch.linspace(
        d_range[0], d_range[1], steps=steps, dtype=torch.float32,
        device=kpts_gt.device)  # (S, )
    pck = (dists[..., None] <= ths[None, None]).\
        type(torch.float32).mean(dim=1)  # (B, S)
    auc = pck.mean(dim=1)  # (B, )

    assert torch.all(pck <= 1.)
    assert torch.all(auc <= 1.)

    return pck.mean(dim=0), auc.mean()
