# 3rd party
import torch

# Project files.
import tcsr.train.helpers as tr_helpers


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
