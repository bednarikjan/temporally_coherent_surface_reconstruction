# 3rd party
import numpy as np
import torch.utils.data


class PointsTensorDataset(torch.utils.data.Dataset):
    def __init__(self, points_tensor, n_points=0, resample_points=False,
                 normalization_matrices=None, orientation_matrices=None):
        self.points_tensor = points_tensor
        if n_points==0:
            n_points = self.points_tensor.shape[1]
        assert(self.points_tensor.shape[1] >= n_points) # randomly sample n_points from points available in the tensor
        assert(self.points_tensor.shape[2] == 3)  # some normalization code needs to be tweaked to enable arbitrary dims... TODO: support arbitrary dims
        self.resample_points = resample_points
        self.n_points = n_points
        self.normalization_matrices = normalization_matrices
        assert(self.normalization_matrices == None or self.normalization_matrices.shape[0] == self.points_tensor.shape[0])
        self.orientation_matrices = orientation_matrices
        assert(self.normalization_matrices == None or self.orientation_matrices.shape[0] == self.points_tensor.shape[0])

    def __len__(self):
        return self.points_tensor.shape[0]

    def __getitem__(self, idx):
        if self.resample_points:
            sampled_vertices = np.random.choice(range(0, self.points_tensor.shape[1]), self.n_points)
        else:
            sampled_vertices = np.arange(self.n_points, dtype=int)
        if self.orientation_matrices is None and self.normalization_matrices is None:
            return self.points_tensor[idx][sampled_vertices]
        #
        # if dataset needs normalization
        if self.orientation_matrices is not None:
            orientation_matrix = np.eye(4,4)
            orientation_matrix[:3, :3] = self.orientation_matrices[idx]
            if self.normalization_matrices is not None:
                object_normalization = np.matmul(orientation_matrix, self.normalization_matrices[idx])
            else:
                object_normalization = orientation_matrix
        elif self.normalization_matrices is not None:
            object_normalization = self.normalization_matrices[idx]
        else:
            assert(False)
        #
        normalized_point_cloud = self.points_tensor[idx][sampled_vertices]
        ones = np.ones([1, normalized_point_cloud.shape[0]])
        normalized_point_cloud = object_normalization.dot(np.concatenate([normalized_point_cloud.T, ones])).T[:, 0:3]
        #
        return normalized_point_cloud


class PointsTensorMultiDataset(torch.utils.data.Dataset):
    """ Container containing a list of open and memory mapped .npy files. It
    allows for drawing samples for this compound dataset.

    Args:
        pts_tensors (list): Opened .npy files contained the pts.
        num_pts (int): Number of pts to sample.
        resample_pts (bool): Whether to reshuffle the pts.
    """
    def __init__(self, pts_tensors, num_pts=2500, resample_pts=False):
        self._pts_tensors = pts_tensors
        self._num_pts = num_pts
        self._resample_pts = resample_pts
        self._seq_inds = np.concatenate([[0], np.cumsum(
            [pt.shape[0] for pt in pts_tensors])])

    def _idx_glob2loc(self, idx):
        idx_seq = np.sum(idx >= self._seq_inds[1:]).item()
        idx_loc = idx - self._seq_inds[idx_seq]
        return idx_seq, idx_loc

    def __len__(self):
        return self._seq_inds[-1]

    def __getitem__(self, idx):
        idx_seq, idx_loc = self._idx_glob2loc(idx)
        pts = self._pts_tensors[idx_seq][idx_loc]
        inds = np.random.permutation(pts.shape[0]) if self._resample_pts \
            else np.arange(pts.shape[0])
        num_pts = (pts.shape[0], self._num_pts)[self._num_pts > 0]
        return pts[inds[:num_pts]]
