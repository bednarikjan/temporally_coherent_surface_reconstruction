# 3rd party
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import h5py

# Project files.
import externals.jblib.file_sys as jbfs
from tcsr.data.points_tensor_dataset import PointsTensorDataset, \
    PointsTensorMultiDataset
from tcsr.models.common import Device


class DatasetPairsBase(torch.utils.data.Dataset):
    """ Base class for the dataset which returns pairs of shapes. The following
    coordinate sysem of the samples is assumed: x right, y up, -z forward.
    """

    coord_sys_rot = {
        'x': [90., 0., 90.],
        'y': [0., 0., 0.],
        'z': [0., -90., -90.]
    }
    up_vec = {
        'x': np.array([1., 0., 0.], dtype=np.float32),
        'y': np.array([0., 1., 0.], dtype=np.float32),
        'z': np.array([0., 0., 1.], dtype=np.float32)
    }

    pairing_modes = (
        'standard', 'consecutive', 'fixed_sample_paired', 'fixed_sample')

    subjects_all = None
    sequences_all = None

    def __init__(self, path_pts=None, path_regs=None, path_seq2inds=None,
                 path_areas=None, path_faces=None, path_align_rots=None,
                 num_pts=2500, subjects=None, sequences=None, mode='random',
                 mode_params=None, resample_pts=True, with_reg=False,
                 with_area=False, center=False, align_rot=False,
                 rand_ax_rot=None, rand_ax_up='y', rand_ax_steps=8,
                 rand_ax_mode='uniform', rand_transl=None, synth_rot=False,
                 synth_rot_ang_per_frame=1.2, synth_rot_up='y', noise=None,
                 pairing_mode='standard', pairing_mode_kwargs={}, **kwargs):
        super(DatasetPairsBase, self).__init__()

        # Legacy arguments support.
        if isinstance(rand_ax_rot, bool):
            rand_ax_rot = (None, 'single_axis')[rand_ax_rot]

        # Check the arguments.
        assert rand_ax_rot is None or rand_ax_rot in \
               ('single_axis', '3_ortho_axes', 'fixed_axis')
        assert rand_ax_mode in ('uniform', 'random')
        assert pairing_mode in DatasetPairsBase.pairing_modes
        if pairing_mode == 'fixed_sample':  # Legacy support.
            pairing_mode = 'fixed_sample_paired'
        if rand_ax_rot is not None:
            assert rand_ax_rot in ('single_axis', '3_ortho_axes')
            if rand_ax_rot == '3_ortho_axes':
                assert rand_ax_steps % 3 == 0
        assert mode in ('random', 'within_seq', 'neighbors')
        if mode == 'neighbors':
            assert 'max_frames' in mode_params
        assert (int(align_rot) + int(synth_rot)) < 2, \
            f"The options 'align_rot' and 'synth_rot' are mutually exclusive."

        # Save the properties.
        self._with_reg = with_reg
        self._with_area = with_area

        self._center = center
        self._align_rot = align_rot

        self._rand_ax_rot = rand_ax_rot
        self._rand_ax_coord_sys = DatasetPairsBase.coord_sys_rot[rand_ax_up]
        self._rand_ax_mode = rand_ax_mode
        self._rand_transl = rand_transl
        self._rand_steps = rand_ax_steps

        self._synth_rot = synth_rot
        self._synth_rot_ang_per_frame = synth_rot_ang_per_frame / 180. * np.pi
        self._synth_rot_up = DatasetPairsBase.up_vec[synth_rot_up]
        self._noise = noise

        self._pairing_mode = pairing_mode
        self._pairing_mode_kwargs = pairing_mode_kwargs

        self._mode = mode
        self._mode_params = mode_params

        # Expected properties.
        self._pcloud_ds = None
        self._faces = None
        self._inds_all = None
        self._seq_start_inds = None

        # Get seq. to indices mapping.
        self._seq2inds = self._load_seq2inds(path=path_seq2inds)

        # Get subjects and sequences selection.
        self._subjects, self._sequences = self._get_subjects_sequences(
            subjects=subjects, sequences=sequences)

        # Get samples indices.
        self._inds_all, self._seq_start_inds = self._get_samples_inds(
            subjects=self._subjects, sequences=self._sequences,
            seq2inds=self._seq2inds)

        # Load the pcloud ds.
        self._pcloud_ds = self._load_pcloud_ds(
            path=path_pts, num_pts=num_pts, resample_pts=resample_pts)

        # Get the GT registrations.
        self._regs = self._load_registrations(path_regs) if with_reg else None

        # Get GT areas.
        self._areas = self._load_areas(path_areas) if with_area else None

        # Get faces.
        self._faces = self._load_faces(path_faces)

        # Get GT rotations for aligning the sequence.
        self._rots = None if path_align_rots is None \
            else self._load_alignment_rotations(path_align_rots)

    def get_faces(self, idx):
        """ Returns the GT mesh (registration) faces for the sample `idx`.

        Args:
            idx (int): Index of the sample.

        Returns:
            np.array: Faces, shape (F, 3).
        """
        return self._faces

    def _load_pcloud_ds(self, path=None, num_pts=None, resample_pts=False):
        """ Loads the pcloud dataset.

        Args:
            path (str): Path to the pclouds file.
            num_pts (int): Number of points to load per sample.
            resample_pts (bool): Whether to randomly sample the ds pts while
                loading `num_pts` points.

        Returns:
            torch.Dataset: Dataset.
        """
        assert path is not None
        assert isinstance(num_pts, int) and num_pts > 0
        return PointsTensorDataset(
            np.load(path, mmap_mode='r'), n_points=num_pts,
            resample_points=resample_pts)

    def _load_seq2inds(self, path=None):
        """ Loads the sequences to indices mapping.
        """
        return np.load(path)

    @staticmethod
    def _unify_subj_seq_format(lst, lst_all):
        """ Unifies the format of subjects/sequences iterable. String 'all'
        is replaced with list of all items.

        Args:
            lst (iterable): List of selected items.
            lst_all (iterable): List of all items.

        Returns:
            list: List of selected items.
        """
        if isinstance(lst, str):
            lst = [lst]
        assert isinstance(lst, (tuple, list))
        if len(lst) == 1 and lst[0] == 'all':
            lst = lst_all
        return lst

    def _get_subjects_sequences(self, subjects=None, sequences=None):
        """ Extracts the selected subjects and/or sequences.

        Args:
            subjects (iterable): List of subjects. If None, only sequecnes are
                considered.
            sequences (iterable): List of sequences.

        Returns:
            subjects (list): List of selected subjects.
            sequences (list): List of selected sequences.
        """
        # Checks.
        assert sequences is not None

        # Get selected subjects and sequences.
        subjects = None if subjects is None else \
            sorted(self._unify_subj_seq_format(subjects, self.subjects_all))
        sequences = sorted(self._unify_subj_seq_format(
            sequences, self.sequences_all))

        return subjects, sequences

    def _get_samples_inds(self, subjects=None, sequences=None, seq2inds=None):
        """ Extracts the global indices of the data samples given the selected
        subjects and sequences. Also extracts the local start/end indices of
        each sequence.

        Args:
            subjects (iterable): List of subjects. If None, only sequecnes are
                considered.
            sequences (iterable): List of sequences.
            seq2inds (dict): Mapping of subjects/sequences to sample indices.

        Returns:
            inds_all (list):
            seq_start_inds (list):
        """
        s2i_keys = sequences if subjects is None else \
            [f"{subj}_{seq}" for subj in subjects for seq in sequences]
        inds_all = []
        seq_start_inds = [0]
        for k in s2i_keys:
            if k not in seq2inds:
                print(f"[WARNING]: Requested sequence {k} is no "
                      "part of the dataset and will be ingnored.")
                continue
            fr, to = seq2inds[k]
            inds_all.extend(list(range(fr, to)))
            seq_start_inds.append(seq_start_inds[-1] + to - fr)
        seq_start_inds = np.array(seq_start_inds)

        return inds_all, seq_start_inds

    def _load_registrations(self, path):
        """ Load the GT registrations.
        """
        return np.load(path, mmap_mode='r')

    def _load_areas(self, path):
        """ Load the GT areas.
        """
        return np.load(path, mmap_mode='r')

    def _load_faces(self, path):
        """ Load the faces of the GT meshes (registrations).
        """
        return np.load(path).astype(np.int32)

    def _load_alignment_rotations(self, path):
        """ Load the rotation tfs used to prealign the sequences.
        """
        return np.load(path, mmap_mode='r')

    def __len__(self):
        return len(self._inds_all)

    def _get_pair_inds(self, idx):
        inds_sampled = [idx]
        if self._pairing_mode in ('standard', 'fixed_sample_paired'):
            # Get idx range of the idx1.
            it = np.sum(idx >= self._seq_start_inds)
            fr1, to1 = self._seq_start_inds[it - 1:it + 1]

            # Get the idx range to draw the second idx from.
            if self._mode == 'random':
                smpl_fr, smpl_to = 0, len(self)
            elif self._mode == 'within_seq':
                smpl_fr, smpl_to = fr1, to1
            elif self._mode == 'neighbors':
                mf = self._mode_params['max_frames']
                smpl_fr, smpl_to = \
                    np.maximum(fr1, idx - mf), np.minimum(to1, idx + mf + 1)

            # Get idx2 of the second sample in the pair.
            idx2 = idx
            attempts = 0
            while idx2 == idx:
                idx2 = np.random.randint(smpl_fr, smpl_to)
                attempts += 1
                if attempts > 100:
                    print(f"[ERROR]: Finding a second index in the pair failed "
                          f"after 100 attempts, returning a pair of two "
                          f"identical indices.")
                    break
            inds_sampled.append(idx2)
        elif self._pairing_mode == 'consecutive':
            inds_sampled.append(min(idx + 1, len(self) - 1))

        # Store sampled and finel indices.
        inds_sampled = torch.tensor(inds_sampled, dtype=torch.int64)
        inds_final = inds_sampled

        # Override final indices in case of 'fixed_sample'.
        if self._pairing_mode == 'fixed_sample_paired':
            inds_final = torch.tensor(
                [self._pairing_mode_kwargs['sample_idx']] * 2,
                dtype=torch.int64)

        return inds_final, inds_sampled

    def _get_pts(self, inds):
        """ Gets the pclouds.
        """
        return torch.stack([torch.from_numpy(
            self._pcloud_ds[self._inds_all[i]]) for i in inds], dim=0)

    def _add_gaussian_noise(self, pts, std):
        """ Adds a Gaussian noise to the points.

        Args:
            pts (torch.Tensor): Pcloud, shape (B, N, 3).
            std (float): Std.

        Returns:
            torch.Tensor: Noisy pcloud, shape (B, N, 3).
        """
        return torch.normal(pts, std)

    def _center_pclouds(self, pts):
        """ Centers the pclouds.

        Args:
            pts (torch.Tensor): Pclouds, shape (B, N, 3)

        Returns:
            ptsc (torch.Tensor): Centerd pclouds, shape (B, N, 3).
            centroids (torch.Tensor): Centroids, shape (B, 3).
        """
        centroids = pts.mean(dim=1, keepdim=True)  # (2, 1, 3)
        ptsc = pts - centroids
        return ptsc, centroids[:, 0]

    def _get_rots(self, inds, rots):
        """ Loads the rotations corresponding to the `inds`.

        Args:
            inds (torch.Tensor): Indices of the samples, shape (B, ).
            rots (torch.Tensor): Rotation transformations, shape (B, 3, 3).

        Returns:
            torch.Tensor: Loaded rotations, shape (B, 3, 3).
        """
        indsg = [self._inds_all[i] for i in inds]
        return torch.from_numpy(rots[indsg])  # (B, 3, 3)

    def _rotate_pclouds(self, pts, tfr):
        """ Rotates the pclouds by given rotations.

        Args:
            pts (torch.Tensor): Pclouds, shape (B, N, 3).
            tfr (torch.Tensor): Rotation mats., shape (B, 3, 3).

        Returns:
            torch.Tensor: Rotated pclouds, shape (B, N, 3).
        """
        assert pts.shape[0] == tfr.shape[0]
        return (tfr @ pts.transpose(1, 2)).transpose(1, 2)  # (B, N, 3)

    def _transform_pts(self, inds, pts, center=False,
                       align_rot=False, rots=None):
        """ First centers and then rotates the pclouds `pts`. If only rotation
        is required, it first centers, rotates and then uncenters the pclouds.

        Args:
            inds (torch.Tensor): Indices of the samples, shape (B, ).
            pts (torch.Tensor): Pclouds, shape (B, N, 3).
            center (bool): Whether to center the pclouds.
            align_rot (bool): Whether to rotate the pclouds.
            rots (torch.Tensor): Rotation transformations for the whole DS,
                shape (A, 3, 3), A is # samples in the DS.

        Returns:
            pts (torch.Tensor): Centered and rotated pclouds, shape (B, N, 3).
            centroids (torch.Tensor): Centroids, shape (B, 3).
            rots_applied (torch.Tensor): Rotations, (B, 3, 3).
            rotc (torch.Tensor): Rots away from the can. frame, shape (B, 3, 3).
        """
        # Checks.
        B = pts.shape[0]
        assert not align_rot or rots is not None
        assert inds.ndim == 1
        assert pts.ndim == 3
        assert inds.shape[0] == B

        # Load rotations.
        rots_align = None if rots is None \
            else self._get_rots(inds, rots)  # (B, 3, 3)
        rots_applied = None

        # Center and rotate.
        centroids = None
        if center or align_rot:
            pts, centroids = self._center_pclouds(pts)  # (B, N, 3), (B, 3)
        if align_rot:
            pts = self._rotate_pclouds(pts, rots_align)  # (B, N, 3), (B, 3, 3)
            pts = pts if center else pts + centroids[:, None]
            rots_applied = rots_align

        # Get rotation away from the canonical frame.
        rotc = torch.eye(3, dtype=torch.float32)[None].expand(B, 3, 3)
        if rots_align is not None and not align_rot:
            rotc = rots_align.inverse()  # (B, 3, 3)
        assert rotc.shape == (B, 3, 3)

        return pts, centroids, rots_applied, rotc

    def _get_regs(self, inds):
        """ Gets the registrations.
        """
        return torch.stack([torch.from_numpy(
            self._regs[self._inds_all[i]]) for i in inds], dim=0)

    def _transform_regs(self, pts, centroids=None, rots=None):
        """ Centers and rotates the GT registrations.

        Args:
            pts (torch.Tensor): Pclouds, shape (B, N, 3).
            centroids (torch.Tensor): Centroids, shape (B, 3).
            rots (torch.Tensor): Rotations, shape (B, 3, 3).

        Returns:
            torch.Tensor: Transformed pclouds, shape (B, N, 3).
        """
        if centroids is not None or rots is not None:
            pts = pts - centroids[:, None]
        if rots is not None:
            pts = (rots @ pts.transpose(1, 2)).transpose(1, 2)
            pts = pts if centroids is not None else pts + centroids[:, None]
        return pts

    def _get_areas(self, inds):
        """ Returns the GT mesh areas for the selected samples.

        Args:
            inds (torch.Tensor): Indices of the samples, shape (B, ).

        Returns:
            torch.Tensor: Areas, shape (B, ).
        """
        return torch.tensor(
            [self._areas[self._inds_all[i]] for i in inds], dtype=torch.float32)

    def _transform_synth_rot(self, inds, pts, pts_reg=None):
        """ Transform the pclouds `pts` and GT registrations `pts_reg` by
        synthetically generated rotation.

        Args:
            inds (torch.Tensor): Originally sampled incides to infer
                the rotation angle, shape (B, ).
            pts (torch.Tensor): Pclouds, shape (B, N, 3).
            pts_reg (torch.Tensor): GT reg. pclouds, shape (B, N, 3).

        Returns:
            pts (torch.Tensor): Tfd. Pclouds, shape (B, N, 3).
            pts_reg (torch.Tensor): Tfd. GT reg. pclouds, shape (B, N, 3).
            frots (torch.Tensor): Generted rot. matrices, shape (B, 3, 3).
        """
        # Checks.
        B = inds.shape[0]
        assert inds.ndim == 1
        assert pts.ndim == 3 and pts.shape[0] == B
        if pts_reg is not None:
            assert pts_reg.ndim == 3 and pts_reg.shape[0] == B

        # Get the starting index for this sequence.
        fr = self._seq_start_inds[np.sum(
            inds[0].item() >= self._seq_start_inds) - 1]

        # Get the rot. angle for each sample and gen. rot. matrices.
        angs = [(i.item() - fr) * self._synth_rot_ang_per_frame for i in inds]
        frots = torch.stack([torch.from_numpy(
            Rotation.from_rotvec(self._synth_rot_up * a).as_matrix().astype(
                np.float32)) for a in angs], dim=0)
        assert frots.shape == (B, 3, 3)

        # Rotate the pclouds.
        pts = (frots @ pts.transpose(1, 2)).transpose(1, 2)
        pts_reg = (frots @ pts_reg.transpose(1, 2)).transpose(1, 2) \
            if pts_reg is not None else None
        return pts, pts_reg, frots

    def _rand_rot_axis_upper_hemisphere(self):
        """ Generates a unit length vector representing a rotation axis in
        the upper hemisphere.

        Returns:
            np.array[float32]: Rotation axis, unit lnegth vector, shape (3, ).
        """
        # Generate random orientation in spherical coordinates.
        theta = np.random.uniform(0., 2. * np.pi)
        phi = np.arccos(np.random.uniform(0., 1.))

        # Convert to Cartesian coordinates.
        x = np.sin(theta) * np.sin(phi)
        y = np.cos(phi)
        z = np.cos(theta) * np.sin(phi)
        pt = np.array([x, y, z], dtype=np.float32)

        # Account for the coord. system.
        tf = Rotation.from_euler(
            'xyz', self._rand_ax_coord_sys, degrees=True).inv()
        return tf.apply(pt)

    def _get_rots_single_axis(self, ax, steps, sampling='random'):
        """ Generates rotation matrices corresponding to the rotations around
            the `ax` axis.

        Args:
            ax (np.array[float32]): Rotation axis, shape (3, ).
            steps (int): Number of samples.
            sampling (str): Sampling type.

        Returns:
            torch.Tensor[float32]: Rotation matrices, shape (S, 3, 3), S is
                `self._rand_ax_steps`.
        """
        # Checks.
        assert ax.shape == (3, )
        assert np.isclose(np.linalg.norm(ax), 1.)

        # Generate random angles.
        if sampling == 'uniform':
            angs = np.linspace(0., 2 * np.pi, num=steps, endpoint=False)
        elif sampling == 'random':
            angs = np.random.uniform(0.001, 2. * np.pi - 0.001, (steps, ))
            angs[0] = 0.

        # Get rotation matrices.
        rvecs = ax * angs[:, None]
        return torch.from_numpy(Rotation.from_rotvec(
            rvecs).as_matrix().astype(np.float32))  # (S, 3, 3)

    @staticmethod
    def _generate_3_ortho_axes_y(y):
        """ Generates 3 orthogonal axes x, y, z, so that x, z are randomly
            drawn from the plane orthogonal to `y`.

        Args:
            y (np.array[float32]): Main axis, shape (3, ).

        Returns:
            np.array[float32]: x, y, z axes composed in rows of a matrix
                of shape (3, 3).
        """
        # Checks.
        assert y.shape == (3, )
        assert np.isclose(np.linalg.norm(y), 1.0)

        # Solve the plane eq. ax + by + cz = 0, fill (1, 1) for any axes pair.
        m = -np.sum((1. - np.eye(3)) * y[None], axis=1) / y  # (3, )

        # Find the two orthogonal vectors in the plane ortho. to `up`.
        x = ((1. - np.eye(3)) + np.diag(m))[np.argmax(y)]
        x /= np.linalg.norm(x)
        z = np.cross(x, y)

        # Randomly rotate x, z around up.
        alpha = np.random.uniform(0., 2. * np.pi)
        if alpha > np.pi / 180.:
            R = Rotation.from_rotvec(alpha * y)
            x, z = R.apply(np.stack([x, z], axis=0))

        # Compose the axes as rows of a 3x3 matrix.
        return np.stack([x, y, z], axis=0)

    def _get_rots_3_ortho_axes(self, ax, steps, sampling='random'):
        """ Generates rotation matrices corresponding to the rotations around
            the 3 orthogonal axes. `ax` corresponds to y (up), while x and z
            are drawn randomly so that they are orthogonal to y and to each
            other.

        Args:
            ax (np.array[float32]): Rotation axis, shape (3, ).
            steps (int): Number of samples.
            sampling (str): Sampling type.

        Returns:
            torch.Tensor[float32]: Rotation matrices, shape (S, 3, 3), S is
                `self._rand_ax_steps`.
        """
        # Checks.
        assert ax.shape == (3,)
        assert np.isclose(np.linalg.norm(ax), 1.)

        # Get the 3 axes.
        xyz = self._generate_3_ortho_axes_y(ax)  # (3, 3)

        # Generate random angles.
        if sampling == 'uniform':
            angs = np.tile(np.linspace(
                0., 2 * np.pi, num=steps // 3, endpoint=False), (3, ))
        elif sampling == 'random':
            angs = np.random.uniform(0.001, 2. * np.pi - 0.001, (steps, ))
            angs[0] = 0.

        # Get rotation matrices.
        rvecs = (xyz[:, None] * angs.reshape((3, -1, 1))).reshape((-1, 3))
        return torch.from_numpy(
            Rotation.from_rotvec(rvecs).as_matrix().astype(np.float32)) #(S,3,3)

    def _augm_rot_transl(self, pts, pts_reg=None):
        """ Augments the samples by rotation and translation.

        Args:
            pts (torch.Tensor): Pclouds, shape (B, N, 3).
            pts_reg (torch.Tensor): GT reg. pclouds, shape (B, N, 3).

        Returns:
            pts (torch.Tensor): Augmented pclouds, shape (BS, N, 3).
            pts_reg (torch.Tensor): Augmented GT reg. pclouds, shape (BS, N, 3).
        """
        B = pts.shape[0]

        # Duplicate the pclouds.
        pts = pts[:, None].expand(B, self._rand_steps, -1, -1).reshape(
            B * self._rand_steps, -1, 3)  # (B*S, N, 3)
        pts_reg = None if pts_reg is None else \
            pts_reg[:, None].expand(2, self._rand_steps, -1, -1).reshape(
                2 * self._rand_steps, -1, 3) # (B*S, G, 3)

        # Rotation
        if self._rand_ax_rot is not None:
            # Get a random vector in the upper hemisphere.
            rvec = self._rand_rot_axis_upper_hemisphere()

            # Get rotations
            if self._rand_ax_rot == 'single_axis':
                rots = self._get_rots_single_axis(
                    rvec, self._rand_steps,
                    sampling=self._rand_ax_mode)  # (A, 3, 3)
            elif self._rand_ax_rot == '3_ortho_axes':
                rots = self._get_rots_3_ortho_axes(
                    rvec, self._rand_steps,
                    sampling=self._rand_ax_mode)  # (A, 3, 3)
            elif self._rand_ax_rot == 'fixed_axis':
                raise NotImplementedError

            # Duplicate rots to reflect the number of samples.
            rots = rots[None].expand(B, -1, -1, -1).reshape(
                B * self._rand_steps, 3, 3)

            # Rotate the pclouds.
            pts = torch.bmm(rots, pts.transpose(1, 2)).transpose(1, 2)
            pts_reg = None if pts_reg is None else \
                torch.bmm(rots, pts_reg.transpose(1, 2)).transpose(1, 2)

        # Translation.
        if self._rand_transl is not None:
            transl = torch.empty((self._rand_steps, 3), dtype=torch.float32). \
                uniform_(-self._rand_transl, self._rand_transl)
            transl[0] = torch.zeros((3, ), dtype=torch.float32)  # (S, 3)
            transl = torch.cat([transl] * B, dim=0)  # (B*S, 3)
            pts = pts + transl[:, None]
            pts_reg = None if pts_reg is None else pts_reg + transl[:, None]

        return pts, pts_reg

    def __getitem__(self, idx):
        """ Loads the sample pair.

        Args:
            idx (int): Index of the the first sample in the pair to load.

        Returns:
            dict: Sample.
        """
        # Get indices of the samples to load.
        inds_final, inds_sampled = self._get_pair_inds(idx)

        # Load the pclouds, center, align.
        pts = self._get_pts(inds_final)

        # Add gaussian noise to the observed points.
        if self._noise is not None:
            pts = self._add_gaussian_noise(pts, self._noise)

        # Center, align the pts.
        pts, centroids, rots_applied, rotc = self._transform_pts(
            inds_final, pts, center=self._center, align_rot=self._align_rot,
            rots=self._rots)

        # Load GT registrations, center, align.
        pts_reg = None
        if self._with_reg:
            pts_reg = self._get_regs(inds_final)
            pts_reg = self._transform_regs(
                pts_reg, centroids=centroids, rots=rots_applied)

        # Load areas.
        areas = None
        if self._with_area:
            areas = self._get_areas(inds_final)

        # Add a synthetic rotation.
        if self._synth_rot:
            pts, pts_reg, synth_rots = self._transform_synth_rot(
                inds_sampled, pts, pts_reg=pts_reg)
            rotc = torch.bmm(rotc, synth_rots)
            rots_applied = synth_rots if rots_applied is None else \
                torch.bmm(synth_rots, rots_applied)  # (B, 3, 3)

        # Translation and rotation augmentation.
        if self._rand_ax_rot is not None or self._rand_transl is not None:
            pts, pts_reg = self._augm_rot_transl(pts, pts_reg=pts_reg)

        # Form the sample.
        smpl = {k: v for k, v in zip(
            ['inds', 'pts', 'pts_reg', 'centroids',
             'rotations', 'areas', 'canonical_rotations'],
            [inds_final, pts, pts_reg, centroids, rots_applied, areas, rotc])
                if v is not None}

        return smpl


class DatasetDFAUSTPairs(DatasetPairsBase):
    """ Dataset [1].

    [1] F. Bogo et al. Dynamic FAUST: Registering Human Bodies in Motion.
    CVPR 2017.

    Args:
        ds_type (str): The origin of the pclouds:
            'clean': Sampled from the GT registration meshes.
            'raw_verts': Sampled from the vertices of the raw meshes.
            'raw_faces': Sampled from the faces of the raw meshes.
    """
    path_pts = '/cvlabsrc1/cvlab/datasets_jan/dfaust/registrations_all_10k.npy'
    path_pts_raw_from_verts = '/cvlabsrc1/cvlab/datasets_jan/dfaust/registrations_all_raw_from_verts_10k.npy'
    path_pts_raw_from_meshes = '/cvlabsrc1/cvlab/datasets_jan/dfaust/registrations_all_raw_from_mesh_10k.npy'
    path_reg_m = '/cvlabsrc1/cvlab/datasets_jan/dfaust/registrations_m.hdf5'
    path_reg_f = '/cvlabsrc1/cvlab/datasets_jan/dfaust/registrations_f.hdf5'
    path_seq2inds = '/cvlabsrc1/cvlab/datasets_jan/dfaust/registrations_all_10k_seq2ind.npz'
    path_areas = '/cvlabsrc1/cvlab/datasets_jan/dfaust/registrations_areas_all.npy'

    subjects_all = [
        '50002', '50004', '50007', '50009', '50020',
        '50021', '50022', '50025', '50026', '50027']
    sequences_all = [
        'chicken_wings', 'hips', 'jiggle_on_toes', 'jumping_jacks', 'knees',
        'light_hopping_loose', 'light_hopping_stiff', 'one_leg_jump',
        'one_leg_loose', 'punching', 'running_on_spot', 'shake_arms',
        'shake_hips', 'shake_shoulders']

    def __init__(self, ds_type='clean', **kwargs):
        # Get dataset type.
        assert ds_type in ('clean', 'raw_verts', 'raw_mesh')
        path_ds = {
            'clean': DatasetDFAUSTPairs.path_pts,
            'raw_verts': DatasetDFAUSTPairs.path_pts_raw_from_verts,
            'raw_mesh': DatasetDFAUSTPairs.path_pts_raw_from_meshes,
        }[ds_type]

        # Init. the parent class.
        super(DatasetDFAUSTPairs, self).__init__(
            path_pts=path_ds, path_seq2inds=DatasetDFAUSTPairs.path_seq2inds,
            path_areas=DatasetDFAUSTPairs.path_areas, **kwargs)

        # Get indices of GT registrations.
        self._reg_keys_inds = self._get_reg_key_inds(
            subjects=self._subjects, sequences=self._sequences,
            seq2inds=self._seq2inds)
        assert len(self._inds_all) == len(self._reg_keys_inds)

    def _load_registrations(self, *args, **kwargs):
        """ Load the GT registrations.
        """
        self._regs_m = h5py.File(DatasetDFAUSTPairs.path_reg_m, 'r')
        self._regs_f = h5py.File(DatasetDFAUSTPairs.path_reg_f, 'r')
        return None

    def _load_faces(self, *args, **kwargs):
        """ Load the faces of the GT meshes (registrations).
        """
        if not hasattr(self, '_regs_m'):
            self._regs_m = h5py.File(DatasetDFAUSTPairs.path_reg_m, 'r')
        return np.array(self._regs_m['faces'])

    def _get_reg_key_inds(self, subjects=None, sequences=None, seq2inds=None):
        """ Extracts the global indices of the data samples given the selected
        subjects and sequences. Also extracts the local start/end indices of
        each sequence.
        """
        s2i_keys = [f"{subj}_{seq}" for subj in subjects for seq in sequences]
        reg_keys_inds = []
        for k in sorted(list(set(s2i_keys).intersection(set(seq2inds.keys())))):
            fr, to = seq2inds[k]
            reg_keys_inds.extend([(k, i) for i in range(to - fr)])
        return reg_keys_inds

    def _get_regs(self, inds):
        """ Gets the registrations.
        """
        pts_gt = []
        for i in inds:
            rk, ri = self._reg_keys_inds[i]
            rf = (self._regs_f, self._regs_m)[rk in self._regs_m]
            pts_gt.append(rf[rk][..., ri].astype(np.float32))
        return torch.from_numpy(np.stack(pts_gt, axis=0))


class DatasetAMAPairs(DatasetPairsBase):
    """ Dataset [1].

    [1] D. Vlasic et al. Articulated mesh animation from multi-view silhouettes.
    TOG 2008.
    """
    path_pts = '/cvlabsrc1/cvlab/datasets_jan/articulated_mesh_animation/pts_10k.npy'
    path_pts_gt = '/cvlabsrc1/cvlab/datasets_jan/articulated_mesh_animation/pts_gt.npy'
    path_pts_gt_valid = '/cvlabsrc1/cvlab/datasets_jan/articulated_mesh_animation/pts_gt_valid.npy'
    path_seq2inds = '/cvlabsrc1/cvlab/datasets_jan/articulated_mesh_animation/pts_10k_seq2ind.npz'
    path_faces = '/cvlabsrc1/cvlab/datasets_jan/articulated_mesh_animation/faces_gt.npz'
    path_areas = '/cvlabsrc1/cvlab/datasets_jan/articulated_mesh_animation/areas_gt.npy'
    path_rotations = '/cvlabsrc1/cvlab/datasets_jan/articulated_mesh_animation/rots_120.npy'

    sequences_all = ['bouncing', 'crane', 'handstand', 'jumping', 'march_1',
                     'march_2', 'samba', 'squat_1', 'squat_2', 'swing']

    def __init__(self, **kwargs):
        super(DatasetAMAPairs, self).__init__(
            path_pts=DatasetAMAPairs.path_pts,
            path_regs=DatasetAMAPairs.path_pts_gt,
            path_seq2inds=DatasetAMAPairs.path_seq2inds,
            path_areas=DatasetAMAPairs.path_areas,
            path_faces=DatasetAMAPairs.path_faces,
            path_align_rots=DatasetAMAPairs.path_rotations, **kwargs)

        # Checks.
        if self._with_reg and len(self._sequences) > 1:
            raise Exception(f"If registrations ('with_reg') are required, only "
                            f"a single sequences can be loaded. Requested "
                            f"{len(self._sequences)} sequences.")

        # Get the GT registrations.
        self._regs_valid = np.load(
            DatasetAMAPairs.path_pts_gt_valid, mmap_mode='r')

    def get_faces(self, idx):
        return self._faces[np.sum(idx >= self._seq_start_inds) - 1]

    def _load_faces(self, path):
        """ Load the faces of the GT meshes (registrations).
        """
        fcs = np.load(path)
        return [fcs[seq].astype(np.int32) for seq in self._sequences]

    def _get_regs(self, inds):
        """ Gets the registrations.
        """
        pts_gt = []
        for i in inds:
            ig = self._inds_all[i]
            pts_gt.append(self._regs[ig][:self._regs_valid[ig]])
        return torch.from_numpy(np.stack(pts_gt, axis=0))


class DatasetAnimalsPairs(DatasetPairsBase):
    """ Datasets [1], [2].

    [1] G. Aujay et al. Harmonic Skeleton for Realistic Character Animation.
    SGIGGRAPH 2007.
    [2] R. Sumner. Deformation Transfer for Triangle Meshes. SIGGRAPH 2004.
    """
    path_ds = '/cvlabdata1/cvlab/datasets_jan/animals/datasets/'
    name_pts = 'pts_10k.npy'
    name_pts_gt = 'pts_gt.npy'
    name_faces = 'faces_gt.npy'
    name_areas = 'areas_gt.npy'

    sequences_all = ['cat_walk', 'horse_gallop', 'horse_collapse',
                     'camel_gallop', 'camel_collapse', 'elephant_gallop']

    def __init__(self, **kwargs):
        super(DatasetAnimalsPairs, self).__init__(**kwargs)

    def _load_seq2inds(self, **kwargs):
        return None

    def _get_samples_inds(self, **kwargs):
        return None, None  # _seq_start_inds is set later via _load_pcloud_ds.

    def _load_pcloud_ds(self, path=None, num_pts=None, resample_pts=False):
        """ Loads the pcloud dataset.
        """
        assert isinstance(num_pts, int) and num_pts > 0
        ds = PointsTensorMultiDataset([np.load(jbfs.jn(
            DatasetAnimalsPairs.path_ds, s, DatasetAnimalsPairs.name_pts),
            mmap_mode='r') for s in self._sequences], num_pts=num_pts,
            resample_pts=resample_pts)
        self._seq_start_inds = ds._seq_inds
        return ds

    def _load_registrations(self, *args, **kwargs):
        """ Load the GT registrations.
        """
        return PointsTensorMultiDataset([np.load(jbfs.jn(
            DatasetAnimalsPairs.path_ds, s, DatasetAnimalsPairs.name_pts_gt),
            mmap_mode='r') for s in self._sequences], num_pts=0)

    def _load_areas(self, *args, **kwargs):
        """ Load the GT areas.
        """
        return np.concatenate([np.load(jbfs.jn(
            DatasetAnimalsPairs.path_ds, s, DatasetAnimalsPairs.name_areas))
            for s in self._sequences], axis=0)

    def _load_faces(self, *args, **kwargs):
        """ Load the faces of the GT meshes (registrations).
        """
        return {s: np.load(jbfs.jn(
            DatasetAnimalsPairs.path_ds, s, DatasetAnimalsPairs.name_faces)).
            astype(np.int32) for s in self._sequences}

    def __len__(self):
        return len(self._pcloud_ds)

    def _get_pts(self, inds):
        """ Gets the pclouds.
        """
        return torch.stack([torch.from_numpy(
            self._pcloud_ds[i.item()]) for i in inds], dim=0)

    def _get_regs(self, inds):
        """ Gets the registrations.
        """
        return torch.stack([torch.from_numpy(
            self._regs[i.item()]) for i in inds], dim=0)

    def _get_areas(self, inds):
        """ Returns the GT mesh areas for the selected samples.
        """
        return torch.tensor(
                [self._areas[i] for i in inds], dtype=torch.float32)

    def get_faces(self, idx):
        """ Returns the GT mesh (registration) faces for the sample `idx`.
        """
        seq = self._sequences[self._pcloud_ds._idx_glob2loc(idx)[0]]
        return self._faces[seq]


class DatasetCAPEPairs(DatasetPairsBase):
    """ Dataset [1] consisting of sequences of people in clothing in motion.
    Clean and raw meshes and GT registrations are available.

    [1] Q. Ma, M. Black et al. Learning to Dress 3D People in Generative
    Clothing. CVPR 2020.

    Args:
        ds_type (str): The origin of the pclouds:
            'clean': Sampled from the GT registration meshes.
            'raw_verts': Sampled from the vertices of the raw meshes.
            'raw_faces': Sampled from the faces of the raw meshes.
    """
    path_ds_raw_from_meshes = '/cvlabdata1/cvlab/datasets_jan/cape/raw_scans/pts_raw_from_mesh_10k.npy'  # TODO
    path_ds_raw_from_verts = ''  # not yet implemented
    path_seq2inds_raw = '/cvlabdata1/cvlab/datasets_jan/cape/raw_scans/seq2ind_raw.npz'
    path_regs_raw = '/cvlabdata1/cvlab/datasets_jan/cape/raw_scans/regs.npy'
    path_faces_raw = '/cvlabdata1/cvlab/datasets_jan/cape/raw_scans/faces.npy'
    path_areas_raw = '/cvlabdata1/cvlab/datasets_jan/cape/raw_scans/areas.npy'
    path_ds_clean = ''  # not yet implemented
    path_seq2inds = ''  # not yet implemented
    path_regs = ''  # not yet implemented
    path_faces = ''  # not yet implemented
    path_areas = ''  # not yet implemented

    subjects_all = ['00032', '00096', '00159', '03223']
    sequences_all = [
        'shortlong_hips', 'shortlong_pose_model',
        'shortlong_shoulders_mill', 'shortlong_tilt_twist_left',
        'shortshort_hips', 'shortshort_pose_model',
        'shortshort_shoulders_mill', 'shortshort_tilt_twist_left']

    def __init__(self, ds_type='raw_mesh', **kwargs):
        assert ds_type in ('clean', 'raw_verts', 'raw_mesh')
        # TODO
        if ds_type != 'raw_mesh':
            raise NotImplementedError

        # Get paths.
        path_ds = {
            'clean': DatasetCAPEPairs.path_ds_clean,
            'raw_verts': DatasetCAPEPairs.path_ds_raw_from_verts,
            'raw_mesh': DatasetCAPEPairs.path_ds_raw_from_meshes}[ds_type]
        path_seq2inds, path_regs, path_faces, path_areas = {
            'cle': [DatasetCAPEPairs.path_seq2inds, DatasetCAPEPairs.path_regs,
                    DatasetCAPEPairs.path_faces, DatasetCAPEPairs.path_areas],
            'raw': [DatasetCAPEPairs.path_seq2inds_raw,
                    DatasetCAPEPairs.path_regs_raw,
                    DatasetCAPEPairs.path_faces_raw,
                    DatasetCAPEPairs.path_areas_raw]}[ds_type[:3]]

        # Init. the parent class.
        super(DatasetCAPEPairs, self).__init__(
            path_pts=path_ds, path_regs=path_regs, path_seq2inds=path_seq2inds,
            path_areas=path_areas, path_faces=path_faces, **kwargs)


class DatasetINRIAPairs(DatasetPairsBase):
    """ Dataset [1] which contains pclouds sampled from the raw meshes, and
    sparse GT correspondences in terms of 14 keypoints.

    [1] J. Yang, S. Wuhrer et al. Estimation of Human Body Shape in Motion
    with Wide Clothing. ECCV 2016.
    """

    path_ds = '/cvlabdata1/cvlab/datasets_jan/inria_dressed_human/dataset_pts/pts_10k.npy'
    path_seq2inds = '/cvlabdata1/cvlab/datasets_jan/inria_dressed_human/dataset_pts/seq2inds.npz'
    path_regs = '/cvlabdata1/cvlab/datasets_jan/inria_dressed_human/dataset_pts/regs.npy'
    path_areas = '/cvlabdata1/cvlab/datasets_jan/inria_dressed_human/dataset_pts/areas.npy'
    path_faces = ''  # TODO

    subjects_all = ['s1', 's2', 's3', 's6']
    sequences_all = [
        'layered_knee', 'layered_spin', 'layered_walk', 'tight_walk',
        'wide_knee', 'wide_spin', 'wide_walk']

    def __init__(self, **kwargs):
        # Init. the parent class.
        super(DatasetINRIAPairs, self).__init__(
            path_pts=DatasetINRIAPairs.path_ds,
            path_regs=DatasetINRIAPairs.path_regs,
            path_seq2inds=DatasetINRIAPairs.path_seq2inds,
            path_areas=DatasetINRIAPairs.path_areas,
            path_faces=DatasetINRIAPairs.path_faces, **kwargs)

    def _load_faces(self, path):
        print('[ERROR]: Faces not implemented for INRIA, will not be loaded.')
        return None


class DatasetCMUPairs(DatasetPairsBase):
    """ Dataset [1] which contains only raw pclouds.

    [1] H. Joo et al. The Panoptic Studio: A Massively Multiview System
    for Social Motion Capture. ICCV 2015.
    """
    path_ds = '/cvlabdata1/cvlab/datasets_jan/cmu_panoptic/pts_raw/pts_10k.npy'
    path_seq2inds = '/cvlabdata1/cvlab/datasets_jan/cmu_panoptic/pts_raw/seq2id.npz'

    sequences_all = ['171026_pose3_waving_arms', '171026_pose3_low_lunges',
                     '171026_pose3_crossing_arms', '171026_pose3_arms_fingers']

    def __init__(self, **kwargs):
        # Check if areas are requested.
        if kwargs.get('with_area', False):
            print('[WARNING]: CMU ds does not contain meshes, the areas are '
                  'not available and will not be loaded.')
            kwargs['with_area'] = False

        super(DatasetCMUPairs, self).__init__(
            path_pts=DatasetCMUPairs.path_ds,
            path_seq2inds=DatasetCMUPairs.path_seq2inds, **kwargs)

    def _load_faces(self, path):
        return None


class DataLoaderDevicePairs(Device):
    def __init__(self, dl, gpu=True):
        super(DataLoaderDevicePairs, self).__init__(gpu=gpu)
        dl.collate_fn = DataLoaderDevicePairs.collat_fn
        self._dl = dl

    def __len__(self):
        return len(self._dl)

    def __iter__(self):
        batches = iter(self._dl)
        for batch in batches:
            yield {k: v.to(self.device) for k, v in batch.items()}

    @staticmethod
    def collat_fn(batch):
        bc = {k: [] for k in batch[0].keys()}
        for it in batch:
            for k, v in it.items():
                bc[k].append(v)
        return {k: torch.cat(v, dim=0) for k, v in bc.items()}


DatasetClasses = {
    'dfaust': DatasetDFAUSTPairs, 'ama': DatasetAMAPairs,
    'anim': DatasetAnimalsPairs, 'cape': DatasetCAPEPairs,
    'inria': DatasetINRIAPairs, 'cmu': DatasetCMUPairs
}
