### This script contains custom made batch indices samplers. They can be used
# e.g. for training where the dataset is gradually grown.
#

# Python std.
import math
import copy

# 3rd party
import numpy as np


class BatchSamplerGradualGrow:
    def_window_growth = {
        'type': 'linear', 'it_start': 30000, 'it_end': 170000, 'step': 4}
    def_sampling = {
        'type': 'uniform', 'ignore_hist_before_start': False,
        'max_hist_val_mult': 1.0}

    def __init__(
            self, bs, N, start='center', window_init=5, window_growth=None,
            unit_abs=True, sampling=None):
        # Save params.
        self._bs = bs
        self._N = N
        self._it = 0
        # Legacy support for sampling parameter as a string.
        if isinstance(sampling, str):
            st = sampling
            sampling = copy.copy(BatchSamplerGradualGrow.def_sampling)
            sampling['type'] = st
        self._sampling = BatchSamplerGradualGrow.def_sampling \
            if sampling is None else sampling
        assert 'type' in self._sampling and \
               self._sampling['type'] in ('random', 'uniform')

        # Save extra sampling based params.
        if self._sampling['type'] == 'uniform':
            self._ign_hist_st = self._sampling['ignore_hist_before_start']
            self._max_hist_mult = self._sampling['max_hist_val_mult']
            assert self._max_hist_mult >= 1.

        # Get initial window.
        self._from, self._to = self._parse_window_init(
            start, window_init, unit_abs, N)
        self._center_init = \
            self._from + (self._to - self._from) // 2

        # Get window growth type.
        window_growth = window_growth if window_growth is not None \
            else BatchSamplerGradualGrow.def_window_growth
        self._growth_iters, self._growth_steps = \
            self._parse_window_growth(
                window_growth, unit_abs, N, window_init)
        self._gi = 0

        # Sampling properties.
        self._hist = np.zeros((N,), dtype=np.int32)

    def _parse_window_init(self, start, window_init, unit_abs, N):
        assert start in ('center', 'left', 'right')
        width = window_init if unit_abs \
            else math.round(window_init * 0.01 * N)
        assert width <= N

        if start == 'center':
            fr = N // 2 - width // 2
            to = fr + width
        elif start == 'left':
            fr = 0
            to = width
        elif start == 'right':
            to = N
            fr = to - width
        else:
            raise Exception(f"Unknown start '{start}'")
        return fr, to

    def _parse_window_growth(self, conf, unit_abs, N, window_init):
        assert conf['type'] in ('linear', 'custom')
        if conf['type'] == 'linear':
            step = conf['step'] if unit_abs else \
                math.round(conf['step'] * 0.01 * N)
            window_init = window_init if unit_abs \
                else math.round(window_init * 0.01 * N)

            num_steps = math.ceil((N - window_init) / step)
            iters = np.linspace(
                conf['it_start'], conf['it_end'],
                num_steps, dtype=np.int32)
            steps = np.ones_like(iters) * step
        elif conf['type'] == 'custom':
            # TODO
            iters = None
            steps = None
            raise NotImplementedError

        assert iters.shape == steps.shape
        return iters, steps

    def _grow_window(self):
        next_it = self._growth_iters[self._gi]
        next_step = self._growth_steps[self._gi]

        # Grow window.
        if next_it == self._it:
            half_small = next_step // 2
            half_big = next_step - half_small

            left_smaller = \
                self._center_init - self._from <= \
                self._to - self._center_init
            left_inc, right_inc = (
                (half_small, half_big), (half_big, half_small)
            )[left_smaller]

            fr_new = max(0, self._from - left_inc)
            to_new = min(self._N, self._to + right_inc)
            left_res = left_inc - (self._from - fr_new)
            right_res = right_inc - (to_new - self._to)
            res = left_res + right_res
            fr_new = max(0, fr_new - res)
            to_new = min(self._N, to_new + res)

            self._from = fr_new
            self._to = to_new
            self._gi = min(
                self._growth_iters.shape[0] - 1, self._gi + 1)

    def _sample(self):
        if self._sampling['type'] == 'random':
            inds = np.random.randint(
                self._from, self._to, (self._bs,))
        elif self._sampling['type'] == 'uniform':
            prob = (np.max(self._hist) * self._max_hist_mult - self._hist)[
                   self._from:self._to]
            prob = prob if np.sum(prob) > 0 \
                else np.ones_like(prob)
            cdf = np.cumsum(prob / np.sum(prob))

            smpls = np.random.uniform(0., 1., (self._bs,))
            inds_prob = np.sum(smpls[:, None] >= cdf, axis=1)
            inds = np.arange(self._from, self._to)[inds_prob]

        # Update the histogram.
        if not self._ign_hist_st or self._gi > 0:
            i, c = np.unique(inds, return_counts=True)
            self._hist[i] = self._hist[i] + c

        return inds

    def __iter__(self):
        while True:
            self._it += 1
            self._grow_window()
            yield self._sample()

    def __len__(self):
        return 1

    def state_dict(self):
        """ Returns the state of the btahc sampler as a :class:`dict`.

        Returns:
            dict: A dict holding the current batch sampler state.
        """
        return {'it': self._it, 'gi': self._gi, 'hist': self._hist,
                'from': self._from, 'to': self._to}

    def load_state_dict(self, state_dict):
        """ Loads the batch sampler state.

        Args:
            state_dict (dict): Batch sampler state. Should be an object
                returned from a call to :meth:`state_dict`.
        """
        # Check the state dict format.
        for k in ['it', 'gi', 'hist', 'from', 'to']:
            if not k in state_dict:
                raise Exception(f"Missing key '{k}' while loading the batch "
                                f"sampler state.")

        # Load the state.
        self._it = state_dict['it']
        self._gi = state_dict['gi']
        self._hist = state_dict['hist']
        self._from = state_dict['from']
        self._to = state_dict['to']
