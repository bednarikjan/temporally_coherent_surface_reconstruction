### Computes the metrics related to the task of dense keypoint transfer and
# stores them in .yaml amnd .csv files for all sequences within a dataset
#
# - CD
#   Chamfer Distance of the reconstructed shapes. This is not immediately
#   relevant for the task of kpts transfer, but it is a good metric to
#   see how precise the reconstructions are.
#
# - mean L2 distance
#   L2 distance between the predicted and GT kpt averaged over the number
#   of kpts.
#
# - mean rank
#   Rank of every predicted kpt (i.e. how many over predicted pts are closer to
#   the GT kpt tha the predicted kpt) averaged over the number of kpts.
#
# - PCK
# - PCK area under curve
#

# Pythod std.
import os
import csv
import argparse
from collections import defaultdict

# Project files.
import externals.jblib.file_sys as jbfs
from tcsr.evaluate.eval import eval_trrun
from tcsr.data.data_loader import DatasetClasses

# 3rd party
import yaml


# Arguments
argparser = argparse.ArgumentParser()
argparser.add_argument(
    'path_root', type=str,
    help='Path to the root of all experiments for the given DS.')
argparser.add_argument(
    '--ds', type=str, choices=['dfaust', 'ama', 'anim', 'inria', 'cape'],
    help='Dataset choice.')
argparser.add_argument(
    '--include_seqs', type=str, default=[], nargs='+',
    help='Only include these sequences. If the list is empty, all the sequences'
         ' are included')
argparser.add_argument(
    '--exclude_seqs', type=str, default=[], nargs='+',
    help='Exclude these sequences. If the list is empty, no sequences are '
         'excluded.')
argparser.add_argument(
    '--include_trruns', type=str, default=[], nargs='+',
    help='Only include these trruns. If the list is empty, all the trruns'
         ' are included')
argparser.add_argument(
    '--exclude_trruns', type=str, default=[], nargs='+',
    help='Exclude these trruns. If the list is empty, no trruns are excluded.')
argparser.add_argument(
    '--subjects', nargs='+', type=list, default=[],
    help='Subjects ids selection.')
argparser.add_argument(
    '--n_iters', type=int, default=500, help='Number of sampled random pairs.')
argparser.add_argument(
    '--n_pts', type=int, default=3125, help='Number of predicted points.')
argparser.add_argument(
    '--pck_steps', type=int, default=100, help='Number of steps for PCK.')
argparser.add_argument(
    '--pck_min', type=float, default=0., help='PCK lower threshold.')
argparser.add_argument(
    '--pck_max', type=float, default=0.02, help='PCK upper threshold.')
argparser.add_argument(
    '--uv_pts_mode', choices=[
        'grid_floor', 'grid_ceil', 'random_floor', 'random_ceil',
        'regular_floor', 'regular_ceil'], type=str, default='regular_floor',
    help='Strategy to generate 2D pts.')
argparser.add_argument(
    '--verbose', type=bool, default=False, help='Verbosity level.')
args = argparser.parse_args()

kwargs = {'subjects': args.subjects}

# Get the sequences to use.
all_seqs = DatasetClasses[args.ds].sequences_all

sequences = args.include_seqs
sequences = [s for s in all_seqs if s not in args.exclude_seqs] \
    if len(sequences) == 0 else sequences
assert all([s in all_seqs for s in sequences])

# Get only those of the selected sequecnes for which the trrun dir exists.
sequences = sorted(list(
    set(jbfs.lsd(args.path_root)).intersection(set(sequences))))

name_out = f"results_it{args.n_iters}_pts{args.n_pts}_" \
           f"pck{args.pck_min:.3f}-{args.pck_max:.3f}_" \
           f"uvmode-{args.uv_pts_mode}"

# Prepare output CSV.
seqs_str = 'seqs-ALL'
if len(sequences) != len(all_seqs):
    seqs_str = 'seqs-' + '_'.join(sequences)
path_out_csv = jbfs.jn(
    args.path_root, name_out + seqs_str + '.csv')
if os.path.exists(path_out_csv):
    path_out_csv_uniq = jbfs.unique_file_name(path_out_csv)
    print(f"[WARNING]: Output file {path_out_csv} already exists, saving in "
          f"{path_out_csv_uniq} instead.")
    path_out_csv = path_out_csv_uniq
path_out_csv = jbfs.unique_file_name(path_out_csv)
with open(path_out_csv, 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(
        ['sequence', 'trrun', 'sl2 (x1e3)', 'rank', 'auc', 'cd (x1e3)'])

# Process all sequences.
defdict = lambda: defaultdict(defdict)
res_all = defdict()
for seqi, seq in enumerate(sequences):
    path_seq = jbfs.jn(args.path_root, seq)

    # Process all trruns.
    trruns = jbfs.lsd(path_seq)
    trruns = sorted(list(set(trruns).difference(set(args.exclude_trruns)))) \
        if len(args.include_trruns) == 0 else sorted(args.include_trruns)
    for tri, trrun in enumerate(trruns):
        path_trrun = jbfs.jn(path_seq, trrun)

        print(f"\rEvaluating seq {seqi + 1}/{len(sequences)} ({seq}), "
              f"trrun {tri + 1}/{len(trruns)}", end='')

        # Eval.
        res = eval_trrun(
            args.ds, path_trrun, args.n_iters, args.n_pts,
            pck_rng=(args.pck_min, args.pck_max), pck_steps=args.pck_steps,
            uv_pts_mode=args.uv_pts_mode, verbose=args.verbose, **kwargs)
        res_all[seq][trrun] = res

        # Store results.
        path_res_dir = jbfs.jn(path_trrun, 'res/dense_corresp')
        path_res_file = jbfs.jn(path_res_dir, name_out + '.yaml')
        jbfs.make_dir(path_res_dir)
        with open(path_res_file, 'w') as f:
            res['pck_rng'] = [args.pck_min, args.pck_max]
            res['uv_pts_mode'] = args.uv_pts_mode
            yaml.dump(res, f)

        # Save line to output CSV.
        with open(path_out_csv, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(
                [seq, trrun,
                 f"{res['sl2_mu'] * 1000.:.2f}+-"
                 f"{res['sl2_std'] * 1000.:.2f}",
                 f"{res['rank_mu'] * 100.:.2f}+-"
                 f"{res['rank_std'] * 100.:.2f}",
                 f"{res['auc_mu'] * 100.:.2f}+-"
                 f"{res['auc_std'] * 100.:.2f}",
                 f"{res['cd_mu'] * 1000.:.3f}+-"
                 f"{res['cd_std'] * 1000.:.3f}"])
