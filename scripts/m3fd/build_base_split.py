#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from m3fd_utils import (
    collect_records, discover_layout, greedy_internal_val, make_manifest,
    write_split, write_text,
)


def main():
    ap = argparse.ArgumentParser(description='Build M3FD base train/val/test split.')
    ap.add_argument('--root', default='datasets/M3FD')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val-ratio', type=float, default=0.10)
    ap.add_argument('--out-dir', default=None, help='Output directory for base train/val/test lists and manifest.')
    args = ap.parse_args()
    root = Path(args.root)
    layout = discover_layout(root / 'raw')
    records, invalid = collect_records(layout)
    if len(invalid) / max(len(records), 1) > 0.005:
        raise SystemExit('Invalid sample ratio exceeds 0.5%; run audit and fix data first.')
    # M3FD provides no official train/validation/test lists in this dataset
    # snapshot. Any lists discovered below raw/ are not treated as authoritative.
    valid = sorted(r['stem'] for r in records if not r['issues'])
    trainval, test = greedy_internal_val(valid, records, seed=args.seed + 1000, ratio=0.10)
    train, val = greedy_internal_val(trainval, records, seed=args.seed, ratio=args.val_ratio / 0.90)
    split_source = 'project_generated_fixed_seed_image_level_split'
    source_checksums = {}
    source_train_count = 0
    source_val_count = 0
    out = Path(args.out_dir) if args.out_dir else root / 'splits' / 'base'
    write_split(out / 'train.txt', train)
    write_split(out / 'val.txt', val)
    write_split(out / 'test.txt', test)
    extra = {
        'source_train_txt_found_but_not_used': str(layout.train_txt) if layout.train_txt else None,
        'source_val_txt_found_but_not_used': str(layout.val_txt) if layout.val_txt else None,
        'source_train_count_used': source_train_count,
        'source_val_count_used': source_val_count,
        'split_source': split_source,
        'internal_val_algorithm': 'deterministic greedy image-level multilabel stratification',
        'seed': args.seed,
        'test_seed': args.seed + 1000,
        'source_checksums': source_checksums,
        'val_ratio_from_base_train': args.val_ratio,
        'protocol_note': ('Project-generated fixed greedy multilabel-stratified base '
                          'split; not an official M3FD split. Base test has 420 pairs '
                          'before the final IR duplicate repair.'),
    }
    manifest = make_manifest(records, {'train': train, 'val': val, 'test': test}, extra)
    write_text(out / 'split_manifest.json', json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
