#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from m3fd_utils import (
    class_stats, collect_records, discover_layout, ensure_dir, lt20_targets, make_manifest,
    read_lines, write_class_counts, write_csv, write_split, write_text, build_lt_subset,
)


def write_image_class_counts(path: Path, records, stems):
    rec = {r['stem']: r for r in records}
    rows = []
    for s in stems:
        if s not in rec:
            continue
        counts = {f'class_{i}': 0 for i in range(6)}
        for b in rec[s]['labels']:
            if b['class_id'] in range(6):
                counts[f'class_{b["class_id"]}'] += 1
        rows.append({'stem': s, **counts})
    write_csv(path, rows)


def main():
    ap = argparse.ArgumentParser(description='Build image-level M3FD-LT20 training splits.')
    ap.add_argument('--root', default='datasets/M3FD')
    ap.add_argument('--base-dir', default='datasets/M3FD/splits/base')
    ap.add_argument('--seeds', default='42,1234,3407')
    ap.add_argument('--out-dir', default=None, help='Output root for per-seed split directories.')
    args = ap.parse_args()
    root = Path(args.root)
    records, invalid = collect_records(discover_layout(root / 'raw'))
    base = Path(args.base_dir)
    base_train = read_lines(base / 'train.txt')
    base_val = read_lines(base / 'val.txt')
    base_test = read_lines(base / 'test.txt')
    order, target, full_inst, full_img = lt20_targets(base_train, records)

    for seed in [int(x) for x in args.seeds.split(',') if x.strip()]:
        train = build_lt_subset(base_train, records, seed=seed)
        out_root = Path(args.out_dir) if args.out_dir else root / 'splits'
        out = out_root / f'lt20_seed{seed}'
        ensure_dir(out)
        write_split(out / 'train.txt', train)
        write_split(out / 'val.txt', base_val)
        write_split(out / 'test.txt', base_test)
        write_class_counts(out / 'class_counts.csv', records, train)
        write_image_class_counts(out / 'image_class_counts.csv', records, train)
        inst, img = class_stats(records, train)
        nonzero = [inst[c] for c in inst if inst[c] > 0]
        actual_if = max(nonzero) / min(nonzero) if nonzero else 0.0
        checks = {
            'min_50_instances_per_class': all(inst[c] >= 50 for c in range(6)),
            'min_20_images_per_class': all(img[c] >= 20 for c in range(6)),
            'actual_imbalance_factor': actual_if,
            'actual_if_at_least_10': actual_if >= 10.0,
            'val_identical_to_base': base_val == read_lines(out / 'val.txt'),
            'test_identical_to_base': base_test == read_lines(out / 'test.txt'),
            'no_cross_split_overlap': not (set(train) & set(base_val) or set(train) & set(base_test) or set(base_val) & set(base_test)),
        }
        extra = {
            'protocol': 'M3FD-LT20',
            'seed': seed,
            'class_frequency_order_desc': order,
            'full_train_instances': {str(c): full_inst[c] for c in range(6)},
            'theoretical_targets': {str(c): target[c] for c in range(6)},
            'actual_imbalance_factor': actual_if,
            'checks': checks,
            'note': 'Image-level selection only; all labels in selected images are kept.',
        }
        manifest = make_manifest(records, {'train': train, 'val': base_val, 'test': base_test}, extra)
        write_text(out / 'split_manifest.json', json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
