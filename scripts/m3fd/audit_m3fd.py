#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from m3fd_utils import (
    CLASS_NAMES, collect_records, discover_layout, ensure_dir, read_source_split_stems,
    plot_basic_figures, sha256_file, visualize_pairs, write_class_counts, write_csv, write_text,
)


def main():
    ap = argparse.ArgumentParser(description='Audit the locally obtained M3FD raw data.')
    ap.add_argument('--root', default='datasets/M3FD', help='M3FD root directory')
    ap.add_argument('--raw', default=None, help='Raw download directory')
    ap.add_argument('--third-party', default='third_party/TarDAL', help='Official TarDAL clone')
    args = ap.parse_args()

    root = Path(args.root)
    raw = Path(args.raw) if args.raw else root / 'raw'
    reports = ensure_dir(root / 'reports')
    figs = ensure_dir(reports / 'figures')
    layout = discover_layout(raw)
    missing_layout = [
        name for name in ['visible_dir', 'ir_dir', 'labels_dir']
        if getattr(layout, name) is None
    ]
    if missing_layout:
        raise SystemExit(
            'M3FD raw data is incomplete; missing required directories: '
            + ', '.join(missing_layout)
            + '. Finish official download/unzip before audit.'
        )
    records, invalid = collect_records(layout)
    if not records:
        raise SystemExit('M3FD raw data contains no discoverable samples; abort audit.')

    image_rows = []
    for r in records:
        image_rows.append({
            'stem': r['stem'],
            'visible': r['visible'],
            'ir': r['ir'],
            'label': r['label'],
            'width': r['width'],
            'height': r['height'],
            'num_boxes': r['num_boxes'],
            'classes': ' '.join(map(str, r['classes'])),
            'issues': ';'.join(r['issues']),
        })
    write_csv(reports / 'image_pairs.csv', image_rows)
    write_csv(reports / 'invalid_samples.csv', invalid, ['stem', 'issues'])
    write_class_counts(reports / 'class_counts_full.csv', records)

    with (reports / 'file_checksums.sha256').open('w', encoding='utf-8') as f:
        for p in sorted(raw.rglob('*')):
            if p.is_file():
                f.write(f'{sha256_file(p)}  {p.relative_to(raw)}\n')

    source_train = read_source_split_stems(layout.train_txt)
    source_val = read_source_split_stems(layout.val_txt)
    train_set, val_set = set(source_train), set(source_val)
    bad_ratio = len(invalid) / max(len(records), 1)
    class_ids = sorted({b['class_id'] for r in records for b in r['labels']})
    category_confirmed = False
    troot = Path(args.third_party)
    evidence = []
    if troot.exists():
        for p in troot.rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.py', '.txt', '.md', '.yaml', '.yml'}:
                text = p.read_text(errors='ignore')
                if all(name.lower() in text.lower() for name in ['people', 'car', 'bus', 'motorcycle', 'lamp', 'truck']):
                    category_confirmed = True
                    evidence.append(str(p))
    status = {
        'layout': {k: str(getattr(layout, k)) for k in ['visible_dir', 'ir_dir', 'labels_dir', 'meta_dir', 'train_txt', 'val_txt', 'pred_txt']},
        'num_records': len(records),
        'num_invalid': len(invalid),
        'invalid_ratio': bad_ratio,
        'class_ids_seen': class_ids,
        'class_names_assumed': CLASS_NAMES,
        'category_mapping_confirmed_from_source_reference_files': category_confirmed,
        'category_evidence_files': evidence[:20],
        'discovered_source_train_list_count': len(source_train),
        'discovered_source_val_list_count': len(source_val),
        'discovered_source_train_val_intersection': len(train_set & val_set),
    }
    write_text(reports / 'data_audit.md', '# M3FD Data Audit\n\n```json\n' + json.dumps(status, indent=2) + '\n```\n')
    plot_basic_figures(records, figs, 'full_raw')
    visualize_pairs(records, figs / 'sample_pairs', n=30, seed=42)
    if bad_ratio > 0.005:
        raise SystemExit(f'Invalid/missing sample ratio {bad_ratio:.4%} exceeds 0.5%; stop before split construction.')


if __name__ == '__main__':
    main()
