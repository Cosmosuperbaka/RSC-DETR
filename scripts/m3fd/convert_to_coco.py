#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from m3fd_utils import collect_records, convert_split_to_coco, discover_layout, ensure_dir, make_manifest, read_lines, write_text


def main():
    ap = argparse.ArgumentParser(description='Convert M3FD YOLO split to paired COCO HBB JSON.')
    ap.add_argument('--root', default='datasets/M3FD')
    ap.add_argument('--split-dir', default='datasets/M3FD/splits/base')
    ap.add_argument('--out-dir', default='datasets/M3FD/processed/full')
    args = ap.parse_args()
    root = Path(args.root)
    records, invalid = collect_records(discover_layout(root / 'raw'))
    if invalid:
        print(f'warning: {len(invalid)} invalid records are excluded from COCO conversion')
    split_dir = Path(args.split_dir)
    out = Path(args.out_dir)
    visible = ensure_dir(out / 'visible')
    infrared = ensure_dir(out / 'infrared')
    ann = ensure_dir(out / 'annotations')
    manifests = ensure_dir(out / 'manifests')
    split_stems = {}
    for split in ['train', 'val', 'test']:
        stems = read_lines(split_dir / f'{split}.txt')
        split_stems[split] = stems
        convert_split_to_coco(records, stems, ann / f'instances_{split}.json', visible, infrared)
    manifest = make_manifest(records, split_stems, {'split_dir': str(split_dir), 'format': 'paired_coco_hbb'})
    write_text(manifests / 'manifest.json', json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
