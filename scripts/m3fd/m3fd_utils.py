#!/usr/bin/env python3
"""Shared utilities for the M3FD data protocol scripts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


CLASS_NAMES = {
    0: 'People',
    1: 'Car',
    2: 'Bus',
    3: 'Lamp',
    4: 'Motorcycle',
    5: 'Truck',
}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
VISIBLE_DIR_NAMES = {'vi', 'vis', 'visible', 'rgb', 'color'}
IR_DIR_NAMES = {'ir', 'infrared', 'thermal'}


@dataclass
class M3FDLayout:
    root: Path
    visible_dir: Optional[Path]
    ir_dir: Optional[Path]
    labels_dir: Optional[Path]
    meta_dir: Optional[Path]
    train_txt: Optional[Path]
    val_txt: Optional[Path]
    pred_txt: Optional[Path]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding='utf-8')


def read_lines(path: Optional[Path]) -> List[str]:
    if path is None or not path.exists():
        return []
    out = []
    for line in path.read_text(errors='ignore').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            out.append(line)
    return out


def normalize_stem(value: str) -> str:
    value = value.strip().replace('\\', '/')
    name = Path(value).name
    stem = Path(name).stem
    for suffix in ['_rgb', '_vi', '_vis', '_visible', '_ir', '_infrared', '_thermal']:
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def list_images(path: Path) -> List[Path]:
    if path is None or not path.exists():
        return []
    return sorted(p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def discover_layout(raw_root: Path) -> M3FDLayout:
    candidates = [p for p in raw_root.rglob('*') if p.is_dir()]
    visible = None
    infrared = None
    labels = None
    meta = None
    for d in candidates:
        name = d.name.lower()
        if visible is None and name in VISIBLE_DIR_NAMES and list_images(d):
            visible = d
        if infrared is None and name in IR_DIR_NAMES and list_images(d):
            infrared = d
        if labels is None and name == 'labels' and list(d.glob('*.txt')):
            labels = d
        if meta is None and name == 'meta':
            meta = d

    txts = {p.name.lower(): p for p in raw_root.rglob('*.txt')}
    train_txt = txts.get('train.txt')
    val_txt = txts.get('val.txt')
    pred_txt = txts.get('pred.txt')
    if meta is not None:
        train_txt = train_txt or (meta / 'train.txt' if (meta / 'train.txt').exists() else None)
        val_txt = val_txt or (meta / 'val.txt' if (meta / 'val.txt').exists() else None)
        pred_txt = pred_txt or (meta / 'pred.txt' if (meta / 'pred.txt').exists() else None)

    return M3FDLayout(raw_root, visible, infrared, labels, meta, train_txt, val_txt, pred_txt)


def load_pairs(layout: M3FDLayout) -> Dict[str, Dict[str, Path]]:
    visible = {normalize_stem(p.name): p for p in list_images(layout.visible_dir)} if layout.visible_dir else {}
    infrared = {normalize_stem(p.name): p for p in list_images(layout.ir_dir)} if layout.ir_dir else {}
    labels = {normalize_stem(p.name): p for p in layout.labels_dir.glob('*.txt')} if layout.labels_dir else {}
    stems = sorted(set(visible) | set(infrared) | set(labels))
    return {s: {'visible': visible.get(s), 'ir': infrared.get(s), 'label': labels.get(s)} for s in stems}


def parse_yolo_label(path: Optional[Path]) -> Tuple[List[Dict], List[str]]:
    rows = []
    errors = []
    if path is None or not path.exists():
        return rows, ['missing_label']
    text = path.read_text(errors='ignore').strip()
    if not text:
        return rows, ['empty_label']
    for idx, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            errors.append(f'line_{idx}:expected_5_fields')
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f'line_{idx}:non_numeric')
            continue
        if cls not in CLASS_NAMES:
            errors.append(f'line_{idx}:bad_class_{cls}')
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            errors.append(f'line_{idx}:coord_out_of_range')
        if w <= 0 or h <= 0:
            errors.append(f'line_{idx}:non_positive_wh')
        rows.append({'class_id': cls, 'x': x, 'y': y, 'w': w, 'h': h, 'line': idx})
    return rows, errors


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        return im.size


def read_source_split_stems(path: Optional[Path]) -> List[str]:
    return [normalize_stem(x) for x in read_lines(path)]


def collect_records(layout: M3FDLayout) -> Tuple[List[Dict], List[Dict]]:
    pairs = load_pairs(layout)
    records = []
    invalid = []
    seen_files = Counter()
    for stem, paths in pairs.items():
        issues = []
        for k in ['visible', 'ir', 'label']:
            if paths.get(k) is None:
                issues.append(f'missing_{k}')
        vis_size = ir_size = None
        if paths.get('visible'):
            seen_files[str(paths['visible'].resolve())] += 1
            try:
                vis_size = image_size(paths['visible'])
            except Exception as e:
                issues.append(f'bad_visible:{type(e).__name__}')
        if paths.get('ir'):
            seen_files[str(paths['ir'].resolve())] += 1
            try:
                ir_size = image_size(paths['ir'])
            except Exception as e:
                issues.append(f'bad_ir:{type(e).__name__}')
        if vis_size and ir_size and vis_size != ir_size:
            issues.append('size_mismatch')
        labels, label_errors = parse_yolo_label(paths.get('label'))
        issues.extend(label_errors)
        if paths.get('label'):
            seen_files[str(paths['label'].resolve())] += 1
        record = {
            'stem': stem,
            'visible': str(paths['visible']) if paths.get('visible') else '',
            'ir': str(paths['ir']) if paths.get('ir') else '',
            'label': str(paths['label']) if paths.get('label') else '',
            'width': vis_size[0] if vis_size else '',
            'height': vis_size[1] if vis_size else '',
            'num_boxes': len(labels),
            'classes': sorted({r['class_id'] for r in labels if r['class_id'] in CLASS_NAMES}),
            'labels': labels,
            'issues': issues,
        }
        records.append(record)
        if issues:
            invalid.append({'stem': stem, 'issues': ';'.join(issues)})

    for file_name, count in seen_files.items():
        if count > 1:
            invalid.append({'stem': Path(file_name).stem, 'issues': f'duplicate_file_ref:{count}'})
    return records, invalid


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in fieldnames})


def class_stats(records: Sequence[Dict], stems: Optional[Iterable[str]] = None) -> Tuple[Counter, Counter]:
    use = set(stems) if stems is not None else None
    inst = Counter()
    img = Counter()
    for r in records:
        if use is not None and r['stem'] not in use:
            continue
        present = set()
        for box in r['labels']:
            c = box['class_id']
            if c in CLASS_NAMES:
                inst[c] += 1
                present.add(c)
        for c in present:
            img[c] += 1
    return inst, img


def write_class_counts(path: Path, records: Sequence[Dict], stems: Optional[Iterable[str]] = None) -> None:
    inst, img = class_stats(records, stems)
    rows = []
    for cid, name in CLASS_NAMES.items():
        rows.append({'class_id': cid, 'class_name': name, 'instances': inst[cid], 'images': img[cid]})
    write_csv(path, rows, ['class_id', 'class_name', 'instances', 'images'])


def safe_link(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        return
    rel = os.path.relpath(src.resolve(), dst.parent.resolve())
    try:
        dst.symlink_to(rel)
    except OSError:
        shutil.copy2(src, dst)


def split_record_map(records: Sequence[Dict]) -> Dict[str, Dict]:
    return {r['stem']: r for r in records}


def write_split(path: Path, stems: Sequence[str]) -> None:
    ensure_dir(path.parent)
    path.write_text('\n'.join(stems) + ('\n' if stems else ''), encoding='utf-8')


def greedy_internal_val(train_stems: Sequence[str], records: Sequence[Dict], seed: int = 42, ratio: float = 0.1) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    rec = split_record_map(records)
    stems = [s for s in train_stems if s in rec and not rec[s]['issues']]
    target_n = max(1, round(len(stems) * ratio))
    total_img = Counter()
    for s in stems:
        for c in set(rec[s]['classes']):
            total_img[c] += 1
    target = {c: total_img[c] * ratio for c in CLASS_NAMES}
    candidates = stems[:]
    rng.shuffle(candidates)
    selected = []
    selected_counts = Counter()
    while len(selected) < target_n and candidates:
        best_i = 0
        best_score = None
        for i, s in enumerate(candidates[: min(len(candidates), 2000)]):
            trial = selected_counts.copy()
            for c in set(rec[s]['classes']):
                trial[c] += 1
            score = sum((trial[c] - target[c]) ** 2 for c in CLASS_NAMES)
            score += 0.01 * abs((len(selected) + 1) - target_n)
            if best_score is None or score < best_score:
                best_score = score
                best_i = i
        s = candidates.pop(best_i)
        selected.append(s)
        for c in set(rec[s]['classes']):
            selected_counts[c] += 1
    val = sorted(selected)
    train = sorted(s for s in stems if s not in set(val))
    return train, val


def yolo_to_xyxy(box: Dict, width: int, height: int) -> Tuple[float, float, float, float]:
    x, y, w, h = box['x'], box['y'], box['w'], box['h']
    bw = w * width
    bh = h * height
    cx = x * width
    cy = y * height
    x0 = max(0.0, cx - bw / 2)
    y0 = max(0.0, cy - bh / 2)
    x1 = min(float(width), cx + bw / 2)
    y1 = min(float(height), cy + bh / 2)
    return x0, y0, x1, y1


def convert_split_to_coco(records: Sequence[Dict], stems: Sequence[str], out_json: Path, visible_out: Path, ir_out: Path) -> Dict:
    rec = split_record_map(records)
    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    for stem in sorted(stems):
        r = rec[stem]
        if r['issues']:
            continue
        vis = Path(r['visible'])
        ir = Path(r['ir'])
        ext = vis.suffix.lower() or '.jpg'
        file_name = f'{stem}{ext}'
        ir_file_name = f'{stem}{ir.suffix.lower() or ext}'
        safe_link(vis, visible_out / file_name)
        safe_link(ir, ir_out / ir_file_name)
        width, height = int(r['width']), int(r['height'])
        images.append({
            'id': img_id,
            'file_name': file_name,
            'ir_file_name': ir_file_name,
            'width': width,
            'height': height,
            'stem': stem,
        })
        for box in r['labels']:
            if box['class_id'] not in CLASS_NAMES:
                continue
            x0, y0, x1, y1 = yolo_to_xyxy(box, width, height)
            bw = x1 - x0
            bh = y1 - y0
            if bw <= 0 or bh <= 0:
                continue
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': int(box['class_id']),
                'bbox': [x0, y0, bw, bh],
                'area': bw * bh,
                'iscrowd': 0,
            })
            ann_id += 1
        img_id += 1
    coco = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': cid, 'name': name} for cid, name in CLASS_NAMES.items()],
    }
    ensure_dir(out_json.parent)
    out_json.write_text(json.dumps(coco, indent=2), encoding='utf-8')
    return coco


def make_manifest(records: Sequence[Dict], split_stems: Dict[str, Sequence[str]], extra: Optional[Dict] = None) -> Dict:
    manifest = {'splits': {}, 'classes': CLASS_NAMES, 'extra': extra or {}}
    for split, stems in split_stems.items():
        inst, img = class_stats(records, stems)
        manifest['splits'][split] = {
            'num_images': len(stems),
            'instance_counts': {str(c): inst[c] for c in CLASS_NAMES},
            'image_counts': {str(c): img[c] for c in CLASS_NAMES},
            'sha256': hashlib.sha256(('\n'.join(stems) + '\n').encode()).hexdigest(),
        }
    return manifest


def lt20_targets(train_stems: Sequence[str], records: Sequence[Dict]) -> Tuple[List[int], Dict[int, float], Counter, Counter]:
    inst, img = class_stats(records, train_stems)
    order = sorted(CLASS_NAMES, key=lambda c: (-inst[c], c))
    positive = [inst[c] for c in CLASS_NAMES if inst[c] > 0]
    n_max = max(positive) if positive else 0
    if positive:
        # If the full training set is already more imbalanced than IF=20, the
        # textbook n_max curve asks for tail instances that do not exist. Cap
        # the head target so the image-level subset can approach IF=20 without
        # inventing annotations or deleting individual boxes.
        n_max = min(n_max, min(positive) * 20)
    target = {}
    cnum = len(CLASS_NAMES)
    for rank, cid in enumerate(order):
        target[cid] = n_max * ((1.0 / 20.0) ** (rank / max(cnum - 1, 1)))
    return order, target, inst, img


def build_lt_subset(train_stems: Sequence[str], records: Sequence[Dict], seed: int) -> List[str]:
    rec = split_record_map(records)
    valid = [s for s in train_stems if s in rec and not rec[s]['issues']]
    order, target, full_inst, full_img = lt20_targets(valid, records)
    rng = random.Random(seed)
    shuffled = valid[:]
    rng.shuffle(shuffled)
    selected = set()
    counts = Counter()
    img_counts = Counter()

    # First guarantee minimum coverage when feasible.
    for cid in order:
        candidates = [s for s in shuffled if cid in rec[s]['classes']]
        for s in candidates:
            if counts[cid] >= 50 and img_counts[cid] >= 20:
                break
            if s not in selected:
                selected.add(s)
                for b in rec[s]['labels']:
                    if b['class_id'] in CLASS_NAMES:
                        counts[b['class_id']] += 1
                for c in set(rec[s]['classes']):
                    img_counts[c] += 1

    def objective(next_counts: Counter, next_img: Counter, n_images: int) -> float:
        loss = 0.0
        for cid in CLASS_NAMES:
            loss += abs(math.log((next_counts[cid] + 1) / (target[cid] + 1)))
            loss += 0.15 * abs(math.log((next_img[cid] + 1) / (max(20, full_img[cid] * 0.35) + 1)))
        loss += 0.001 * max(0, len(valid) * 0.15 - n_images)
        return loss

    improved = True
    while improved:
        improved = False
        base = objective(counts, img_counts, len(selected))
        best = None
        for s in shuffled:
            if s in selected:
                continue
            trial_counts = counts.copy()
            trial_img = img_counts.copy()
            for b in rec[s]['labels']:
                if b['class_id'] in CLASS_NAMES:
                    trial_counts[b['class_id']] += 1
            for c in set(rec[s]['classes']):
                trial_img[c] += 1
            # Avoid moving head classes too far beyond target unless it helps tail coverage.
            score = objective(trial_counts, trial_img, len(selected) + 1)
            if best is None or score < best[0]:
                best = (score, s, trial_counts, trial_img)
        if best and best[0] < base - 1e-6:
            _, s, counts, img_counts = best
            selected.add(s)
            improved = True
    return sorted(selected)


def plot_basic_figures(records: Sequence[Dict], out_dir: Path, split_name: str = 'full') -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    inst, img = class_stats(records)
    labels = [CLASS_NAMES[c] for c in CLASS_NAMES]
    xs = np.arange(len(labels))

    plt.figure(figsize=(8, 4))
    plt.bar(xs, [inst[c] for c in CLASS_NAMES])
    plt.xticks(xs, labels, rotation=30, ha='right')
    plt.title(f'{split_name} instances per class')
    plt.tight_layout()
    plt.savefig(out_dir / f'{split_name}_instances_per_class.png')
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(xs, [img[c] for c in CLASS_NAMES])
    plt.xticks(xs, labels, rotation=30, ha='right')
    plt.title(f'{split_name} images per class')
    plt.tight_layout()
    plt.savefig(out_dir / f'{split_name}_images_per_class.png')
    plt.close()

    areas = defaultdict(list)
    ratios = defaultdict(list)
    for r in records:
        if r['issues']:
            continue
        width, height = int(r['width']), int(r['height'])
        for b in r['labels']:
            if b['class_id'] not in CLASS_NAMES:
                continue
            areas[b['class_id']].append(b['w'] * width * b['h'] * height)
            ratios[b['class_id']].append((b['w'] * width) / max(b['h'] * height, 1e-6))
    for data, name, xlabel in [(areas, 'area_distribution', 'box area'), (ratios, 'aspect_ratio_distribution', 'width / height')]:
        plt.figure(figsize=(9, 5))
        values = [data[c] if data[c] else [0] for c in CLASS_NAMES]
        plt.boxplot(values, tick_labels=labels, showfliers=False)
        plt.xticks(rotation=30, ha='right')
        plt.ylabel(xlabel)
        plt.title(f'{split_name} {name}')
        plt.tight_layout()
        plt.savefig(out_dir / f'{split_name}_{name}.png')
        plt.close()


def visualize_pairs(records: Sequence[Dict], out_dir: Path, n: int = 30, seed: int = 42) -> None:
    ensure_dir(out_dir)
    rng = random.Random(seed)
    valid = [r for r in records if not r['issues']]
    rng.shuffle(valid)
    for r in valid[:n]:
        vis = Image.open(r['visible']).convert('RGB')
        ir = Image.open(r['ir']).convert('RGB')
        draw = ImageDraw.Draw(vis)
        width, height = vis.size
        for b in r['labels']:
            if b['class_id'] not in CLASS_NAMES:
                continue
            x0, y0, x1, y1 = yolo_to_xyxy(b, width, height)
            draw.rectangle([x0, y0, x1, y1], outline='red', width=2)
            draw.text((x0, y0), CLASS_NAMES[b['class_id']], fill='yellow')
        canvas = Image.new('RGB', (width * 2, height))
        canvas.paste(vis, (0, 0))
        canvas.paste(ir, (width, 0))
        canvas.thumbnail((1200, 600))
        canvas.save(out_dir / f'pair_{r["stem"]}.jpg', quality=90)
