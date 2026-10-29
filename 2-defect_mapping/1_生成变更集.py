#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_生成变更集.py — 把人工清洗过的子图标签与裁切记录比对，生成待写回原图总库的变更集

本脚本是 **纯只读** 的：它只读子图 XML、crop_mapping.json、原图 XML 和历史台账，
把结论写到 --output 指定的输出目录，绝不修改原图总库的任何文件。

产出：
  changeset.json      —— 可以自动执行的变更（交给 2_应用变更集.py）
  review_queue.json   —— 需要人工看图确认的条目（交给 4_审核可视化.py）
  report.txt          —— 统计报告

用法示例：
  python 1_生成变更集.py \
    --dataset-dir /raid/wtj/甘肃新增训练集/gd_data \
    --component-type gd \
    --original-xml /raid/wtj/甘肃新增训练集/酒泉_天水_庆阳_映射原图xml \
    --output /raid/wtj/甘肃新增训练集/gd_data/映射原图
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mapping_core as mc
from mapping_core import BBox

CHANGESET_VERSION = 1

# ---- 变更类型 ----
ACT_SET_NAME = 'set_name'
ACT_SET_BBOX = 'set_bbox'
ACT_SET_NAME_BBOX = 'set_name_bbox'
ACT_DELETE = 'delete'
ACT_ADD = 'add'
ACT_KEEP = 'keep'  # 仅参与一致性校验，不产生任何写入

# ---- 待人工确认的原因码 ----
REASONS = {
    'clipped_deleted':
        '被裁切的缺陷在子图里被删除；原图框有一部分在部件框之外，不能直接删',
    'clipped_bbox_modified_border':
        '被裁切的缺陷改了框，但新框仍贴着裁切边界，说明原图里可能仍被截断',
    'target_not_found':
        '按裁切记录的类名+坐标在原图里找不到对应的框（原图可能已被改动）',
    'target_ambiguous':
        '原图里存在多个类名和坐标完全相同的重复框，无法确定改哪一个',
    'added_overlaps_existing':
        '新增框与原图已有框重叠，但类名不同：可能是纠正类别，也可能是两个不同缺陷',
    'added_out_of_scope':
        '新增框的类别不在本数据集的 defect_classes 作用域内',
    'added_invalid_bbox':
        '新增框换算到原图后坐标非法（越界或宽高为 0）',
    'cross_crop_inconsistent':
        '同一个原图缺陷被多个子图裁到，各子图的清洗结果互相矛盾',
    'cross_crop_added_conflict':
        '多个子图在原图同一位置新增了不同类别；同一类别里未达到同类重叠阈值的框会一起保留',
    'same_class_overlap':
        '同类框 IoU 达到阈值，或小框大部分落在大框里。四边跨度都在容差内的已自动只留最大框；其余请选择只留较大的、只留较小的，或都留下',
    'ledger_conflict':
        '该框已被其他数据集改过，且本次要改成不同的结果',
    'sub_xml_unreadable':
        '子图 XML 解析失败，无法判断人工做了什么改动',
}


# ============================================================
#  原图 XML 索引
# ============================================================

class OriginalIndex:
    """把用到的原图 XML 一次性读进内存：类名坐标列表、图像尺寸、md5"""

    def __init__(self, xml_dir: str, names: List[str], workers: int = 32):
        self.xml_dir = xml_dir
        self.objects: Dict[str, List[Tuple[str, BBox]]] = {}
        self.size: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
        self.md5: Dict[str, str] = {}
        self.missing: List[str] = []
        self.unreadable: List[str] = []

        def load(name: str):
            path = os.path.join(xml_dir, name)
            if not os.path.exists(path):
                return name, None
            try:
                with open(path, 'rb') as f:
                    data = f.read()
                root = ET.fromstring(data)
            except Exception:
                return name, False
            objs = []
            for elem in root.findall('object'):
                name_elem = elem.find('name')
                bnd = elem.find('bndbox')
                if name_elem is None or name_elem.text is None or bnd is None:
                    continue
                try:
                    bbox = tuple(int(round(float(bnd.find(t).text)))
                                 for t in ('xmin', 'ymin', 'xmax', 'ymax'))
                except (AttributeError, TypeError, ValueError):
                    continue
                objs.append((name_elem.text.strip(), bbox))
            width = height = None
            size_elem = root.find('size')
            if size_elem is not None:
                try:
                    width = int(float(size_elem.find('width').text))
                    height = int(float(size_elem.find('height').text))
                except (AttributeError, TypeError, ValueError):
                    pass
            import hashlib
            return name, (objs, (width, height), hashlib.md5(data).hexdigest())

        with ThreadPoolExecutor(workers) as pool:
            for name, result in pool.map(load, names):
                if result is None:
                    self.missing.append(name)
                elif result is False:
                    self.unreadable.append(name)
                else:
                    self.objects[name], self.size[name], self.md5[name] = result

    def has(self, name: str) -> bool:
        return name in self.objects

    def locate(self, xml_name: str, raw_name: str, bbox: BBox) -> int:
        """
        在原图里精确定位一个框，返回命中个数。

        用「类名 + 精确坐标」而不是 IoU 模糊匹配：实测 5 个数据集 67776 条裁切记录
        100% 能精确命中，而 IoU 匹配在挂点这种重叠框密集的场景会匹配到错误的框。
        """
        target = (raw_name, tuple(bbox))
        return sum(1 for obj in self.objects.get(xml_name, []) if obj == target)


# ============================================================
#  第一阶段：逐子图 diff，产出候选变更
# ============================================================

def make_option(key: str, label: str, change: dict) -> dict:
    """
    构造一个「人工可以直接选用的处理方案」。

    change 直接就是 changeset.json 里 changes[] 的元素格式，
    这样 4_审核可视化.py 的 build-changeset 模式可以零转换地组装成补充变更集。
    """
    return {'key': key, 'label': label, 'change': change}


def _sources_of(item: dict) -> List[str]:
    sources: List[str] = []
    for src in list(item.get('sources') or []) + [item.get('sub_xml')]:
        if src and src not in sources:
            sources.append(src)
    return sources


def classes_match(comp: mc.ComponentConfig, name_a: str, name_b: str) -> bool:
    """原图类名和子图类名是否同一类（过一遍 class_mapping）。"""
    if not name_a or not name_b:
        return False
    if name_a == name_b:
        return True
    return comp.map_name(name_a) == comp.map_name(name_b)


def same_class_conflict(box_a, box_b, same_iou: float, contain_ratio: float) -> bool:
    """同类框进人工审核：IoU 达到阈值，或者小框大部分落在大框里。"""
    if mc.iou(box_a, box_b) >= same_iou:
        return True
    return mc.containment_ratio(box_a, box_b) >= contain_ratio


def boxes_edge_spread(boxes: Sequence) -> int:
    """左、上、右、下各自的最外坐标减最里坐标，取四边里最大的那一截。"""
    coords = [tuple(int(v) for v in box) for box in boxes]
    if len(coords) <= 1:
        return 0
    return max(max(box[k] for box in coords) - min(box[k] for box in coords)
               for k in range(4))


def unique_added_by_name(cluster: List[dict]) -> List[dict]:
    """
    按类别收成代表。每个框都保留：没达到同类重叠/包含阈值的框不能丢掉。
    同时兼容生成阶段的 add_name/add_bbox 和审核队列里的 name/bbox。
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    order: List[str] = []
    for c in cluster:
        name = c.get('add_name') or c.get('name')
        if not name:
            continue
        if name not in groups:
            order.append(name)
        groups[name].append(c)

    reps: List[dict] = []
    for name in order:
        group = groups[name]
        first = group[0]
        bboxes: List[list] = []
        box_sources: List[List[str]] = []
        sources: List[str] = []
        for g in group:
            bbox = list(g.get('add_bbox') or g.get('bbox'))
            srcs = _sources_of(g)
            bboxes.append(bbox)
            box_sources.append(srcs)
            for src in srcs:
                if src not in sources:
                    sources.append(src)
        reps.append({
            'add_name': name,
            'add_bbox': list(bboxes[0]),
            'bboxes': bboxes,
            'box_sources': box_sources,
            'sub_xml': sources[0] if sources else '',
            'sources': sources,
            'part_bbox': first.get('part_bbox'),
            'bbox_in_crop': first.get('bbox_in_crop'),
            'dataset': first.get('dataset'),
            'original_xml': first.get('original_xml'),
            'merged_from': len(group),
        })
    return reps


def _rep_box_count(rep: dict) -> int:
    return len(rep.get('bboxes') or [rep.get('add_bbox')])


def _rep_extra(rep: dict) -> str:
    n = _rep_box_count(rep)
    if n > 1:
        return f'，该类别有 {n} 个框都会写入'
    return ''


def _add_changes_from_rep(rep: dict) -> List[dict]:
    boxes = rep.get('bboxes') or [rep['add_bbox']]
    box_sources = rep.get('box_sources') or [rep.get('sources') or []]
    if len(box_sources) != len(boxes):
        box_sources = [list(rep.get('sources') or [])] * len(boxes)
    return [{
        'action': ACT_ADD,
        'new_name': rep['add_name'],
        'new_bbox': list(bbox),
        'sources': list(sources),
    } for bbox, sources in zip(boxes, box_sources)]


def _pack_changes(changes: List[dict]) -> dict:
    if len(changes) == 1:
        return changes[0]
    return {'action': ACT_ADD, '_multi': changes}


def _overlap_existing_review(reps: List[dict], xml_name: str,
                             raw_name: str, obj_bbox: BBox, iou_score: float) -> dict:
    """多个子图在同一位置新增、又叠在原图已有异类框上时，合成一条审核项。"""
    sources: List[str] = []
    for r in reps:
        for src in (r.get('sources') or [r.get('sub_xml')]):
            if src and src not in sources:
                sources.append(src)
    first = reps[0]
    detail = {
        'existing_name': raw_name,
        'existing_bbox': list(obj_bbox),
        'iou': round(iou_score, 4),
        'sources': sources,
    }
    if len(reps) == 1:
        extra = _rep_extra(first)
        add_changes = _add_changes_from_rep(first)
        options = [
            make_option(
                'retag',
                f'认定是同一个缺陷、人工纠正了类别：'
                f'把原图的 {raw_name} 改成 {first["add_name"]}{extra}',
                {'action': ACT_SET_NAME, 'target_name': raw_name,
                 'target_bbox': list(obj_bbox),
                 'new_name': first['add_name'], 'sources': sources}),
            make_option(
                'add',
                f'认定是不同的缺陷：额外新增 {first["add_name"]} '
                f'{[list(b) for b in (first.get("bboxes") or [first["add_bbox"]])]}'
                f'（原图已有框保持不变）{extra}',
                _pack_changes(add_changes)),
        ]
        add_name = first['add_name']
        add_bbox = list(first['add_bbox'])
        if _rep_box_count(first) > 1:
            detail['candidates'] = [{
                'sub_xml': ' / '.join(srcs),
                'name': first['add_name'],
                'bbox': list(bbox),
            } for bbox, srcs in zip(first.get('bboxes') or [], first.get('box_sources') or [])]
    else:
        detail['note'] = '多个子图在同一位置新增了不同类别，且与原图已有框重叠'
        detail['candidates'] = []
        for r in reps:
            boxes = r.get('bboxes') or [r['add_bbox']]
            srcs_list = r.get('box_sources') or [r.get('sources') or []]
            for bbox, srcs in zip(boxes, srcs_list):
                detail['candidates'].append({
                    'sub_xml': ' / '.join(srcs),
                    'name': r['add_name'],
                    'bbox': list(bbox),
                    'merged_from': 1,
                })
        options = []
        for r in reps:
            extra = _rep_extra(r)
            options.append(make_option(
                f'retag_{r["add_name"]}',
                f'把原图的 {raw_name} 改成 {r["add_name"]}{extra}',
                {'action': ACT_SET_NAME, 'target_name': raw_name,
                 'target_bbox': list(obj_bbox),
                 'new_name': r['add_name'],
                 'sources': list(r.get('sources') or [r.get('sub_xml')])}))
        options.extend(added_conflict_options(reps))
        add_name = ' / '.join(r['add_name'] for r in reps)
        add_bbox = list(first['add_bbox'])

    return {
        'reason': 'added_overlaps_existing',
        'dataset': first.get('dataset'),
        'sub_xml': ' / '.join(sources),
        'original_xml': xml_name,
        'add_name': add_name,
        'add_bbox': add_bbox,
        'part_bbox': first.get('part_bbox'),
        'bbox_in_crop': first.get('bbox_in_crop'),
        'detail': detail,
        'options': options,
    }


def added_conflict_options(reps: List[dict]) -> List[dict]:
    """
    按「不重复类别」出方案：单选某一类，或选若干类的组合。
    同一类别里没达到同类重叠阈值的框会一起写入。
    """
    options: List[dict] = []
    n = len(reps)
    for r in range(1, n + 1):
        for combo in combinations(reps, r):
            names = [c['add_name'] for c in combo]
            changes = []
            for c in combo:
                changes.extend(_add_changes_from_rep(c))
            packed = _pack_changes(changes)
            if r == 1:
                c = combo[0]
                src = '、'.join(c.get('sources') or [c['sub_xml']])
                boxes = c.get('bboxes') or [c['add_bbox']]
                options.append(make_option(
                    f'only_{c["add_name"]}',
                    f'只采纳 {c["add_name"]} {[list(b) for b in boxes]}'
                    f'（来自 {src}{_rep_extra(c)}）',
                    packed))
            elif r == n:
                options.append(make_option(
                    'all',
                    f'全部保留（{n} 个不同类别：' + '、'.join(names) + '）',
                    packed))
            else:
                options.append(make_option(
                    'plus_' + '__'.join(names),
                    '采纳 ' + ' + '.join(names),
                    packed))
    return options


def build_ledger_before_index(records: List[dict]) -> Dict[Tuple[str, Tuple], List[dict]]:
    """按「原图 XML + 修改前的坐标」索引历史台账，用于识别已经应用过的变更"""
    index: Dict[Tuple[str, Tuple], List[dict]] = defaultdict(list)
    for rec in records:
        if rec.get('action') == ACT_ADD:
            continue
        bbox = rec.get('before_bbox')
        if not bbox:
            continue
        index[(rec.get('original_xml'), tuple(bbox))].append(rec)
    return index


def build_proposals(sub_xml: str, entry: dict, sub_annotations_dir: str,
                    comp: mc.ComponentConfig, index: OriginalIndex, args,
                    dataset: str,
                    ledger_before: Dict[Tuple[str, Tuple], List[dict]]
                    ) -> Tuple[List[dict], List[dict], Counter]:
    """对一个子图做 diff，返回 (候选变更, 待确认条目, 统计计数)"""
    proposals: List[dict] = []
    reviews: List[dict] = []
    stats: Counter = Counter()

    original_xml = entry['original_defect_xml']
    part_bbox = entry['part_bbox']

    new_boxes = mc.read_objects(os.path.join(sub_annotations_dir, sub_xml))
    if new_boxes is None:
        reviews.append({'reason': 'sub_xml_unreadable', 'dataset': dataset,
                        'sub_xml': sub_xml, 'original_xml': original_xml,
                        'detail': {}, 'options': []})
        stats['sub_xml_unreadable'] += 1
        return proposals, reviews, stats

    old_defects = entry['defects']
    old_boxes = [(d['name'], tuple(d['bbox_in_crop'])) for d in old_defects]

    if not index.has(original_xml):
        # 原图 XML 缺失：整个子图的改动都无法落地
        if old_boxes or new_boxes:
            reviews.append({'reason': 'target_not_found', 'dataset': dataset,
                            'sub_xml': sub_xml, 'original_xml': original_xml,
                            'detail': {'note': '原图 XML 不存在于指定目录'},
                            'options': []})
            stats['original_missing'] += 1
        return proposals, reviews, stats

    width, height = index.size.get(original_xml, (None, None))
    ops = mc.pair_diff(old_boxes, new_boxes, iou_threshold=args.pair_iou)

    for op in ops:
        kind = op['kind']

        # -------- 新增 --------
        if kind == mc.KIND_ADDED:
            new_name, bbox_in_crop = new_boxes[op['new_index']]
            src_bbox = mc.crop_to_source(bbox_in_crop, part_bbox)
            src_bbox = mc.clip_to_image(src_bbox, width, height)
            base = {'dataset': dataset, 'sub_xml': sub_xml, 'original_xml': original_xml,
                    'add_name': new_name, 'add_bbox': list(src_bbox),
                    'bbox_in_crop': list(bbox_in_crop), 'part_bbox': list(part_bbox)}

            if not mc.is_valid_bbox(src_bbox):
                reviews.append({'reason': 'added_invalid_bbox', **base, 'detail': {},
                                'options': []})
                stats['added_invalid_bbox'] += 1
                continue

            if not args.allow_out_of_scope_add and new_name not in comp.defect_classes:
                reviews.append({'reason': 'added_out_of_scope', **base,
                                'detail': {'defect_classes': comp.defect_classes},
                                'options': [make_option(
                                    'add', f'仍然新增 {new_name} {list(src_bbox)}',
                                    {'action': ACT_ADD, 'new_name': new_name,
                                     'new_bbox': list(src_bbox), 'sources': [sub_xml]})]})
                stats['added_out_of_scope'] += 1
                continue

            proposals.append({'action': ACT_ADD, **base})
            stats['added'] += 1
            continue

        # -------- 以下都作用在「裁切时就存在的框」上 --------
        old = old_defects[op['old_index']]
        mapped_name = old['name']                    # 子图里看到的名字（已过 class_mapping）
        target_bbox = tuple(old['bbox_in_source_original'])
        was_clipped = old['was_clipped']

        # 反查原图里的真实类名：可能是映射前的细分类名（如 lmsd -> lslmqk）
        candidates = [(raw, bb) for raw, bb in index.objects[original_xml]
                      if bb == target_bbox and comp.map_name(raw) == mapped_name]

        base = {'dataset': dataset, 'sub_xml': sub_xml, 'original_xml': original_xml,
                'target_bbox': list(target_bbox), 'target_name_mapped': mapped_name,
                'was_clipped': was_clipped, 'part_bbox': list(part_bbox)}

        if len(candidates) == 0:
            # 原图里找不到 —— 有可能这个框此前已经被本工具改过（同一份变更集重复跑，
            # 或者别的数据集先动过）。先查历史台账，确认后继续按台账记录的原始类名走，
            # 交给后面的 check_ledger 判定「已应用」还是「冲突」，避免误报成找不到。
            priors = [r for r in ledger_before.get((original_xml, target_bbox), [])
                      if comp.map_name(r.get('before_name') or '') == mapped_name]
            if priors:
                candidates = [(priors[0]['before_name'], target_bbox)]
                stats['target_resolved_via_ledger'] += 1
            elif kind != mc.KIND_UNCHANGED:
                reviews.append({'reason': 'target_not_found', **base, 'detail': {
                    'op_kind': kind,
                    'note': '原图中没有类名+坐标都匹配的框，可能已被其他数据集改过'},
                    'options': []})
                stats['target_not_found'] += 1
                continue
            else:
                stats['target_not_found_but_unchanged'] += 1
                continue

        if len(candidates) > 1:
            if kind != mc.KIND_UNCHANGED:
                reviews.append({'reason': 'target_ambiguous', **base, 'detail': {
                    'op_kind': kind, 'duplicate_count': len(candidates)},
                    'options': []})
                stats['target_ambiguous'] += 1
            else:
                stats['target_ambiguous_but_unchanged'] += 1
            continue

        target_name_raw = candidates[0][0]
        base['target_name_raw'] = target_name_raw

        # -------- 未改动 --------
        if kind == mc.KIND_UNCHANGED:
            proposals.append({'action': ACT_KEEP, **base})
            stats['unchanged'] += 1
            continue

        # -------- 删除 --------
        if kind == mc.KIND_DELETED:
            if was_clipped:
                # 原图框有一截在部件框外面，人工只看到了截断的部分，不能自动删
                reviews.append({'reason': 'clipped_deleted', **base, 'detail': {},
                                'options': [make_option(
                                    'delete',
                                    f'从原图删除 {target_name_raw} {list(target_bbox)}',
                                    {'action': ACT_DELETE, 'target_name': target_name_raw,
                                     'target_bbox': list(target_bbox),
                                     'was_clipped': True, 'sources': [sub_xml]})]})
                stats['deleted_clipped_to_review'] += 1
            else:
                proposals.append({'action': ACT_DELETE, **base})
                stats['deleted'] += 1
            continue

        # -------- 修改 --------
        new_name, new_bbox_in_crop = new_boxes[op['new_index']]
        cat_changed = op['cat_changed']
        bbox_changed = op['bbox_changed']

        new_name_final = new_name if cat_changed else None
        new_bbox_final = None

        if bbox_changed:
            shift = mc.max_corner_shift(old['bbox_in_crop'], new_bbox_in_crop)
            if shift <= args.noise_tolerance:
                # 标注工具取整造成的 1~2 像素抖动，不当作真实改动
                bbox_changed = False
                stats['bbox_noise_ignored'] += 1

        if bbox_changed:
            if was_clipped and mc.touches_crop_border(new_bbox_in_crop, part_bbox,
                                                      args.border_tol):
                # 新框仍贴着裁切边界 -> 原图里很可能还是截断的，直接替换会把框改小
                replace_bbox = mc.clip_to_image(
                    mc.crop_to_source(new_bbox_in_crop, part_bbox), width, height)
                union_bbox = mc.clip_to_image(
                    mc.merge_clipped_bbox(target_bbox, new_bbox_in_crop, part_bbox,
                                          args.border_tol), width, height)
                shared = {'target_name': target_name_raw, 'target_bbox': list(target_bbox),
                          'was_clipped': True, 'sources': [sub_xml]}
                options = [
                    make_option('union',
                                f'只采纳人工调整过的那几条边，贴边的边保留原图坐标 -> '
                                f'{list(union_bbox)}',
                                {'action': ACT_SET_NAME_BBOX if cat_changed else ACT_SET_BBOX,
                                 **shared, 'new_bbox': list(union_bbox),
                                 **({'new_name': new_name} if cat_changed else {})}),
                    make_option('replace',
                                f'直接用子图新框替换（原图框会缩到部件范围内） -> '
                                f'{list(replace_bbox)}',
                                {'action': ACT_SET_NAME_BBOX if cat_changed else ACT_SET_BBOX,
                                 **shared, 'new_bbox': list(replace_bbox),
                                 **({'new_name': new_name} if cat_changed else {})}),
                ]
                if cat_changed:
                    options.append(make_option(
                        'name_only', f'只改类别为 {new_name}，坐标保持原图不变',
                        {'action': ACT_SET_NAME, **shared, 'new_name': new_name}))
                reviews.append({'reason': 'clipped_bbox_modified_border', **base, 'detail': {
                    'new_name': new_name, 'cat_changed': cat_changed,
                    'new_bbox_in_crop': list(new_bbox_in_crop),
                    'replace_bbox': list(replace_bbox),
                    'union_bbox': list(union_bbox)},
                    'options': options})
                stats['clipped_bbox_to_review'] += 1
                continue

            candidate_bbox = mc.clip_to_image(
                mc.crop_to_source(new_bbox_in_crop, part_bbox), width, height)
            if not mc.is_valid_bbox(candidate_bbox):
                reviews.append({'reason': 'added_invalid_bbox', **base, 'detail': {
                    'note': '修改后的坐标非法', 'new_bbox': list(candidate_bbox)},
                    'options': []})
                stats['modified_invalid_bbox'] += 1
                continue
            new_bbox_final = list(candidate_bbox)

        if new_name_final is None and new_bbox_final is None:
            proposals.append({'action': ACT_KEEP, **base})
            stats['unchanged'] += 1
            continue

        if new_name_final is not None and new_bbox_final is not None:
            action = ACT_SET_NAME_BBOX
            stats['modified_name_and_bbox'] += 1
        elif new_name_final is not None:
            action = ACT_SET_NAME
            stats['modified_name'] += 1
        else:
            action = ACT_SET_BBOX
            stats['modified_bbox'] += 1

        if new_name_final is not None and comp.is_ambiguous_target(new_name_final):
            # 改成了「映射后」的歧义类名。按既定决策直接写入，这里只做统计留痕。
            stats['modified_name_to_mapped_class'] += 1

        proposals.append({'action': action, **base,
                          'new_name': new_name_final, 'new_bbox': new_bbox_final,
                          'new_bbox_in_crop': list(new_bbox_in_crop)})

    return proposals, reviews, stats


# ============================================================
#  第二阶段：跨子图一致性
# ============================================================

def outcome_of(proposal: dict) -> Tuple:
    """把一条变更折算成「这个框最终会变成什么」，用于比较多个子图的结论是否一致"""
    action = proposal['action']
    if action == ACT_DELETE:
        return ('delete',)
    name = proposal.get('new_name') or proposal['target_name_raw']
    bbox = tuple(proposal.get('new_bbox') or proposal['target_bbox'])
    return ('set', name, bbox)


def resolve_cross_crop(proposals: List[dict], args) -> Tuple[List[dict], List[dict], Counter]:
    """
    同一个原图缺陷可能被多个子图裁到（挂点尤其常见），需要把各子图的结论汇总判断。

    策略：
      - 删除 与 保留/修改 并存      -> 冲突（删除是破坏性的，必须所有子图一致）
      - 多个互不相同的修改          -> 冲突
      - 修改 与 未改动 并存         -> 采纳修改（通常是人工只在其中一个子图上作业）
                                      加 --strict-consistency 可把这种也算冲突
    """
    accepted: List[dict] = []
    reviews: List[dict] = []
    stats: Counter = Counter()

    groups: Dict[Tuple, List[dict]] = defaultdict(list)
    for p in proposals:
        if p['action'] == ACT_ADD:
            accepted.append(p)
            continue
        key = (p['original_xml'], p['target_name_raw'], tuple(p['target_bbox']))
        groups[key].append(p)

    for key, items in groups.items():
        if len(items) == 1:
            if items[0]['action'] != ACT_KEEP:
                accepted.append(items[0])
            continue

        stats['multi_crop_groups'] += 1
        keeps = [p for p in items if p['action'] == ACT_KEEP]
        changes = [p for p in items if p['action'] != ACT_KEEP]

        if not changes:
            continue

        distinct = {outcome_of(p) for p in changes}
        has_delete = any(p['action'] == ACT_DELETE for p in changes)

        conflict_reason = None
        if len(distinct) > 1:
            conflict_reason = '多个子图给出了互相矛盾的修改结果'
        elif has_delete and keeps:
            conflict_reason = '一部分子图删除了该缺陷，另一部分子图保留了它'
        elif keeps and args.strict_consistency:
            conflict_reason = '一部分子图修改了该缺陷，另一部分子图未做改动（严格模式）'

        if conflict_reason:
            options = []
            seen_outcomes = set()
            for p in changes:
                outcome = outcome_of(p)
                if outcome in seen_outcomes:
                    continue
                seen_outcomes.add(outcome)
                shared = {'target_name': key[1], 'target_bbox': list(key[2]),
                          'was_clipped': p.get('was_clipped', False),
                          'sources': [p['sub_xml']]}
                if p['action'] == ACT_DELETE:
                    change = {'action': ACT_DELETE, **shared}
                    label = f'采纳「删除」（来自 {p["sub_xml"]}）'
                else:
                    change = {'action': p['action'], **shared}
                    if p.get('new_name') is not None:
                        change['new_name'] = p['new_name']
                    if p.get('new_bbox') is not None:
                        change['new_bbox'] = list(p['new_bbox'])
                    label = (f'采纳 {p["action"]} -> 类别 {outcome[1]} 坐标 {list(outcome[2])}'
                             f'（来自 {p["sub_xml"]}）')
                options.append(make_option(f'opt{len(options)}', label, change))

            reviews.append({
                'reason': 'cross_crop_inconsistent',
                'dataset': items[0]['dataset'],
                'sub_xml': ' / '.join(p['sub_xml'] for p in items),
                'original_xml': key[0],
                'target_name_raw': key[1],
                'target_bbox': list(key[2]),
                'detail': {
                    'note': conflict_reason,
                    'candidates': [{'sub_xml': p['sub_xml'], 'action': p['action'],
                                    'new_name': p.get('new_name'),
                                    'new_bbox': p.get('new_bbox')} for p in items],
                },
                'options': options,
            })
            stats['cross_crop_conflict'] += 1
            continue

        # 结论一致：取第一条，并记录所有来源子图
        winner = dict(changes[0])
        winner['sources'] = [p['sub_xml'] for p in items]
        if keeps:
            stats['modify_over_keep'] += 1
        accepted.append(winner)
        stats['cross_crop_merged'] += 1

    return accepted, reviews, stats


# ============================================================
#  第三阶段：新增框去重
# ============================================================

def _matching_same_class_existing(bbox, add_name: str, existing: List[Tuple[str, BBox]],
                                  comp: mc.ComponentConfig, same_iou: float,
                                  contain_ratio: float) -> List[dict]:
    """同类原图框里，所有重叠或包含达到阈值的框。"""
    found = []
    for raw_name, obj_bbox in existing:
        if not classes_match(comp, raw_name, add_name):
            continue
        iou_score = mc.iou(bbox, obj_bbox)
        contain = mc.containment_ratio(bbox, obj_bbox)
        if iou_score < same_iou and contain < contain_ratio:
            continue
        found.append({
            'key': (raw_name, tuple(obj_bbox)),
            'name': raw_name,
            'bbox': tuple(obj_bbox),
            'iou': iou_score,
            'containment': contain,
        })
    return found


def _best_different_class_overlap(bbox, add_name: str, existing: List[Tuple[str, BBox]],
                                  comp: mc.ComponentConfig, diff_iou: float):
    """异类原图框里 IoU 达到阈值的最强一个。没有则返回 None。"""
    best = None
    for raw_name, obj_bbox in existing:
        if classes_match(comp, raw_name, add_name):
            continue
        score = mc.iou(bbox, obj_bbox)
        if score < diff_iou:
            continue
        if best is None or score > best[0]:
            best = (score, raw_name, tuple(obj_bbox))
    if best is None:
        return None
    score, raw_name, obj_bbox = best
    return {'key': (raw_name, obj_bbox), 'name': raw_name, 'bbox': obj_bbox, 'iou': score}


def _kept_adds_from_rep(rep: dict) -> List[dict]:
    """一个类别代表里的每个框各写成一条新增，不丢框。"""
    boxes = rep.get('bboxes') or [rep['add_bbox']]
    box_sources = rep.get('box_sources') or [rep.get('sources') or []]
    if len(box_sources) != len(boxes):
        box_sources = [list(rep.get('sources') or [])] * len(boxes)
    kept = []
    for bbox, sources in zip(boxes, box_sources):
        kept.append({
            'action': ACT_ADD,
            'dataset': rep.get('dataset'),
            'sub_xml': (sources[0] if sources else rep.get('sub_xml') or ''),
            'original_xml': rep.get('original_xml'),
            'add_name': rep['add_name'],
            'add_bbox': list(bbox),
            'bbox_in_crop': list(rep['bbox_in_crop']) if rep.get('bbox_in_crop') else [],
            'part_bbox': list(rep['part_bbox']) if rep.get('part_bbox') else [],
            'sources': list(sources),
        })
    return kept


def _describe_cand(cand: dict) -> str:
    if cand['role'] == 'existing':
        where = '原图已有'
    else:
        src = '、'.join(cand.get('sources') or []) or cand.get('sub_xml') or '?'
        where = f'新增，来自 {src}'
    return f'{where} {cand["name"]} {list(cand["bbox"])} 面积 {cand["area"]}'


def _effect_text(changes: List[dict], kept_existing: bool) -> str:
    acts = [c['action'] for c in changes]
    if acts == [ACT_KEEP]:
        return '原图框保持不动，不新增'
    if acts == [ACT_SET_BBOX]:
        return '把原图框改成这个坐标，不另加新框'
    if ACT_DELETE in acts:
        return '删掉原图上参与冲突的框，再写入选中的新增'
    if kept_existing:
        return '原图框保持不动，并写入选中的新增'
    if len(acts) > 1:
        return f'写入 {len(acts)} 个新增框'
    return '写入这个新增框'


def _add_change_from_cand(cand: dict) -> dict:
    return {'action': ACT_ADD, 'new_name': cand['name'], 'new_bbox': list(cand['bbox']),
            'sources': list(cand.get('sources') or [])}


def _changes_for_same_class(chosen: List[dict], cands: List[dict],
                            same_iou: float, contain_ratio: float) -> List[dict]:
    """把「留下哪些框」折成可写入的变更。原图框还在且只改尺寸时用 set_bbox。"""
    chosen_ids = {id(c) for c in chosen}
    existings = [c for c in cands if c['role'] == 'existing']
    chosen_adds = [c for c in chosen if c['role'] == 'add']
    sources: List[str] = []
    for c in cands:
        for src in c.get('sources') or []:
            if src not in sources:
                sources.append(src)
    if not existings:
        return [_add_change_from_cand(a) for a in chosen_adds]

    existing = existings[0]
    if id(existing) in chosen_ids:
        changes = [_add_change_from_cand(a) for a in chosen_adds]
        if not changes:
            return [{'action': ACT_KEEP, 'target_name': existing['name'],
                     'target_bbox': list(existing['bbox']), 'sources': sources}]
        return changes
    if (len(chosen_adds) == 1 and same_class_conflict(
            chosen_adds[0]['bbox'], existing['bbox'], same_iou, contain_ratio)):
        return [{'action': ACT_SET_BBOX, 'target_name': existing['name'],
                 'target_bbox': list(existing['bbox']),
                 'new_bbox': list(chosen_adds[0]['bbox']),
                 'sources': list(chosen_adds[0].get('sources') or sources)}]
    changes = [{'action': ACT_DELETE, 'target_name': existing['name'],
                'target_bbox': list(existing['bbox']), 'sources': sources}]
    changes.extend(_add_change_from_cand(a) for a in chosen_adds)
    return changes


def _same_class_option(key: str, prefix: str, chosen: List[dict], cands: List[dict],
                       same_iou: float, contain_ratio: float) -> dict:
    changes = _changes_for_same_class(chosen, cands, same_iou, contain_ratio)
    existings = [c for c in cands if c['role'] == 'existing']
    kept_existing = bool(existings) and id(existings[0]) in {id(c) for c in chosen}
    label = (f'{prefix}：' + '；'.join(_describe_cand(c) for c in chosen)
             + f'（{_effect_text(changes, kept_existing)}）')
    return make_option(key, label, _pack_changes(changes))


def _same_class_review(add_items: List[dict], existing_hit: Optional[dict],
                       xml_name: str, all_existing: List[Tuple[str, BBox]],
                       comp: mc.ComponentConfig, args) -> dict:
    """同类重叠或大框套小框：一张审核卡，方案是留大、留小或都留。"""
    cands: List[dict] = []
    for item in add_items:
        bbox = [int(v) for v in item['add_bbox']]
        cands.append({
            'role': 'add',
            'name': item['add_name'],
            'bbox': bbox,
            'area': int(mc.box_area(bbox)),
            'sub_xml': item.get('sub_xml') or '',
            'sources': _sources_of(item),
            'part_bbox': item.get('part_bbox'),
            'bbox_in_crop': item.get('bbox_in_crop'),
            'dataset': item.get('dataset'),
        })
    if existing_hit is not None:
        bbox = [int(v) for v in existing_hit['bbox']]
        cands.append({
            'role': 'existing',
            'name': existing_hit['name'],
            'bbox': bbox,
            'area': int(mc.box_area(bbox)),
            'sub_xml': '',
            'sources': [],
        })
    cands.sort(key=lambda c: (-c['area'], c['bbox'], 0 if c['role'] == 'existing' else 1))

    same_iou = args.same_iou
    contain_ratio = args.contain_ratio
    areas = [c['area'] for c in cands]
    options: List[dict] = []
    if len(cands) == 2:
        large, small = cands
        if large['area'] > small['area']:
            large_prefix, small_prefix = '只留较大的框', '只留较小的框'
        else:
            large_prefix, small_prefix = '只留这个框', '只留另一个框'
        options.append(_same_class_option(
            'keep_large', large_prefix, [large], cands, same_iou, contain_ratio))
        options.append(_same_class_option(
            'keep_small', small_prefix, [small], cands, same_iou, contain_ratio))
        options.append(_same_class_option(
            'keep_both', '两个都留', cands, cands, same_iou, contain_ratio))
    else:
        max_area, min_area = max(areas), min(areas)
        unique_max = areas.count(max_area) == 1
        unique_min = areas.count(min_area) == 1
        for i, cand in enumerate(cands):
            if unique_max and cand['area'] == max_area:
                prefix = '只留面积最大的框'
            elif unique_min and cand['area'] == min_area:
                prefix = '只留面积最小的框'
            else:
                prefix = '只留这个框'
            options.append(_same_class_option(
                f'only_{i}', prefix, [cand], cands, same_iou, contain_ratio))
        options.append(_same_class_option(
            'keep_all', f'全部保留（{len(cands)} 个框都留下）', cands, cands,
            same_iou, contain_ratio))

    pairs = []
    for i, j in combinations(range(len(cands)), 2):
        pairs.append({
            'i': i, 'j': j,
            'iou': round(mc.iou(cands[i]['bbox'], cands[j]['bbox']), 4),
            'containment': round(mc.containment_ratio(cands[i]['bbox'], cands[j]['bbox']), 4),
        })
    if len(cands) == 2:
        note = (f'同类框重叠或大框套小框，需要人工选择。'
                f'IoU={pairs[0]["iou"]}，包含率={pairs[0]["containment"]}。')
    else:
        note = f'{len(cands)} 个同类框互相重叠或套在一起，需要人工选择留哪一个，或全部保留。'

    diff_notes = []
    ignored = existing_hit['key'] if existing_hit else None
    for cand in cands:
        if cand['role'] != 'add':
            continue
        hit = _best_different_class_overlap(
            cand['bbox'], cand['name'], all_existing, comp, args.dup_iou)
        if hit is None or hit['key'] == ignored:
            continue
        diff_notes.append({
            'add_bbox': list(cand['bbox']),
            'existing_name': hit['name'],
            'existing_bbox': list(hit['bbox']),
            'iou': round(hit['iou'], 4),
        })
    if diff_notes:
        note += ' 其中有的新增框还与原图异类框达到重叠阈值，选「都留」或「全部保留」时那些异类框仍会留着。'

    add_cands = [c for c in cands if c['role'] == 'add']
    largest = cands[0]
    part_bbox = next((c.get('part_bbox') for c in add_cands if c.get('part_bbox')), None)
    bbox_in_crop = next((c.get('bbox_in_crop') for c in add_cands if c.get('bbox_in_crop')), None)
    return {
        'reason': 'same_class_overlap',
        'dataset': add_cands[0].get('dataset') if add_cands else None,
        'sub_xml': ' / '.join(c['sub_xml'] for c in add_cands if c.get('sub_xml')),
        'original_xml': xml_name,
        'add_name': add_cands[0]['name'] if add_cands else '',
        'add_bbox': list(largest['bbox']),
        'part_bbox': part_bbox,
        'bbox_in_crop': bbox_in_crop,
        'detail': {
            'note': note,
            'same_iou': same_iou,
            'contain_ratio': contain_ratio,
            'dup_iou': args.dup_iou,
            'pairs': pairs,
            'different_class_overlaps': diff_notes,
            'candidates': [{
                'role': c['role'],
                'name': c['name'],
                'bbox': list(c['bbox']),
                'area': c['area'],
                'sub_xml': c.get('sub_xml') or '',
                'sources': list(c.get('sources') or []),
            } for c in cands],
        },
        'options': options,
    }


def _cross_crop_added_review(reps: List[dict], xml_name: str) -> dict:
    p = reps[0]
    candidates = []
    for r in reps:
        boxes = r.get('bboxes') or [r['add_bbox']]
        srcs_list = r.get('box_sources') or [r.get('sources') or []]
        for bbox, srcs in zip(boxes, srcs_list):
            candidates.append({
                'sub_xml': ' / '.join(srcs),
                'name': r['add_name'],
                'bbox': list(bbox),
                'merged_from': 1,
            })
    return {
        'reason': 'cross_crop_added_conflict',
        'dataset': p.get('dataset'),
        'sub_xml': ' / '.join(r.get('sub_xml') or '' for r in reps),
        'original_xml': xml_name,
        'add_name': ' / '.join(r['add_name'] for r in reps),
        'add_bbox': list(p['add_bbox']),
        'part_bbox': p.get('part_bbox'),
        'detail': {
            'note': '不同类别的新增框重叠。同一类别里未达到同类重叠阈值的框，选中该类别时会一起写入',
            'candidates': candidates,
        },
        'options': added_conflict_options(reps),
    }


def _finish_different_class(items: List[dict], existing: List[Tuple[str, BBox]],
                            xml_name: str, comp: mc.ComponentConfig, diff_iou: float,
                            kept: List[dict], reviews: List[dict], stats: Counter) -> None:
    """异类新增：IoU ≥ diff_iou 进审核；同一类别里没达到同类阈值的框都保留。"""
    if not items:
        return
    groups = mc.cluster_indices(len(items), lambda i, j: (
        items[i]['sub_xml'] != items[j]['sub_xml']
        and not classes_match(comp, items[i]['add_name'], items[j]['add_name'])
        and mc.iou(items[i]['add_bbox'], items[j]['add_bbox']) >= diff_iou
    ))
    for idxs in groups:
        cluster = [items[i] for i in idxs]
        pending: List[dict] = []
        by_existing: Dict[Tuple, List[dict]] = defaultdict(list)
        for item in cluster:
            hit = _best_different_class_overlap(
                item['add_bbox'], item['add_name'], existing, comp, diff_iou)
            if hit is None:
                pending.append(item)
            else:
                by_existing[hit['key']].append(item)

        for key, group_items in by_existing.items():
            reps = unique_added_by_name(group_items)
            raw_name, obj_bbox = key
            score = max(mc.iou(it['add_bbox'], obj_bbox) for it in group_items)
            reviews.append(_overlap_existing_review(
                reps, xml_name, raw_name, obj_bbox, score))
            stats['added_overlaps_existing'] += 1

        if not pending:
            continue
        # 已经对过原图框。这里只处理它们彼此之间还剩的异类重叠，避免再递归。
        sub_groups = mc.cluster_indices(len(pending), lambda i, j: (
            pending[i]['sub_xml'] != pending[j]['sub_xml']
            and not classes_match(comp, pending[i]['add_name'], pending[j]['add_name'])
            and mc.iou(pending[i]['add_bbox'], pending[j]['add_bbox']) >= diff_iou
        ))
        for sub in sub_groups:
            reps = unique_added_by_name([pending[i] for i in sub])
            if len(reps) <= 1:
                for rep in reps:
                    kept.extend(_kept_adds_from_rep(rep))
            else:
                reviews.append(_cross_crop_added_review(reps, xml_name))
                stats['added_conflict'] += 1


def _unique_existing_hits(group_hits: List[Optional[dict]]) -> List[dict]:
    """一簇新增框命中的原图同类框，按框去重，保留首次出现的顺序。"""
    found: List[dict] = []
    seen = set()
    for hit in group_hits:
        if hit is None or hit['key'] in seen:
            continue
        seen.add(hit['key'])
        found.append(hit)
    return found


def _auto_keep_largest(add_items: List[dict], existing_hits: List[dict],
                       xml_name: str, index: OriginalIndex,
                       comp: mc.ComponentConfig,
                       pending_targets: set) -> Optional[List[dict]]:
    """
    这一簇四边都在容差内：只留面积最大的框。

    写不稳的情况返回 None，调用方改走人工审核：
      - 原图框在文件里不是恰好一个（删或改会命中失败或命中多个）
      - 这个原图框已经有别的变更要改它
    面积相同时优先留原图框，避免无谓改写。
    """
    for hit in existing_hits:
        key = (xml_name, hit['name'], tuple(int(v) for v in hit['bbox']))
        if key in pending_targets:
            return None
        if index.locate(xml_name, hit['name'], tuple(hit['bbox'])) != 1:
            return None

    members: List[dict] = []
    for item in add_items:
        bbox = [int(v) for v in item['add_bbox']]
        members.append({
            'role': 'add', 'name': item['add_name'], 'bbox': bbox,
            'area': int(mc.box_area(bbox)), 'item': item,
        })
    for hit in existing_hits:
        bbox = [int(v) for v in hit['bbox']]
        members.append({
            'role': 'existing', 'name': hit['name'], 'bbox': bbox,
            'area': int(mc.box_area(bbox)),
        })
    winner = max(members, key=lambda c: (
        c['area'], 1 if c['role'] == 'existing' else 0, tuple(c['bbox'])))

    sources: List[str] = []
    for item in add_items:
        for src in _sources_of(item):
            if src not in sources:
                sources.append(src)
    dataset = add_items[0].get('dataset')
    sub_xml = sources[0] if sources else (add_items[0].get('sub_xml') or '')
    part_bbox = list(add_items[0].get('part_bbox') or [])

    out: List[dict] = []
    existings = [m for m in members if m['role'] == 'existing']
    host = None
    if winner['role'] == 'existing':
        host = winner
    elif existings:
        host = max(existings, key=lambda c: (c['area'], tuple(c['bbox'])))
        if tuple(host['bbox']) != tuple(winner['bbox']):
            out.append({
                'action': ACT_SET_BBOX,
                'dataset': dataset,
                'sub_xml': sub_xml,
                'original_xml': xml_name,
                'target_name_raw': host['name'],
                'target_name_mapped': comp.map_name(host['name']),
                'target_bbox': list(host['bbox']),
                'new_name': None,
                'new_bbox': list(winner['bbox']),
                'was_clipped': False,
                'part_bbox': part_bbox,
                'sources': list(sources),
            })
    else:
        item = winner['item']
        out.append({
            'action': ACT_ADD,
            'dataset': item.get('dataset'),
            'sub_xml': item.get('sub_xml') or sub_xml,
            'original_xml': xml_name,
            'add_name': item['add_name'],
            'add_bbox': list(winner['bbox']),
            'bbox_in_crop': list(item['bbox_in_crop']) if item.get('bbox_in_crop') else [],
            'part_bbox': list(item['part_bbox']) if item.get('part_bbox') else [],
            'sources': list(sources),
        })

    for ex in existings:
        if host is not None and ex is host:
            continue
        out.append({
            'action': ACT_DELETE,
            'dataset': dataset,
            'sub_xml': sub_xml,
            'original_xml': xml_name,
            'target_name_raw': ex['name'],
            'target_name_mapped': comp.map_name(ex['name']),
            'target_bbox': list(ex['bbox']),
            'was_clipped': False,
            'part_bbox': [],
            'sources': list(sources),
        })
    return out


def resolve_added(proposals: List[dict], index: OriginalIndex,
                  comp: mc.ComponentConfig, args) -> Tuple[List[dict], List[dict], Counter]:
    """
    新增框去重。

      1) 同类，不论是否同一张子图：IoU ≥ same_iou，或包含率 ≥ contain_ratio，算同一簇。
         和原图已有同类框达到同样条件的，也算进这一簇。
      2) 一簇里每一个框（含命中的原图框）左、上、右、下的跨度都不超过 same_edge_px
         时，自动只留面积最大的框。超过的进人工审核。
      3) 异类：IoU ≥ dup_iou 才进审核；低于该值的两个都留。同一张子图里的异类重叠不参与。
    """
    kept: List[dict] = []
    reviews: List[dict] = []
    stats: Counter = Counter()

    adds = [p for p in proposals if p['action'] == ACT_ADD]
    kept.extend(p for p in proposals if p['action'] != ACT_ADD)

    pending_targets = {
        (p['original_xml'], p.get('target_name_raw'), tuple(p['target_bbox']))
        for p in proposals
        if p['action'] not in (ACT_ADD, ACT_KEEP) and p.get('target_bbox')
    }
    # 同一轮里已经自动处理过的原图框。另一簇再碰到它就改走审核，避免对同一个框写两次。
    auto_touched = set()

    by_xml: Dict[str, List[dict]] = defaultdict(list)
    for p in adds:
        by_xml[p['original_xml']].append(p)

    same_iou = args.same_iou
    contain_ratio = args.contain_ratio
    diff_iou = args.dup_iou
    edge_px = args.same_edge_px

    for xml_name, items in by_xml.items():
        existing = index.objects.get(xml_name, [])
        match_lists = [
            _matching_same_class_existing(
                item['add_bbox'], item['add_name'], existing, comp, same_iou, contain_ratio)
            for item in items
        ]
        hits = []
        for matches in match_lists:
            best_match = None
            for hit in matches:
                rank = (hit['containment'], hit['iou'])
                if best_match is None or rank > best_match[0]:
                    best_match = (rank, hit)
            hits.append(None if best_match is None else best_match[1])

        def _same_link(i: int, j: int, _items=items, _hits=hits) -> bool:
            if not classes_match(comp, _items[i]['add_name'], _items[j]['add_name']):
                return False
            box_i, box_j = _items[i]['add_bbox'], _items[j]['add_bbox']
            if same_class_conflict(box_i, box_j, same_iou, contain_ratio):
                return True
            hi, hj = _hits[i], _hits[j]
            return bool(hi and hj and hi['key'] == hj['key'])

        consumed = [False] * len(items)
        for idxs in mc.cluster_indices(len(items), _same_link):
            group_hits = [hits[i] for i in idxs]
            if len(idxs) == 1 and group_hits[0] is None:
                continue
            add_items = [items[i] for i in idxs]
            existings = _unique_existing_hits(
                [hit for i in idxs for hit in match_lists[i]])
            spread_boxes = [it['add_bbox'] for it in add_items]
            spread_boxes.extend(hit['bbox'] for hit in existings)
            touched = any(
                (xml_name, hit['name'], tuple(int(v) for v in hit['bbox'])) in auto_touched
                for hit in existings)
            if (not touched and boxes_edge_spread(spread_boxes) <= edge_px):
                auto = _auto_keep_largest(
                    add_items, existings, xml_name, index, comp, pending_targets)
                if auto is not None:
                    kept.extend(auto)
                    for change in auto:
                        if change['action'] == ACT_ADD:
                            continue
                        key = (change['original_xml'], change['target_name_raw'],
                               tuple(change['target_bbox']))
                        pending_targets.add(key)
                    for hit in existings:
                        auto_touched.add(
                            (xml_name, hit['name'], tuple(int(v) for v in hit['bbox'])))
                    stats['same_class_edge_auto_kept'] += 1
                    for i in idxs:
                        consumed[i] = True
                    continue
            best_hit = None
            for hit in group_hits:
                if hit is None:
                    continue
                rank = (hit['containment'], hit['iou'])
                if best_hit is None or rank > best_hit[0]:
                    best_hit = (rank, hit)
            review = _same_class_review(
                add_items,
                None if best_hit is None else best_hit[1],
                xml_name, existing, comp, args)
            others = []
            seen = set()
            if best_hit is not None:
                seen.add(best_hit[1]['key'])
            for hit in existings:
                if hit['key'] in seen:
                    continue
                seen.add(hit['key'])
                others.append({
                    'name': hit['name'],
                    'bbox': list(hit['bbox']),
                    'iou': round(hit['iou'], 4),
                    'containment': round(hit['containment'], 4),
                })
            if others:
                review['detail']['other_existing'] = others
                review['detail']['note'] += (
                    ' 另外的同类原图框也达到阈值，没有放进方案，它们保持不动。')
            reviews.append(review)
            stats['same_class_overlap_to_review'] += 1
            for i in idxs:
                consumed[i] = True

        leftovers = [items[i] for i in range(len(items)) if not consumed[i]]
        _finish_different_class(
            leftovers, existing, xml_name, comp, diff_iou, kept, reviews, stats)

    return kept, reviews, stats


# ============================================================
#  第四阶段：历史台账冲突检测
# ============================================================

def check_ledger(proposals: List[dict], ledger_records: List[dict],
                 dataset: str) -> Tuple[List[dict], List[dict], Counter]:
    """
    逐数据集分多次回写同一份原图库，必须靠台账避免互相覆盖。

      - 同数据集、同一个框、同样的结果已经应用过 -> 跳过（幂等，可重复跑）
      - 其他数据集改过同一个框且结果不同        -> 冲突，交人工确认
    """
    kept: List[dict] = []
    reviews: List[dict] = []
    stats: Counter = Counter()

    history: Dict[Tuple, List[dict]] = defaultdict(list)
    for rec in ledger_records:
        if rec.get('action') == ACT_ADD:
            key = (rec.get('original_xml'), 'ADD', tuple(rec.get('after_bbox') or ()))
        else:
            key = (rec.get('original_xml'), rec.get('before_name'),
                   tuple(rec.get('before_bbox') or ()))
        history[key].append(rec)

    for p in proposals:
        if p['action'] == ACT_ADD:
            key = (p['original_xml'], 'ADD', tuple(p['add_bbox']))
            past = history.get(key, [])
            same = [r for r in past if r.get('after_name') == p['add_name']]
            if same:
                stats['ledger_already_applied'] += 1
                continue
            kept.append(p)
            continue

        key = (p['original_xml'], p['target_name_raw'], tuple(p['target_bbox']))
        past = history.get(key, [])
        if not past:
            kept.append(p)
            continue

        want = outcome_of(p)
        matched = False
        for rec in past:
            if rec.get('action') == ACT_DELETE:
                got = ('delete',)
            else:
                got = ('set', rec.get('after_name'), tuple(rec.get('after_bbox') or ()))
            if got == want:
                matched = True
                break

        if matched:
            stats['ledger_already_applied'] += 1
            continue

        force_change = {'action': p['action'], 'target_name': p['target_name_raw'],
                        'target_bbox': list(p['target_bbox']),
                        'was_clipped': p.get('was_clipped', False),
                        'sources': p.get('sources', [p['sub_xml']])}
        if p.get('new_name') is not None:
            force_change['new_name'] = p['new_name']
        if p.get('new_bbox') is not None:
            force_change['new_bbox'] = list(p['new_bbox'])

        reviews.append({
            'reason': 'ledger_conflict',
            'dataset': dataset, 'sub_xml': p['sub_xml'], 'original_xml': p['original_xml'],
            'target_name_raw': p['target_name_raw'], 'target_bbox': p['target_bbox'],
            'detail': {'want': list(want),
                       'history': [{'dataset': r.get('dataset'), 'action': r.get('action'),
                                    'after_name': r.get('after_name'),
                                    'after_bbox': r.get('after_bbox'),
                                    'applied_at': r.get('applied_at')} for r in past]},
            'options': [make_option(
                'force', f'覆盖此前的改动，仍按本数据集的结果执行（{list(want)}）',
                force_change)],
        })
        stats['ledger_conflict'] += 1

    return kept, reviews, stats


# ============================================================
#  报告
# ============================================================

def write_report(path: str, args, comp: mc.ComponentConfig, totals: Counter,
                 review_counter: Counter, changeset: dict, extra_lines: List[str]) -> str:
    lines: List[str] = []
    add = lines.append

    add('=' * 72)
    add('缺陷子图标签 -> 原图总库  变更集生成报告')
    add('=' * 72)
    add(f'生成时间      : {mc.now_iso()}')
    add(f'数据集        : {args.dataset_dir}')
    add(f'部件类型      : {args.component_type}')
    add(f'子图 XML 目录 : {args.sub_annotations}')
    add(f'原图 XML 目录 : {args.original_xml}')
    add(f'历史台账      : {args.ledger}')
    add(f'配对 IoU 阈值 : {args.pair_iou}')
    add(f'异类去重 IoU  : {args.dup_iou}   同类重叠 IoU: {args.same_iou}'
        f'   包含率: {args.contain_ratio}')
    add(f'贴边容差      : {args.border_tol} px   取整噪声容差: {args.noise_tolerance} px'
        f'   同类四边自动合并: {args.same_edge_px} px')
    add(f'一致性模式    : {"严格" if args.strict_consistency else "宽松（修改优先于未改动）"}')
    add('')

    add('-' * 72)
    add('一、扫描概况')
    add('-' * 72)
    for key in ('crop_mapping_entries', 'sub_xml_present', 'matched_pairs',
                'mapping_without_subxml', 'subxml_without_mapping',
                'original_referenced', 'original_missing_file', 'original_unreadable'):
        if key in totals:
            add(f'  {key:32s} {totals[key]}')
    add('')

    add('-' * 72)
    add('二、逐子图 diff 结果')
    add('-' * 72)
    for key in ('unchanged', 'modified_name', 'modified_bbox', 'modified_name_and_bbox',
                'deleted', 'added', 'bbox_noise_ignored',
                'modified_name_to_mapped_class'):
        add(f'  {key:32s} {totals.get(key, 0)}')
    add('')

    add('-' * 72)
    add('三、去重与合并')
    add('-' * 72)
    for key in ('multi_crop_groups', 'cross_crop_merged', 'modify_over_keep',
                'same_class_edge_auto_kept',
                'same_class_overlap_to_review',
                'target_resolved_via_ledger', 'ledger_already_applied'):
        add(f'  {key:32s} {totals.get(key, 0)}')
    add('  同类不论是否同一张子图：IoU≥same_iou 或包含率≥contain_ratio 收成一簇，原图同类框同样算进去')
    add(f'  一簇里每个框的四边跨度都不超过 {args.same_edge_px}px 时自动只留面积最大的框')
    add('  超过的进审核。异类仍按 dup_iou 进审核')
    add('')

    add('-' * 72)
    add('四、待人工确认（不会自动写入）')
    add('-' * 72)
    if review_counter:
        for reason, count in review_counter.most_common():
            add(f'  {reason:32s} {count}')
            add(f'      {REASONS.get(reason, "")}')
    else:
        add('  无')
    add('')

    action_counter = Counter()
    file_count = len(changeset['files'])
    for changes in changeset['files'].values():
        for ch in changes['changes']:
            action_counter[ch['action']] += 1

    add('-' * 72)
    add('五、最终变更集（changeset.json）')
    add('-' * 72)
    add(f'  受影响的原图 XML 文件数        {file_count}')
    for action in (ACT_SET_NAME, ACT_SET_BBOX, ACT_SET_NAME_BBOX, ACT_DELETE, ACT_ADD):
        add(f'  {action:32s} {action_counter.get(action, 0)}')
    add(f'  {"变更总条数":30s} {sum(action_counter.values())}')
    add('')

    if extra_lines:
        add('-' * 72)
        add('六、提示')
        add('-' * 72)
        for line in extra_lines:
            add(f'  {line}')
        add('')

    add('=' * 72)
    add('本脚本没有修改任何原图文件。确认无误后运行 2_应用变更集.py 才会真正写入。')
    add('=' * 72)

    text = '\n'.join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    return text


# ============================================================
#  主流程
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='生成子图标签回写原图总库的变更集（纯只读，不修改任何原图）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 1_生成变更集.py \
    --dataset-dir /raid/datasets_defect_2026/datasets_val/全量_正样本/gd_data \
    --component-type gd \
    --original-xml /raid/Nas-122/项目数据/输电项目/缺陷/标记样本库/全图测试集2026/Annotations \
    --output /raid/datasets_defect_2026/datasets_val/全量_正样本/gd_data/映射原图过程

""")

    parser.add_argument('--dataset-dir', required=True,
                        help='子图数据集根目录（含 images / Annotations / crop_mapping_*.json）')
    parser.add_argument('--original-xml', required=True,
                        help='原图缺陷 XML 目录（只读）')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--component-type', required=True, choices=mc.COMPONENT_TYPES,
                        help='数据集类型：gd 挂点，jyz 绝缘子，gt 杆塔，ddx 导地线，jc 基础，global 全图')
    parser.add_argument('--sub-annotations', default=None,
                        help='子图 XML 目录（默认 <dataset-dir>/Annotations）')
    parser.add_argument('--crop-mapping', action='append', default=None,
                        help='crop_mapping json 路径，可重复指定；默认自动发现正/负样本两个文件')
    parser.add_argument('--config', default=None,
                        help='裁切配置 config.yaml（默认 ../1-crop_defect_new/config.yaml）')
    parser.add_argument('--ledger', default=None,
                        help='变更台账路径（默认放在原图 XML 目录的上一级）')

    parser.add_argument('--pair-iou', type=float, default=0.5,
                        help='新旧框配对的 IoU 阈值（默认 0.5）')
    parser.add_argument('--dup-iou', type=float, default=0.5,
                        help='异类新增框重叠进审核的 IoU 阈值（默认 0.5）')
    parser.add_argument('--same-iou', type=float, default=0.3,
                        help='同类新增框重叠进审核的 IoU 阈值（默认 0.3）')
    parser.add_argument('--contain-ratio', type=float, default=0.9,
                        help='同类框包含率（交集/较小框面积）达到该值就收成一簇，'
                             '同一张子图和不同子图用同一标准（默认 0.9）')
    parser.add_argument('--same-edge-px', type=int, default=5,
                        help='同类一簇里每个框的左、上、右、下跨度都不超过该像素时，'
                             '自动只留面积最大的框，不进审核（默认 5）')
    parser.add_argument('--border-tol', type=int, default=2,
                        help='判断框是否贴着裁切边界的容差像素（默认 2）')
    parser.add_argument('--noise-tolerance', type=int, default=2,
                        help='四角偏移不超过该像素数时视为取整噪声、不算改动（默认 2）')
    parser.add_argument('--strict-consistency', action='store_true',
                        help='严格模式：同一原图缺陷在多个子图中「一个改了一个没改」也算冲突')
    parser.add_argument('--allow-out-of-scope-add', action='store_true',
                        help='允许自动写入类别不在本数据集 defect_classes 内的新增框')
    parser.add_argument('--workers', type=int, default=32, help='读取 XML 的线程数')

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f'❌ 数据集目录不存在: {dataset_dir}')
        return 1
    if not os.path.isdir(args.original_xml):
        print(f'❌ 原图 XML 目录不存在: {args.original_xml}')
        return 1

    if not args.sub_annotations:
        args.sub_annotations = str(dataset_dir / 'Annotations')
    if not os.path.isdir(args.sub_annotations):
        print(f'❌ 子图 XML 目录不存在: {args.sub_annotations}')
        return 1

    if not args.config:
        args.config = str(Path(__file__).resolve().parent.parent /
                          '1-crop_defect_new' / 'config.yaml')
    if not os.path.exists(args.config):
        print(f'❌ 配置文件不存在: {args.config}')
        return 1

    if not args.crop_mapping:
        found = [str(p) for p in sorted(dataset_dir.glob('crop_mapping*.json'))]
        if not found:
            print(f'❌ 在 {dataset_dir} 下没有找到 crop_mapping*.json，请用 --crop-mapping 指定')
            return 1
        args.crop_mapping = found

    args.ledger = mc.ledger_path_for(args.original_xml, args.ledger)
    dataset_name = dataset_dir.name

    print('=' * 72)
    print('生成变更集（只读，不会修改任何原图文件）')
    print('=' * 72)
    print(f'  数据集      : {dataset_dir}')
    print(f'  部件类型    : {args.component_type}')
    print(f'  子图 XML    : {args.sub_annotations}')
    print(f'  原图 XML    : {args.original_xml}')
    print(f'  裁切记录    : {len(args.crop_mapping)} 个文件')
    for p in args.crop_mapping:
        print(f'                {p}')
    print(f'  历史台账    : {args.ledger}')
    print(f'  输出目录    : {args.output}')
    print('=' * 72)

    configs, _ = mc.load_config(args.config)
    if args.component_type not in configs:
        print(f'❌ 配置文件中没有部件类型 {args.component_type}')
        return 1
    comp = configs[args.component_type]
    print(f'\n缺陷类别 {len(comp.defect_classes)} 个，类别映射 {len(comp.class_mapping)} 条')

    # ---------- 加载裁切记录 ----------
    print('\n[1/6] 加载裁切记录 ...')
    crop_mapping: Dict[str, dict] = {}
    for path in args.crop_mapping:
        raw = mc.load_json(path)
        for sub_xml, entry in raw.items():
            try:
                crop_mapping[sub_xml] = mc.normalize_crop_entry(entry)
            except (KeyError, TypeError, ValueError) as exc:
                print(f'  ⚠️  跳过无法解析的记录 {sub_xml}: {exc}')
        print(f'  {os.path.basename(path)}: {len(raw)} 条')
    print(f'  合计 {len(crop_mapping)} 条')

    # ---------- 扫描现存子图 ----------
    print('\n[2/6] 扫描子图 XML ...')
    present = {f for f in os.listdir(args.sub_annotations) if f.endswith('.xml')}
    matched = sorted(present & set(crop_mapping))
    only_mapping = set(crop_mapping) - present
    only_subxml = present - set(crop_mapping)
    print(f'  现存子图 XML          : {len(present)}')
    print(f'  与裁切记录匹配        : {len(matched)}')
    print(f'  裁切记录有但子图已删除: {len(only_mapping)}  (视为「未参与清洗」，不当作删除 GT)')
    print(f'  子图有但裁切记录缺失  : {len(only_subxml)}  (无法定位原图，跳过)')

    totals = Counter({
        'crop_mapping_entries': len(crop_mapping),
        'sub_xml_present': len(present),
        'matched_pairs': len(matched),
        'mapping_without_subxml': len(only_mapping),
        'subxml_without_mapping': len(only_subxml),
    })

    if not matched:
        print('\n❌ 没有任何子图能与裁切记录对上，请检查 --crop-mapping / --sub-annotations')
        return 1

    # ---------- 读原图 ----------
    print('\n[3/6] 读取涉及到的原图 XML ...')
    referenced = sorted({crop_mapping[s]['original_defect_xml'] for s in matched})
    index = OriginalIndex(args.original_xml, referenced, workers=args.workers)
    totals['original_referenced'] = len(referenced)
    totals['original_missing_file'] = len(index.missing)
    totals['original_unreadable'] = len(index.unreadable)
    print(f'  引用原图 XML : {len(referenced)}')
    print(f'  文件缺失     : {len(index.missing)}')
    print(f'  解析失败     : {len(index.unreadable)}')
    for name in index.missing[:5]:
        print(f'    缺失: {name}')
    for name in index.unreadable[:5]:
        print(f'    解析失败: {name}')

    # ---------- 逐子图 diff ----------
    print('\n[4/6] 逐子图比对 ...')
    ledger_records = mc.read_ledger(args.ledger)
    if ledger_records:
        print(f'  历史台账已有 {len(ledger_records)} 条记录')
    ledger_before = build_ledger_before_index(ledger_records)

    all_proposals: List[dict] = []
    all_reviews: List[dict] = []
    for i, sub_xml in enumerate(matched, 1):
        if i % 20000 == 0:
            print(f'  {i}/{len(matched)}')
        proposals, reviews, stats = build_proposals(
            sub_xml, crop_mapping[sub_xml], args.sub_annotations,
            comp, index, args, dataset_name, ledger_before)
        all_proposals.extend(proposals)
        all_reviews.extend(reviews)
        totals.update(stats)
    print(f'  候选变更 {len([p for p in all_proposals if p["action"] != ACT_KEEP])} 条'
          f'（另有 {len([p for p in all_proposals if p["action"] == ACT_KEEP])} 条未改动记录参与一致性校验）')

    # ---------- 跨子图一致性 ----------
    print('\n[5/6] 跨子图一致性校验 + 新增框去重 + 台账冲突检测 ...')
    all_proposals, reviews, stats = resolve_cross_crop(all_proposals, args)
    all_reviews.extend(reviews)
    totals.update(stats)

    all_proposals, reviews, stats = resolve_added(all_proposals, index, comp, args)
    all_reviews.extend(reviews)
    totals.update(stats)

    all_proposals, reviews, stats = check_ledger(all_proposals, ledger_records, dataset_name)
    all_reviews.extend(reviews)
    totals.update(stats)

    # ---------- 组装变更集 ----------
    print('\n[6/6] 写出结果 ...')
    files: Dict[str, dict] = {}
    for p in all_proposals:
        if p['action'] == ACT_KEEP:
            continue
        xml_name = p['original_xml']
        bucket = files.setdefault(xml_name, {'md5': index.md5.get(xml_name), 'changes': []})
        change = {'action': p['action'], 'sources': p.get('sources', [p['sub_xml']])}
        if p['action'] == ACT_ADD:
            change['new_name'] = p['add_name']
            change['new_bbox'] = list(p['add_bbox'])
        else:
            change['target_name'] = p['target_name_raw']
            change['target_bbox'] = list(p['target_bbox'])
            change['was_clipped'] = p.get('was_clipped', False)
            if p.get('new_name') is not None:
                change['new_name'] = p['new_name']
            if p.get('new_bbox') is not None:
                change['new_bbox'] = list(p['new_bbox'])
        bucket['changes'].append(change)

    changeset = {
        'version': CHANGESET_VERSION,
        'meta': {
            'created_at': mc.now_iso(),
            'dataset': dataset_name,
            'dataset_dir': str(dataset_dir),
            'component_type': args.component_type,
            'sub_annotations': args.sub_annotations,
            'original_xml_dir': os.path.abspath(args.original_xml),
            'crop_mapping': [os.path.abspath(p) for p in args.crop_mapping],
            'config': os.path.abspath(args.config),
            'ledger': os.path.abspath(args.ledger),
            'pair_iou': args.pair_iou,
            'dup_iou': args.dup_iou,
            'same_iou': args.same_iou,
            'contain_ratio': args.contain_ratio,
            'same_edge_px': args.same_edge_px,
            'border_tol': args.border_tol,
            'noise_tolerance': args.noise_tolerance,
            'strict_consistency': args.strict_consistency,
        },
        'files': files,
    }

    for i, item in enumerate(all_reviews):
        item['id'] = i
        item['reason_text'] = REASONS.get(item['reason'], '')

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    mc.dump_json(changeset, str(out_dir / 'changeset.json'))
    mc.dump_json(all_reviews, str(out_dir / 'review_queue.json'))

    review_counter = Counter(item['reason'] for item in all_reviews)
    extra: List[str] = []
    if only_mapping:
        extra.append(f'{len(only_mapping)} 条裁切记录在子图目录里已不存在，'
                     f'按「未参与清洗」处理，没有据此删除任何原图标注。')
    if index.missing:
        extra.append(f'{len(index.missing)} 个原图 XML 在指定目录中缺失。')
    if only_subxml:
        extra.append(f'{len(only_subxml)} 个子图 XML 在裁切记录中找不到，无法定位原图，已跳过。')

    report = write_report(str(out_dir / 'report.txt'), args, comp, totals,
                          review_counter, changeset, extra)
    print()
    print(report)
    print(f'\n输出目录: {out_dir}')
    print(f'  changeset.json    ({len(files)} 个原图 XML 待修改)')
    print(f'  review_queue.json ({len(all_reviews)} 条待人工确认)')
    print('  report.txt')
    return 0


if __name__ == '__main__':
    sys.exit(main())
