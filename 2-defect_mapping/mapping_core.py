#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mapping_core.py — 缺陷子图标签映射回原图总库的核心库

本模块只提供纯计算函数和显式的 XML 读写辅助，自身不会主动写任何文件。
所有磁盘写入都由 `2_应用变更集.py` 显式触发。

主要能力：
  1. 加载裁切用的 config.yaml（复用裁切程序的语义）
  2. 以「原地修改」方式读写 VOC XML —— 保留 <path>/<source>/<score> 等原有字段、
     tab 缩进、以及是否带 XML 声明
  3. 几何工具：IoU、子图坐标 <-> 原图坐标、clip、贴边判定
  4. 配对式 diff：把「裁切时的框」和「清洗后的框」配对，区分 未改/改/删/增
  5. 变更台账（ledger）读写
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

BBox = Tuple[int, int, int, int]

# 裁切程序支持的部件类型
COMPONENT_TYPES = ['ddx', 'gt', 'jyz', 'gd', 'global', 'jc']

# 数据集目录名 -> 部件类型（用于 --component-type 缺省推断）
DATASET_DIR_TO_TYPE = {
    'dx_data': 'ddx',
    'ddx_data': 'ddx',
    'gt_data': 'gt',
    'jyz_data': 'jyz',
    'gd_data': 'gd',
    'jc_data': 'jc',
    'global_data': 'global',
}


# ============================================================
#  一、配置
# ============================================================

class ComponentConfig:
    """单个部件类型的裁切配置（字段语义与 1-crop_defect_new/统一裁切程序_*.py 保持一致）"""

    def __init__(self, config_dict: dict):
        config_dict = config_dict or {}
        self.component_classes = list(config_dict.get('component_classes') or [])
        self.defect_classes = list(config_dict.get('defect_classes') or [])
        self.overlap_thresholds = dict(config_dict.get('overlap_thresholds') or {})
        self.expand_component = bool(config_dict.get('expand_component', False))
        self.expand_long_ratio = float(config_dict.get('expand_long_ratio', 0.35))
        self.expand_short_ratio = float(config_dict.get('expand_short_ratio', 0.45))
        self.class_mapping = dict(config_dict.get('class_mapping') or {})
        # 裁切时从原图 XML 读取所用的过滤集合：最终类别 ∪ 映射前的原始类别
        self.defect_classes_raw = sorted(set(self.defect_classes) | set(self.class_mapping))

    def map_name(self, raw_name: str) -> str:
        """把原图里的原始类名换算成子图里使用的最终类名"""
        return self.class_mapping.get(raw_name, raw_name)

    def is_ambiguous_target(self, final_name: str) -> bool:
        """该最终类名是否由多个原始类名映射而来（反查有歧义）"""
        return final_name in set(self.class_mapping.values())

    def in_raw_scope(self, raw_name: str) -> bool:
        """原图里的这个类名是否属于本数据集的作用域"""
        return raw_name in self.defect_classes_raw


def load_config(config_path: str) -> Tuple[Dict[str, ComponentConfig], float]:
    """加载 config.yaml，返回 (各部件配置, 默认面积阈值)"""
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}

    configs = {}
    for ctype in COMPONENT_TYPES:
        if ctype in raw:
            configs[ctype] = ComponentConfig(raw[ctype])

    default_threshold = float((raw.get('global_settings') or {}).get('default_threshold', 0.80))
    return configs, default_threshold


# ============================================================
#  二、VOC XML 原地读写
# ============================================================

def _make_parser() -> ET.XMLParser:
    """构造尽量保留注释/处理指令的解析器（Python 3.8+）"""
    try:
        builder = ET.TreeBuilder(insert_comments=True, insert_pis=True)
        return ET.XMLParser(target=builder)
    except TypeError:
        # 老版本不支持 insert_comments，退化为默认解析器
        return ET.XMLParser()


def _text_int(elem: Optional[ET.Element]) -> Optional[int]:
    if elem is None or elem.text is None:
        return None
    try:
        return int(round(float(elem.text.strip())))
    except (TypeError, ValueError):
        return None


class XmlObject:
    """XML 中的一个 <object> 节点的轻量视图"""

    __slots__ = ('elem', 'name', 'bbox', 'index')

    def __init__(self, elem: ET.Element, name: str, bbox: BBox, index: int):
        self.elem = elem
        self.name = name
        self.bbox = bbox
        self.index = index

    def key(self) -> Tuple[str, BBox]:
        return (self.name, self.bbox)

    def __repr__(self):
        return f'XmlObject(#{self.index}, {self.name}, {self.bbox})'


class VocXml:
    """
    一份 VOC XML 的可编辑视图。

    设计原则：
      - 用 ElementTree 解析后**原地**改动 <name> / <bndbox> 文本，或增删 <object> 节点
      - 除被改动的节点外，其余元素、属性、缩进空白全部原样保留
      - 是否输出 XML 声明与原文件保持一致（本项目的原图 XML 均无声明）
      - 新增 <object> 时深拷贝一个现有 object 作为模板，从而继承 <pose>/<score> 等字段
      - 非 ASCII 字符的写法与原文件保持一致：原文件是纯 ASCII（中文写成 &#21495; 这类
        数字实体）就仍旧输出实体，原文件是字面 UTF-8 中文就仍旧输出字面中文。
        本项目两种写法的原图 XML 都存在，不区分的话未改动的字段也会产生字节差异。
    """

    def __init__(self, path: str):
        self.path = str(path)
        with open(self.path, 'rb') as f:
            raw = f.read()
        self.had_declaration = raw.lstrip()[:5] == b'<?xml'
        # 纯 ASCII 原文 -> 用 ascii 编码写出，ElementTree 会把非 ASCII 转回数字实体
        self._write_encoding = 'ascii' if raw.isascii() else 'utf-8'
        # 根元素之后的尾字节（通常是末尾换行）。ElementTree 序列化时会丢弃，需手动补回
        last_gt = raw.rfind(b'>')
        self._trailing = raw[last_gt + 1:] if last_gt >= 0 else b''
        # XML 规范要求解析时把 CRLF 归一化成 LF，写回时要还原成原文件的换行风格
        self._newline = b'\r\n' if b'\r\n' in raw else b'\n'

        self.tree = ET.parse(self.path, parser=_make_parser())
        self.root = self.tree.getroot()
        self._dirty = False

    # ---------- 读 ----------

    def objects(self) -> List[XmlObject]:
        """返回所有坐标合法的 <object>；坐标缺失/非法的节点会被跳过（也不会被改动）"""
        result = []
        for idx, elem in enumerate(self.root.findall('object')):
            name_elem = elem.find('name')
            bnd = elem.find('bndbox')
            if name_elem is None or name_elem.text is None or bnd is None:
                continue
            xmin = _text_int(bnd.find('xmin'))
            ymin = _text_int(bnd.find('ymin'))
            xmax = _text_int(bnd.find('xmax'))
            ymax = _text_int(bnd.find('ymax'))
            if None in (xmin, ymin, xmax, ymax):
                continue
            result.append(XmlObject(elem, name_elem.text.strip(), (xmin, ymin, xmax, ymax), idx))
        return result

    def size(self) -> Tuple[Optional[int], Optional[int]]:
        size_elem = self.root.find('size')
        if size_elem is None:
            return None, None
        return _text_int(size_elem.find('width')), _text_int(size_elem.find('height'))

    # ---------- 改 ----------

    def set_name(self, obj: XmlObject, new_name: str) -> None:
        """就地改类名。绝不删旧框再加新框，避免下次裁切出重复框。"""
        obj.elem.find('name').text = new_name
        obj.name = new_name
        self._dirty = True

    def set_bbox(self, obj: XmlObject, new_bbox: BBox) -> None:
        """就地改坐标"""
        bnd = obj.elem.find('bndbox')
        for tag, value in zip(('xmin', 'ymin', 'xmax', 'ymax'), new_bbox):
            child = bnd.find(tag)
            if child is None:
                child = ET.SubElement(bnd, tag)
            child.text = str(int(value))
        obj.bbox = tuple(int(v) for v in new_bbox)
        self._dirty = True

    def remove(self, obj: XmlObject) -> None:
        """删除一个 <object> 节点，并把它的 tail 交给前一个兄弟，避免留下空行"""
        children = list(self.root)
        try:
            pos = children.index(obj.elem)
        except ValueError:
            return
        if pos == len(children) - 1 and pos > 0:
            # 删的是最后一个元素：把它的 tail（收尾缩进）传给新的最后一个
            children[pos - 1].tail = obj.elem.tail
        self.root.remove(obj.elem)
        self._dirty = True

    def add_object(self, name: str, bbox: BBox) -> ET.Element:
        """新增一个 <object>，尽量沿用同文件既有 object 的字段结构与缩进"""
        template = None
        for elem in self.root.findall('object'):
            if elem.find('bndbox') is not None and elem.find('name') is not None:
                template = elem
                break

        if template is not None:
            new_elem = copy.deepcopy(template)
        else:
            new_elem = self._build_object_from_scratch()

        new_elem.find('name').text = name
        bnd = new_elem.find('bndbox')
        for tag, value in zip(('xmin', 'ymin', 'xmax', 'ymax'), bbox):
            child = bnd.find(tag)
            if child is None:
                child = ET.SubElement(bnd, tag)
            child.text = str(int(value))

        children = list(self.root)
        if children:
            # 新节点接在最后：原最后一个元素改用「元素间缩进」，新节点接管「收尾缩进」
            new_elem.tail = children[-1].tail
            children[-1].tail = self.root.text if self.root.text else '\n\t'
        else:
            new_elem.tail = '\n'
            if not self.root.text:
                self.root.text = '\n\t'

        self.root.append(new_elem)
        self._dirty = True
        return new_elem

    def _build_object_from_scratch(self) -> ET.Element:
        """文件里一个 object 都没有时，按本项目 XML 的缩进风格手工构造一个"""
        base = self.root.text if self.root.text else '\n\t'
        inner = base + '\t'
        deeper = inner + '\t'

        obj = ET.Element('object')
        obj.text = inner
        for tag, value in (('name', ''), ('pose', 'Unspecified'),
                           ('truncated', '0'), ('difficult', '0')):
            child = ET.SubElement(obj, tag)
            child.text = value
            child.tail = inner

        bnd = ET.SubElement(obj, 'bndbox')
        bnd.text = deeper
        bnd.tail = base
        for tag in ('xmin', 'ymin', 'xmax', 'ymax'):
            child = ET.SubElement(bnd, tag)
            child.text = '0'
            child.tail = deeper
        list(bnd)[-1].tail = inner
        return obj

    # ---------- 写 ----------

    @property
    def dirty(self) -> bool:
        return self._dirty

    def serialize(self) -> bytes:
        """序列化成字节，并还原原文件的编码风格、换行风格与末尾字节"""
        buf = io.BytesIO()
        self.tree.write(buf, encoding=self._write_encoding,
                        xml_declaration=self.had_declaration)
        data = buf.getvalue() + self._trailing
        if self._newline != b'\n':
            data = data.replace(b'\r\n', b'\n').replace(b'\n', self._newline)
        return data

    def save(self, path: Optional[str] = None) -> None:
        """原子写入：先写临时文件再 os.replace，避免中途崩溃损坏原文件"""
        target = str(path) if path else self.path
        target_dir = os.path.dirname(os.path.abspath(target)) or '.'
        os.makedirs(target_dir, exist_ok=True)

        data = self.serialize()
        fd, tmp_path = tempfile.mkstemp(prefix='.mapping_tmp_', suffix='.xml', dir=target_dir)
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def read_objects(xml_path: str) -> Optional[List[Tuple[str, BBox]]]:
    """轻量解析：只取 (类名, 坐标) 列表。解析失败返回 None（与「空标注」区分开）"""
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None

    result = []
    for elem in root.findall('object'):
        name_elem = elem.find('name')
        bnd = elem.find('bndbox')
        if name_elem is None or name_elem.text is None or bnd is None:
            continue
        xmin = _text_int(bnd.find('xmin'))
        ymin = _text_int(bnd.find('ymin'))
        xmax = _text_int(bnd.find('xmax'))
        ymax = _text_int(bnd.find('ymax'))
        if None in (xmin, ymin, xmax, ymax):
            continue
        result.append((name_elem.text.strip(), (xmin, ymin, xmax, ymax)))
    return result


def md5_of(path: str) -> str:
    """文件 md5，用于 plan -> apply 之间的防篡改校验"""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
#  三、几何工具
# ============================================================

def iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    """标准 IoU"""
    inter_w = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    inter_h = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def cluster_indices(n: int, should_merge) -> List[List[int]]:
    """按 should_merge(i, j) 把 0..n-1 分成连通分量，组内保持首次出现顺序。"""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if should_merge(i, j):
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def crop_to_source(bbox_in_crop: Sequence[int], part_bbox: Sequence[int]) -> BBox:
    """子图坐标 -> 原图坐标（加上部件框左上角偏移）"""
    dx, dy = int(part_bbox[0]), int(part_bbox[1])
    return (int(bbox_in_crop[0]) + dx, int(bbox_in_crop[1]) + dy,
            int(bbox_in_crop[2]) + dx, int(bbox_in_crop[3]) + dy)


def clip_to_image(bbox: Sequence[int], width: Optional[int], height: Optional[int]) -> BBox:
    """把框限制在图像范围内；宽高未知时只保证非负"""
    xmin, ymin, xmax, ymax = (int(v) for v in bbox)
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    if width:
        xmin = min(xmin, width)
        xmax = min(xmax, width)
    if height:
        ymin = min(ymin, height)
        ymax = min(ymax, height)
    return (xmin, ymin, max(xmin, xmax), max(ymin, ymax))


def is_valid_bbox(bbox: Sequence[int], min_side: int = 1) -> bool:
    return (bbox[2] - bbox[0]) >= min_side and (bbox[3] - bbox[1]) >= min_side


def touches_crop_border(bbox_in_crop: Sequence[int], part_bbox: Sequence[int],
                        tolerance: int = 2) -> bool:
    """
    判断子图里的框是否仍贴着裁切边界。

    贴边意味着这个框在原图里很可能仍然是被截断的（部件框外还有一截），
    此时不能直接用子图坐标替换原图完整框，必须交人工确认。
    """
    crop_w = int(part_bbox[2]) - int(part_bbox[0])
    crop_h = int(part_bbox[3]) - int(part_bbox[1])
    xmin, ymin, xmax, ymax = (int(v) for v in bbox_in_crop)
    return (xmin <= tolerance or ymin <= tolerance
            or xmax >= crop_w - tolerance or ymax >= crop_h - tolerance)


# ============================================================
#  四、配对式 diff
# ============================================================

# diff 结果的四种类型
KIND_UNCHANGED = 'unchanged'
KIND_MODIFIED = 'modified'
KIND_DELETED = 'deleted'
KIND_ADDED = 'added'


def pair_diff(old_boxes: Sequence[Tuple[str, BBox]],
              new_boxes: Sequence[Tuple[str, BBox]],
              iou_threshold: float = 0.5) -> List[dict]:
    """
    把「裁切时写出的框」和「人工清洗后的框」配对。

    这是整个映射的关键：如果只做精确匹配、把剩下的一律算成增删，
    那么人工把框挪动 1 个像素就会被判成「删一个 + 加一个」；
    对 was_clipped 的框来说，这会把原图里的完整框删掉、换成一个截断的小框。

    步骤：
      1) (类名, 坐标) 完全一致 -> unchanged
      2) 剩下的按 IoU 降序贪心配对（IoU >= 阈值）-> modified，并标出改的是类别还是坐标
      3) 仍未配上的旧框 -> deleted
      4) 仍未配上的新框 -> added

    返回的每条记录都带 old_index / new_index，便于调用方回查原始信息。
    """
    used_old, used_new = set(), set()
    results: List[dict] = []

    # ---- 1) 精确匹配 ----
    exact_pool: Dict[Tuple[str, BBox], List[int]] = {}
    for j, item in enumerate(new_boxes):
        exact_pool.setdefault((item[0], tuple(item[1])), []).append(j)

    for i, (name, bbox) in enumerate(old_boxes):
        pool = exact_pool.get((name, tuple(bbox)))
        if pool:
            j = pool.pop()
            used_old.add(i)
            used_new.add(j)
            results.append({'kind': KIND_UNCHANGED, 'old_index': i, 'new_index': j,
                            'cat_changed': False, 'bbox_changed': False})

    # ---- 2) IoU 贪心配对 ----
    candidates = []
    for i, (old_name, old_bbox) in enumerate(old_boxes):
        if i in used_old:
            continue
        for j, (new_name, new_bbox) in enumerate(new_boxes):
            if j in used_new:
                continue
            score = iou(old_bbox, new_bbox)
            if score >= iou_threshold:
                # IoU 主导，同类名仅作为极小的加权用于打破平局
                candidates.append((score + (1e-6 if old_name == new_name else 0.0), score, i, j))

    candidates.sort(reverse=True)
    for _, score, i, j in candidates:
        if i in used_old or j in used_new:
            continue
        used_old.add(i)
        used_new.add(j)
        cat_changed = old_boxes[i][0] != new_boxes[j][0]
        bbox_changed = tuple(old_boxes[i][1]) != tuple(new_boxes[j][1])
        results.append({'kind': KIND_MODIFIED, 'old_index': i, 'new_index': j,
                        'cat_changed': cat_changed, 'bbox_changed': bbox_changed,
                        'pair_iou': round(score, 4)})

    # ---- 3) / 4) 剩余 ----
    for i in range(len(old_boxes)):
        if i not in used_old:
            results.append({'kind': KIND_DELETED, 'old_index': i, 'new_index': None})
    for j in range(len(new_boxes)):
        if j not in used_new:
            results.append({'kind': KIND_ADDED, 'old_index': None, 'new_index': j})

    return results


def merge_clipped_bbox(target_bbox: Sequence[int], new_bbox_in_crop: Sequence[int],
                       part_bbox: Sequence[int], tolerance: int = 2) -> BBox:
    """
    把「人工在子图里调整过的框」与「原图里的完整框」合并。

    人工只能看到部件框以内的那一截，所以：
      - 贴着裁切边界的那几条边，说明框在原图里还往外延伸，保留原图的坐标
      - 没贴边的那几条边，是人工真正调整过的，采用新坐标
    """
    naive = crop_to_source(new_bbox_in_crop, part_bbox)
    crop_w = int(part_bbox[2]) - int(part_bbox[0])
    crop_h = int(part_bbox[3]) - int(part_bbox[1])
    cxmin, cymin, cxmax, cymax = (int(v) for v in new_bbox_in_crop)
    xmin, ymin, xmax, ymax = naive

    if cxmin <= tolerance:
        xmin = min(int(target_bbox[0]), xmin)
    if cymin <= tolerance:
        ymin = min(int(target_bbox[1]), ymin)
    if cxmax >= crop_w - tolerance:
        xmax = max(int(target_bbox[2]), xmax)
    if cymax >= crop_h - tolerance:
        ymax = max(int(target_bbox[3]), ymax)

    return (xmin, ymin, xmax, ymax)


def max_corner_shift(box_a: Sequence[int], box_b: Sequence[int]) -> int:
    """两个框四个角坐标的最大偏移量，用来识别标注工具取整造成的 1~2 像素噪声"""
    return max(abs(int(a) - int(b)) for a, b in zip(box_a, box_b))


# ============================================================
#  五、crop_mapping 归一化
# ============================================================

def normalize_crop_entry(entry: dict) -> dict:
    """
    把 crop_mapping 的一条记录归一化成统一结构，抹平「裁切子图」与「全局全图」的差异。

    全局条目（jc / global，is_global=true）本质是「恒等裁切」：
    子图就是原图，所以 part_bbox 视作 (0,0,W,H) 的等价物 —— 偏移为 0，
    bbox_in_crop == bbox_in_source_original，且不存在 was_clipped。

    返回:
      {
        'original_defect_xml': str,
        'part_bbox': [x1,y1,x2,y2],
        'is_global': bool,
        'is_negative_sample': bool,
        'defects': [ {name, bbox_in_crop, bbox_in_source, bbox_in_source_original,
                      was_clipped, overlap_ratio}, ... ]
      }
    """
    is_global = bool(entry.get('is_global'))
    is_negative = bool(entry.get('is_negative_sample'))

    if is_global:
        defects = []
        for d in entry.get('defects') or []:
            bbox = [int(v) for v in d['bbox']]
            defects.append({
                'name': d['name'],
                'bbox_in_crop': bbox,
                'bbox_in_source': bbox,
                'bbox_in_source_original': bbox,
                'was_clipped': False,
                'overlap_ratio': 1.0,
            })
        part_bbox = [0, 0, 0, 0]
    else:
        defects = []
        for d in entry.get('defects_at_crop_time') or []:
            defects.append({
                'name': d['name'],
                'bbox_in_crop': [int(v) for v in d['bbox_in_crop']],
                'bbox_in_source': [int(v) for v in d['bbox_in_source']],
                'bbox_in_source_original': [int(v) for v in d['bbox_in_source_original']],
                'was_clipped': bool(d.get('was_clipped', False)),
                'overlap_ratio': d.get('overlap_ratio'),
            })
        part_bbox = [int(v) for v in entry['part_bbox']]

    return {
        'original_defect_xml': entry['original_defect_xml'],
        'part_bbox': part_bbox,
        'is_global': is_global,
        'is_negative_sample': is_negative,
        'defects': defects,
    }


# ============================================================
#  六、变更台账（ledger）
# ============================================================

LEDGER_FILENAME = 'mapping_ledger.jsonl'


def ledger_path_for(original_xml_dir: str, explicit: Optional[str] = None) -> str:
    """台账默认放在原图 XML 目录的同级，跟着原图库走，这样换机器也不会丢"""
    if explicit:
        return str(explicit)
    return str(Path(original_xml_dir).parent / LEDGER_FILENAME)


def read_ledger(path: str) -> List[dict]:
    """读取历史变更台账；文件不存在返回空列表。单行损坏只跳过该行。"""
    records: List[dict] = []
    if not path or not os.path.exists(path):
        return records
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f'⚠️  台账第 {line_no} 行无法解析，已跳过: {path}')
    return records


def append_ledger(path: str, records: Iterable[dict]) -> int:
    """追加写台账（jsonl，只追加不覆盖，天然抗并发与中断）"""
    records = list(records)
    if not records:
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return len(records)


def write_ledger(path: str, records: Iterable[dict]) -> int:
    """整文件原子重写台账。回滚撤回记录时用；正常 apply 仍走 append_ledger。"""
    records = list(records)
    dst_dir = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(dst_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.ledger_tmp_', suffix='.jsonl', dir=dst_dir)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return len(records)


def ledger_key(original_xml: str, target_name: str, target_bbox: Sequence[int]) -> str:
    """台账里定位「原图某一个具体的框」的唯一键"""
    return f'{original_xml}|{target_name}|{",".join(str(int(v)) for v in target_bbox)}'


def norm_path(path: Optional[str]) -> str:
    if not path:
        return ''
    return os.path.normpath(os.path.abspath(path))


def manifest_xml_set(manifest: dict) -> set:
    return {item['xml'] for item in manifest.get('files', [])}


def match_ledger_indices_for_rollback(
        records: List[dict],
        manifest: dict,
) -> Tuple[List[int], str]:
    """
    找出「本次备份对应那一次 apply」写入的台账行。
    用 manifest 里的 applied_at + changeset 对齐，绝不按整个数据集整段删除。
    """
    xml_set = manifest_xml_set(manifest)
    dataset = manifest.get('dataset')
    applied_at = manifest.get('applied_at')
    cs = norm_path(manifest.get('changeset') or '')

    indices: List[int] = []
    for i, rec in enumerate(records):
        if rec.get('original_xml') not in xml_set:
            continue
        if dataset and rec.get('dataset') not in (None, '', dataset):
            continue
        if rec.get('applied_at') != applied_at:
            continue
        if norm_path(rec.get('changeset') or '') != cs:
            continue
        indices.append(i)
    return indices, 'applied_at+changeset'


def later_ledger_indices(
        records: List[dict],
        this_indices: Sequence[int],
        manifest: dict,
) -> List[int]:
    """
    本次备份涉及的 XML 上，比这次 apply 更晚的台账行。
    回滚会整文件还原 XML，这些更晚的写入会被一起盖掉，默认必须先回滚它们。

    排序以 jsonl 行序为准（只追加，后写入的一定在后面），不用时间戳比较。
    这样同一秒内连续 apply 两次也不会把更早的那批误判成「更晚」。
    """
    xml_set = manifest_xml_set(manifest)
    this_set = set(this_indices)
    later: List[int] = []

    if this_indices:
        last_this = max(this_indices)
        for i, rec in enumerate(records):
            if i in this_set:
                continue
            if rec.get('original_xml') not in xml_set:
                continue
            if i > last_this:
                later.append(i)
        return later

    # 本次写入已从台账消失时，退回用时间戳 / 变更集路径判断是否还有更晚的
    cutoff = manifest.get('applied_at') or ''
    this_cs = norm_path(manifest.get('changeset') or '')
    if not cutoff and not this_cs:
        return []
    for i, rec in enumerate(records):
        if rec.get('original_xml') not in xml_set:
            continue
        t = rec.get('applied_at') or ''
        cs = norm_path(rec.get('changeset') or '')
        if cutoff and t > cutoff:
            later.append(i)
        elif cutoff and t == cutoff and this_cs and cs and cs != this_cs:
            later.append(i)
    return later


# ============================================================
#  七、杂项
# ============================================================

def now_stamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


def guess_component_type(dataset_dir: str) -> Optional[str]:
    """根据数据集目录名推断部件类型，推断不出返回 None"""
    return DATASET_DIR_TO_TYPE.get(Path(dataset_dir).name)


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def dump_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
