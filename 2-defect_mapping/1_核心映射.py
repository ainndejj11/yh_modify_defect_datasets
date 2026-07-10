#!/usr/bin/env python3
"""
缺陷数据集映射脚本 - 步骤1：核心映射逻辑 + 冲突检测

功能：
1. 扫描子图XML，对比crop_mapping.json，识别增删改操作
2. 自动映射确定性操作（未被裁切的缺陷修改、新增框）
3. 检测需要人工清洗的情况（被裁切缺陷的修改、一对多冲突）
4. 生成映射结果和人工审核队列

输入：
- 子图数据目录（包含images和Annotations）
- crop_mapping.json文件路径
- 原图缺陷XML目录

输出：
- 映射后的新原图XML（保存到指定目录）
- mapping_result.json（自动映射记录）
- manual_review_queue.json（人工清洗队列）
- mapping_report.txt（统计报告）
"""

import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import shutil
from datetime import datetime


# ==================== 工具函数 ====================

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 计算交集
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmin >= inter_xmax or inter_ymin >= inter_ymax:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def transform_bbox_to_original(sub_bbox, crop_bbox):
    """将子图坐标转换为原图坐标"""
    crop_xmin, crop_ymin, crop_xmax, crop_ymax = crop_bbox
    sub_xmin, sub_ymin, sub_xmax, sub_ymax = sub_bbox

    original_xmin = sub_xmin + crop_xmin
    original_ymin = sub_ymin + crop_ymin
    original_xmax = sub_xmax + crop_xmin
    original_ymax = sub_ymax + crop_ymin

    return (original_xmin, original_ymin, original_xmax, original_ymax)


def parse_xml_objects(xml_path):
    """解析XML文件，返回对象列表"""
    if not os.path.exists(xml_path):
        return []

    objects = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')

            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            objects.append({
                'name': name,
                'bbox': (xmin, ymin, xmax, ymax)
            })
    except Exception as e:
        print(f"❌ 解析XML失败 {xml_path}: {e}")
        return []

    return objects


def create_xml(filename, width, height, objects, save_path):
    """创建VOC格式的XML文件"""
    root = ET.Element('annotation')

    ET.SubElement(root, 'folder').text = 'JPEGImages'
    ET.SubElement(root, 'filename').text = filename

    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'

    # 添加对象
    for obj in objects:
        obj_elem = ET.SubElement(root, 'object')
        ET.SubElement(obj_elem, 'name').text = obj['name']
        ET.SubElement(obj_elem, 'pose').text = 'Unspecified'
        ET.SubElement(obj_elem, 'truncated').text = '0'
        ET.SubElement(obj_elem, 'difficult').text = '0'

        bndbox = ET.SubElement(obj_elem, 'bndbox')
        xmin, ymin, xmax, ymax = obj['bbox']
        ET.SubElement(bndbox, 'xmin').text = str(int(xmin))
        ET.SubElement(bndbox, 'ymin').text = str(int(ymin))
        ET.SubElement(bndbox, 'xmax').text = str(int(xmax))
        ET.SubElement(bndbox, 'ymax').text = str(int(ymax))

    # 保存XML文件
    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    tree.write(save_path, encoding='utf-8', xml_declaration=True)


def get_image_size_from_xml(xml_path):
    """从XML中获取图像尺寸"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        return width, height
    except:
        return None, None


# ==================== 核心映射逻辑 ====================

class DefectMapper:
    def __init__(self, sub_images_dir, sub_annotations_dir, crop_mapping_path, original_xml_dir, output_dir):
        self.sub_images_dir = Path(sub_images_dir)
        self.sub_annotations_dir = Path(sub_annotations_dir)
        self.crop_mapping_path = crop_mapping_path
        self.original_xml_dir = Path(original_xml_dir)
        self.output_dir = Path(output_dir)

        # 加载crop_mapping.json
        with open(crop_mapping_path, 'r', encoding='utf-8') as f:
            self.crop_mapping = json.load(f)

        # 统计数据
        self.stats = {
            'total_sub_images': 0,
            'auto_mapped': 0,
            'manual_review': 0,
            'operations': {
                'added': 0,
                'deleted': 0,
                'category_modified': 0,
                'bbox_modified': 0,
                'category_and_bbox_modified': 0,
                'unchanged': 0
            }
        }

        # 映射结果和人工审核队列
        self.mapping_results = []
        self.manual_review_queue = []

        # 原图缺陷修改记录（用于检测一对多冲突）
        # key: (original_xml, defect_index), value: [list of modifications]
        self.original_defect_modifications = defaultdict(list)

    def match_defect_in_original(self, crop_info, sub_defect, iou_threshold=0.7):
        """
        在原图缺陷列表中匹配子图缺陷

        返回：(matched_defect_info, defect_index) 或 (None, None) 表示新增
        """
        # 子图坐标转原图坐标
        crop_bbox = crop_info['part_bbox']
        original_bbox = transform_bbox_to_original(sub_defect['bbox'], crop_bbox)

        # 在裁切时保留的缺陷中匹配
        for idx, original_defect in enumerate(crop_info['defects_at_crop_time']):
            # 优先使用裁切后的框进行匹配（因为子图是基于裁切后的框标注的）
            # crop_mapping.json中使用 bbox_in_source 表示裁切后的坐标
            original_defect_bbox = tuple(original_defect['bbox_in_source'])
            iou = calculate_iou(original_bbox, original_defect_bbox)

            if iou > iou_threshold:
                return original_defect, idx

            # 如果裁切后的框匹配失败，尝试用原始框匹配（容错机制）
            if original_defect['was_clipped']:
                original_defect_bbox_original = tuple(original_defect['bbox_in_source_original'])
                iou_original = calculate_iou(original_bbox, original_defect_bbox_original)
                if iou_original > iou_threshold:
                    return original_defect, idx

        return None, None

    def detect_operation(self, sub_defect, matched_original, crop_info):
        """
        检测对缺陷的操作类型

        返回：operation_type, details
        """
        if matched_original is None:
            # 新增框
            return 'added', {'sub_defect': sub_defect}

        # 检查类别是否修改
        category_changed = (sub_defect['name'] != matched_original['name'])

        # 检查尺寸是否修改
        # 子图中的bbox vs 裁切时保留的bbox_in_crop
        bbox_in_crop = tuple(matched_original['bbox_in_crop'])
        bbox_changed = (sub_defect['bbox'] != bbox_in_crop)

        if category_changed and bbox_changed:
            return 'category_and_bbox_modified', {
                'sub_defect': sub_defect,
                'matched_original': matched_original
            }
        elif category_changed:
            return 'category_modified', {
                'sub_defect': sub_defect,
                'matched_original': matched_original
            }
        elif bbox_changed:
            return 'bbox_modified', {
                'sub_defect': sub_defect,
                'matched_original': matched_original
            }
        else:
            return 'unchanged', {
                'sub_defect': sub_defect,
                'matched_original': matched_original
            }

    def should_manual_review(self, operation_type, details, crop_info):
        """
        判断是否需要人工审核

        规则：
        1. 被裁切过的缺陷的任何修改 → 人工审核
        2. 被裁切过的缺陷被删除 → 人工审核
        """
        if operation_type == 'added':
            # 新增框直接自动映射
            return False, None

        if operation_type == 'unchanged':
            # 未修改的直接跳过
            return False, None

        matched_original = details['matched_original']

        # 如果是被裁切过的缺陷
        if matched_original['was_clipped']:
            reason = f"被裁切缺陷的{operation_type}操作"
            return True, reason

        # 未被裁切的缺陷，可以自动映射
        return False, None

    def process_sub_image(self, sub_xml_name):
        """处理单个子图的映射"""
        # 获取crop_mapping信息
        if sub_xml_name not in self.crop_mapping:
            print(f"⚠️  子图 {sub_xml_name} 不在 crop_mapping 中，跳过")
            return

        crop_info = self.crop_mapping[sub_xml_name]
        original_img_path = crop_info['original_img_path']
        original_xml_name = crop_info['original_defect_xml']

        # 解析子图XML
        sub_xml_path = self.sub_annotations_dir / sub_xml_name
        sub_defects = parse_xml_objects(sub_xml_path)

        # 记录子图中已匹配的原图缺陷索引
        matched_original_indices = set()

        # 记录本次处理的操作
        operations = []

        # 遍历子图中的每个缺陷
        for sub_defect in sub_defects:
            # 匹配原图缺陷
            matched_original, defect_idx = self.match_defect_in_original(crop_info, sub_defect)

            # 检测操作类型
            operation_type, details = self.detect_operation(sub_defect, matched_original, crop_info)

            # 判断是否需要人工审核
            need_review, review_reason = self.should_manual_review(operation_type, details, crop_info)

            if need_review:
                # 加入人工审核队列
                self.manual_review_queue.append({
                    'type': 'clipped_defect_modification',
                    'sub_xml': sub_xml_name,
                    'sub_defect': sub_defect,
                    'matched_original': matched_original,
                    'operation_type': operation_type,
                    'reason': review_reason,
                    'original_xml': original_xml_name,
                    'original_img_path': original_img_path,
                    'crop_info': crop_info
                })
                self.stats['manual_review'] += 1
            else:
                # 可以自动映射
                operations.append({
                    'operation_type': operation_type,
                    'details': details,
                    'need_review': False
                })
                self.stats['operations'][operation_type] += 1

            # 记录已处理的原图缺陷索引
            if matched_original is not None and defect_idx is not None:
                matched_original_indices.add(defect_idx)

        # 检查被删除的缺陷
        for idx, original_defect in enumerate(crop_info['defects_at_crop_time']):
            if idx not in matched_original_indices:
                # 缺陷被删除
                if original_defect['was_clipped']:
                    # 被裁切过的删除 → 人工审核
                    self.manual_review_queue.append({
                        'type': 'clipped_defect_deletion',
                        'sub_xml': sub_xml_name,
                        'matched_original': original_defect,
                        'operation_type': 'deleted',
                        'reason': '被裁切缺陷的删除操作',
                        'original_xml': original_xml_name,
                        'original_img_path': original_img_path,
                        'crop_info': crop_info
                    })
                    self.stats['manual_review'] += 1
                else:
                    # 未被裁切的删除 → 自动映射
                    operations.append({
                        'operation_type': 'deleted',
                        'details': {'matched_original': original_defect},
                        'need_review': False
                    })
                    self.stats['operations']['deleted'] += 1

        # 记录映射结果
        self.mapping_results.append({
            'sub_xml': sub_xml_name,
            'original_xml': original_xml_name,
            'operations': operations,
            'crop_info': crop_info
        })

        self.stats['total_sub_images'] += 1

    def detect_conflicts(self):
        """
        检测一对多冲突

        遍历所有映射结果，找出对同一原图缺陷的不同修改
        包括：
        1. 删除冲突（一个删除，一个保留/修改）
        2. 类别冲突（多个子图修改为不同类别）
        3. bbox冲突（多个子图修改为不同bbox）
        4. 新增框冲突（多个子图在相近位置新增不同类别的框）
        """
        # 按原图分组
        original_xml_groups = defaultdict(list)
        for result in self.mapping_results:
            original_xml = result['original_xml']
            original_xml_groups[original_xml].append(result)

        conflicts = []

        for original_xml, results in original_xml_groups.items():
            if len(results) <= 1:
                continue

            # 1. 检测原图缺陷的修改冲突
            # key: defect的唯一标识 (name, bbox_original)
            # value: [(sub_xml, operation_type, details), ...]
            defect_modifications = defaultdict(list)

            # 2. 收集所有新增框（用于检测新增框冲突）
            added_defects = []

            for result in results:
                sub_xml = result['sub_xml']
                crop_info = result['crop_info']

                for op in result['operations']:
                    operation_type = op['operation_type']
                    details = op['details']

                    if operation_type == 'added':
                        # 收集新增框（转换为原图坐标）
                        sub_defect = details['sub_defect']
                        crop_bbox = crop_info['part_bbox']
                        original_bbox = transform_bbox_to_original(sub_defect['bbox'], crop_bbox)

                        added_defects.append({
                            'sub_xml': sub_xml,
                            'name': sub_defect['name'],
                            'bbox': original_bbox,
                            'details': details
                        })
                    else:
                        # 记录修改/删除操作
                        if 'matched_original' in details:
                            matched_original = details['matched_original']
                            # 使用原始框作为唯一标识
                            defect_key = (
                                matched_original['name'],
                                tuple(matched_original['bbox_in_source_original'])
                            )

                            defect_modifications[defect_key].append({
                                'sub_xml': sub_xml,
                                'operation_type': operation_type,
                                'details': details,
                                'crop_info': crop_info
                            })

            # 检测原图缺陷的修改冲突
            for defect_key, modifications in defect_modifications.items():
                if len(modifications) <= 1:
                    continue

                # 检查操作类型
                operation_types = [m['operation_type'] for m in modifications]

                # 判断冲突类型
                conflict_type = None

                # 删除冲突：一个删除，其他保留/修改
                if 'deleted' in operation_types:
                    if len(set(operation_types)) > 1:
                        conflict_type = 'delete_conflict'
                    # 如果全部都是删除，不算冲突（结果一致）

                # 类别冲突：多个子图修改为不同类别
                elif operation_types.count('category_modified') > 0 or operation_types.count('category_and_bbox_modified') > 0:
                    categories = []
                    for m in modifications:
                        if 'sub_defect' in m['details']:
                            categories.append(m['details']['sub_defect']['name'])

                    if len(set(categories)) > 1:
                        conflict_type = 'category_conflict'

                # bbox冲突：多个子图修改为不同bbox
                if operation_types.count('bbox_modified') > 0 or operation_types.count('category_and_bbox_modified') > 0:
                    bboxes = []
                    for m in modifications:
                        if 'sub_defect' in m['details']:
                            # 转换为原图坐标进行比较
                            sub_defect = m['details']['sub_defect']
                            crop_bbox = m['crop_info']['part_bbox']
                            original_bbox = transform_bbox_to_original(sub_defect['bbox'], crop_bbox)
                            bboxes.append(tuple(original_bbox))

                    if len(set(bboxes)) > 1:
                        # 如果已经有类别冲突，标记为同时冲突
                        if conflict_type == 'category_conflict':
                            conflict_type = 'category_and_bbox_conflict'
                        else:
                            conflict_type = 'bbox_conflict'

                if conflict_type:
                    conflicts.append({
                        'type': conflict_type,
                        'original_xml': original_xml,
                        'defect_key': defect_key,
                        'modifications': modifications
                    })

            # 检测新增框冲突：多个子图在相近位置（IoU > 0.5）新增不同类别的框
            if len(added_defects) > 1:
                checked_pairs = set()
                for i, defect1 in enumerate(added_defects):
                    for j, defect2 in enumerate(added_defects):
                        if i >= j:
                            continue

                        pair_key = (i, j)
                        if pair_key in checked_pairs:
                            continue
                        checked_pairs.add(pair_key)

                        # 计算IoU
                        iou = calculate_iou(defect1['bbox'], defect2['bbox'])

                        # 如果位置重叠且类别不同，视为冲突
                        if iou > 0.5 and defect1['name'] != defect2['name']:
                            conflicts.append({
                                'type': 'added_defect_conflict',
                                'original_xml': original_xml,
                                'defect1': defect1,
                                'defect2': defect2,
                                'iou': round(iou, 4)
                            })

        return conflicts

    def apply_auto_mapping(self):
        """
        执行自动映射，直接更新原图XML

        遍历所有映射结果，将自动映射的操作应用到原图XML（直接修改原图XML目录）
        """
        print("\n" + "=" * 70)
        print("🚀 开始执行自动映射")
        print("=" * 70)

        # 按原图分组
        original_xml_groups = defaultdict(list)
        for result in self.mapping_results:
            original_xml = result['original_xml']
            original_xml_groups[original_xml].append(result)

        total_xmls = len(original_xml_groups)
        processed = 0

        for original_xml, results in original_xml_groups.items():
            processed += 1
            print(f"\n[{processed}/{total_xmls}] 处理原图XML: {original_xml}")

            # 解析原图XML
            original_xml_path = self.original_xml_dir / original_xml
            if not original_xml_path.exists():
                print(f"  ⚠️  原图XML不存在: {original_xml_path}，跳过")
                continue

            # 获取原图尺寸和对象列表
            width, height = get_image_size_from_xml(original_xml_path)
            if width is None:
                print(f"  ⚠️  无法获取原图尺寸，跳过")
                continue

            original_objects = parse_xml_objects(original_xml_path)

            # 记录要删除的缺陷索引
            defects_to_delete = set()

            # 记录要修改的缺陷
            defects_to_modify = {}

            # 记录要新增的缺陷
            defects_to_add = []

            # 遍历所有子图的操作
            for result in results:
                crop_info = result['crop_info']

                for op in result['operations']:
                    operation_type = op['operation_type']
                    details = op['details']

                    if operation_type == 'added':
                        # 新增框：子图坐标转原图坐标
                        sub_defect = details['sub_defect']
                        crop_bbox = crop_info['part_bbox']
                        original_bbox = transform_bbox_to_original(sub_defect['bbox'], crop_bbox)

                        # 限制bbox在图像范围内
                        xmin = max(0, int(original_bbox[0]))
                        ymin = max(0, int(original_bbox[1]))
                        xmax = min(width, int(original_bbox[2]))
                        ymax = min(height, int(original_bbox[3]))

                        # 检查bbox有效性
                        if xmax > xmin and ymax > ymin:
                            defects_to_add.append({
                                'name': sub_defect['name'],
                                'bbox': (xmin, ymin, xmax, ymax)
                            })
                        else:
                            print(f"  ⚠️  新增框坐标无效: {sub_defect['name']} {original_bbox}，跳过")

                    elif operation_type == 'deleted':
                        # 删除框：找到原图中对应的缺陷索引
                        matched_original = details['matched_original']
                        original_bbox = tuple(matched_original['bbox_in_source_original'])

                        # 在原图对象列表中查找
                        for idx, obj in enumerate(original_objects):
                            if calculate_iou(obj['bbox'], original_bbox) > 0.9:
                                defects_to_delete.add(idx)
                                break

                    elif operation_type == 'category_modified':
                        # 修改类别
                        matched_original = details['matched_original']
                        sub_defect = details['sub_defect']
                        original_bbox = tuple(matched_original['bbox_in_source_original'])

                        # 在原图对象列表中查找
                        for idx, obj in enumerate(original_objects):
                            if calculate_iou(obj['bbox'], original_bbox) > 0.9:
                                if idx not in defects_to_modify:
                                    defects_to_modify[idx] = {'category': sub_defect['name']}
                                else:
                                    defects_to_modify[idx]['category'] = sub_defect['name']
                                break

                    elif operation_type == 'bbox_modified':
                        # 修改尺寸：子图坐标转原图坐标
                        matched_original = details['matched_original']
                        sub_defect = details['sub_defect']
                        crop_bbox = crop_info['part_bbox']
                        original_bbox_original = tuple(matched_original['bbox_in_source_original'])
                        new_original_bbox = transform_bbox_to_original(sub_defect['bbox'], crop_bbox)

                        # 限制bbox在图像范围内
                        xmin = max(0, int(new_original_bbox[0]))
                        ymin = max(0, int(new_original_bbox[1]))
                        xmax = min(width, int(new_original_bbox[2]))
                        ymax = min(height, int(new_original_bbox[3]))

                        # 检查bbox有效性
                        if xmax <= xmin or ymax <= ymin:
                            print(f"  ⚠️  修改后的bbox无效: {sub_defect['name']} {new_original_bbox}，跳过")
                            continue

                        # 在原图对象列表中查找
                        for idx, obj in enumerate(original_objects):
                            if calculate_iou(obj['bbox'], original_bbox_original) > 0.9:
                                if idx not in defects_to_modify:
                                    defects_to_modify[idx] = {'bbox': (xmin, ymin, xmax, ymax)}
                                else:
                                    defects_to_modify[idx]['bbox'] = (xmin, ymin, xmax, ymax)
                                break

                    elif operation_type == 'category_and_bbox_modified':
                        # 同时修改类别和尺寸
                        matched_original = details['matched_original']
                        sub_defect = details['sub_defect']
                        crop_bbox = crop_info['part_bbox']
                        original_bbox_original = tuple(matched_original['bbox_in_source_original'])
                        new_original_bbox = transform_bbox_to_original(sub_defect['bbox'], crop_bbox)

                        # 限制bbox在图像范围内
                        xmin = max(0, int(new_original_bbox[0]))
                        ymin = max(0, int(new_original_bbox[1]))
                        xmax = min(width, int(new_original_bbox[2]))
                        ymax = min(height, int(new_original_bbox[3]))

                        # 检查bbox有效性
                        if xmax <= xmin or ymax <= ymin:
                            print(f"  ⚠️  修改后的bbox无效: {sub_defect['name']} {new_original_bbox}，跳过")
                            continue

                        # 在原图对象列表中查找
                        for idx, obj in enumerate(original_objects):
                            if calculate_iou(obj['bbox'], original_bbox_original) > 0.9:
                                defects_to_modify[idx] = {
                                    'category': sub_defect['name'],
                                    'bbox': (xmin, ymin, xmax, ymax)
                                }
                                break

            # 应用修改
            new_objects = []
            for idx, obj in enumerate(original_objects):
                if idx in defects_to_delete:
                    continue  # 跳过要删除的

                if idx in defects_to_modify:
                    # 应用修改
                    modification = defects_to_modify[idx]
                    new_obj = {
                        'name': modification.get('category', obj['name']),
                        'bbox': modification.get('bbox', obj['bbox'])
                    }
                    new_objects.append(new_obj)
                else:
                    # 保持不变
                    new_objects.append(obj)

            # 添加新增的缺陷
            new_objects.extend(defects_to_add)

            # 直接保存到原图XML路径（覆盖原文件，因为已经备份）
            # 获取原图文件名
            original_img_name = original_xml.replace('.xml', '.jpg')
            create_xml(original_img_name, width, height, new_objects, original_xml_path)

            print(f"  ✅ 已更新: 删除{len(defects_to_delete)}个, 修改{len(defects_to_modify)}个, 新增{len(defects_to_add)}个")
            self.stats['auto_mapped'] += 1

        print("\n" + "=" * 70)
        print("✅ 自动映射完成")
        print("=" * 70)

    def process_conflicts(self, conflicts):
        """
        处理一对多冲突，加入人工审核队列
        """
        print(f"\n检测到 {len(conflicts)} 个一对多冲突")

        for conflict in conflicts:
            conflict_type = conflict['type']
            original_xml = conflict['original_xml']

            # 处理新增框冲突（结构不同）
            if conflict_type == 'added_defect_conflict':
                defect1 = conflict['defect1']
                defect2 = conflict['defect2']

                conflict_info = {
                    'type': 'one_to_many_conflict',
                    'conflict_type': conflict_type,
                    'original_xml': original_xml,
                    'reason': f"多个子图在重叠位置(IoU={conflict['iou']})新增了不同类别的缺陷",
                    'defects': [
                        {
                            'sub_xml': defect1['sub_xml'],
                            'name': defect1['name'],
                            'bbox': list(defect1['bbox'])
                        },
                        {
                            'sub_xml': defect2['sub_xml'],
                            'name': defect2['name'],
                            'bbox': list(defect2['bbox'])
                        }
                    ]
                }

                self.manual_review_queue.append(conflict_info)
                self.stats['manual_review'] += 1
                continue

            # 处理原图缺陷的修改冲突
            defect_key = conflict['defect_key']
            modifications = conflict['modifications']

            # 构建冲突信息
            conflict_info = {
                'type': 'one_to_many_conflict',
                'conflict_type': conflict_type,
                'original_xml': original_xml,
                'defect_name': defect_key[0],
                'defect_bbox_original': list(defect_key[1]),
                'modifications': []
            }

            for mod in modifications:
                sub_xml = mod['sub_xml']
                operation_type = mod['operation_type']
                details = mod['details']
                crop_info = mod.get('crop_info')

                mod_info = {
                    'sub_xml': sub_xml,
                    'operation_type': operation_type,
                    'crop_info': crop_info
                }

                if 'sub_defect' in details:
                    mod_info['sub_defect'] = details['sub_defect']

                conflict_info['modifications'].append(mod_info)

            self.manual_review_queue.append(conflict_info)
            self.stats['manual_review'] += 1

    def run(self):
        """执行完整的映射流程"""
        print("=" * 70)
        print("🚀 开始缺陷数据集映射 - 步骤1：核心映射逻辑")
        print("=" * 70)

        # 0. 创建输出目录和备份原图XML
        print("\n" + "=" * 70)
        print("📦 准备工作")
        print("=" * 70)

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 输出目录已创建: {self.output_dir}")

        # 备份原图XML目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir_name = f"Annotations_{timestamp}"
        backup_dir = self.output_dir / backup_dir_name

        print(f"📋 正在备份原图XML目录...")
        print(f"   源目录: {self.original_xml_dir}")
        print(f"   备份到: {backup_dir}")

        try:
            shutil.copytree(self.original_xml_dir, backup_dir)
            print(f"✅ 备份完成")
        except Exception as e:
            print(f"❌ 备份失败: {e}")
            print("   映射操作已终止，请检查原图XML目录")
            return

        # 1. 扫描所有子图XML
        if not self.sub_annotations_dir.exists():
            print(f"❌ 子图XML目录不存在: {self.sub_annotations_dir}")
            return

        sub_xml_files = list(self.sub_annotations_dir.glob('*.xml'))
        print(f"\n📁 找到 {len(sub_xml_files)} 个子图XML文件")

        # 2. 处理每个子图
        print("\n" + "=" * 70)
        print("📊 处理子图映射")
        print("=" * 70)

        for idx, xml_file in enumerate(sub_xml_files, 1):
            if idx % 100 == 0:
                print(f"  处理进度: {idx}/{len(sub_xml_files)}")

            sub_xml_name = xml_file.name
            self.process_sub_image(sub_xml_name)

        # 3. 检测一对多冲突
        print("\n" + "=" * 70)
        print("🔍 检测一对多冲突")
        print("=" * 70)

        conflicts = self.detect_conflicts()
        if conflicts:
            self.process_conflicts(conflicts)
            print(f"  ⚠️  发现 {len(conflicts)} 个冲突，已加入人工审核队列")
        else:
            print("  ✅ 未发现冲突")

        # 4. 执行自动映射（直接修改原图XML目录）
        self.apply_auto_mapping()

        # 5. 保存结果
        self.save_results()

        # 6. 打印统计报告
        self.print_report(backup_dir_name)

    def save_results(self):
        """保存映射结果和人工审核队列到输出目录"""
        print("\n" + "=" * 70)
        print("💾 保存结果")
        print("=" * 70)

        # 保存mapping_result.json
        mapping_result_path = self.output_dir / 'mapping_result.json'
        with open(mapping_result_path, 'w', encoding='utf-8') as f:
            json.dump(self.mapping_results, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 映射结果已保存: {mapping_result_path}")

        # 保存manual_review_queue.json
        manual_review_path = self.output_dir / 'manual_review_queue.json'
        with open(manual_review_path, 'w', encoding='utf-8') as f:
            json.dump(self.manual_review_queue, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 人工审核队列已保存: {manual_review_path}")
        print(f"     共 {len(self.manual_review_queue)} 条需要人工审核")

    def print_report(self, backup_dir_name):
        """打印统计报告"""
        print("\n" + "=" * 70)
        print("📊 映射统计报告")
        print("=" * 70)
        print(f"总子图数量: {self.stats['total_sub_images']}")
        print(f"自动映射的原图XML数: {self.stats['auto_mapped']}")
        print(f"需要人工审核的缺陷数: {self.stats['manual_review']}")
        print("\n操作统计:")
        print(f"  新增框: {self.stats['operations']['added']}")
        print(f"  删除框: {self.stats['operations']['deleted']}")
        print(f"  修改类别: {self.stats['operations']['category_modified']}")
        print(f"  修改尺寸: {self.stats['operations']['bbox_modified']}")
        print(f"  未修改: {self.stats['operations']['unchanged']}")
        print("=" * 70)

        # 保存报告到文件
        report_path = self.output_dir / 'mapping_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("缺陷数据集映射统计报告\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"原图XML备份: {backup_dir_name}\n")
            f.write(f"原图XML已直接修改: {self.original_xml_dir}\n\n")
            f.write(f"总子图数量: {self.stats['total_sub_images']}\n")
            f.write(f"自动映射的原图XML数: {self.stats['auto_mapped']}\n")
            f.write(f"需要人工审核的缺陷数: {self.stats['manual_review']}\n\n")
            f.write("操作统计:\n")
            f.write(f"  新增框: {self.stats['operations']['added']}\n")
            f.write(f"  删除框: {self.stats['operations']['deleted']}\n")
            f.write(f"  修改类别: {self.stats['operations']['category_modified']}\n")
            f.write(f"  修改尺寸: {self.stats['operations']['bbox_modified']}\n")
            f.write(f"  未修改: {self.stats['operations']['unchanged']}\n")
            f.write("=" * 70 + "\n")

        print(f"\n📄 报告已保存: {report_path}")


# ==================== 主程序入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description='缺陷数据集映射脚本 - 步骤1：核心映射逻辑 + 冲突检测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python 映射脚本_步骤1_核心映射.py \
    --sub-images /raid/datasets_defect_2026/datasets_val/dx_data_正样本/images \
    --sub-annotations /raid/datasets_defect_2026/datasets_val/dx_data_正样本/Annotations \
    --crop-mapping /raid/datasets_defect_2026/datasets_val/dx_data_正样本/crop_mapping_正样本.json \
    --original-xml /raid/datasets_defect_2026/全图测试集/Annotations \
    --output /raid/datasets_defect_2026/全图测试集/mapping_results

说明:
  1. 原图XML会先备份到输出目录下的 Annotations_年月日_时间 文件夹
  2. 然后直接在 --original-xml 指定的目录中修改XML文件
  3. 所有结果文件（mapping_result.json、manual_review_queue.json、报告等）都保存在输出目录

输出:
  - 输出目录/Annotations_年月日_时间/: 原图XML备份
  - 输出目录/mapping_result.json: 自动映射的详细记录
  - 输出目录/manual_review_queuae.json: 需要人工审核的列表
  - 输出目录/mapping_report.txt: 统计报告
        """
    )

    parser.add_argument(
        '--sub-images',
        required=True,
        help='子图images目录路径'
    )

    parser.add_argument(
        '--sub-annotations',
        required=True,
        help='子图Annotations目录路径（修改后的XML）'
    )

    parser.add_argument(
        '--crop-mapping',
        required=True,
        help='crop_mapping.json文件路径'
    )

    parser.add_argument(
        '--original-xml',
        required=True,
        help='原图缺陷XML目录路径（会直接修改此目录中的文件，修改前会自动备份）'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='输出目录路径（存放所有结果文件和XML备份）'
    )

    args = parser.parse_args()

    # 检查路径
    if not os.path.exists(args.sub_images):
        print(f"❌ 错误: 子图images目录不存在: {args.sub_images}")
        sys.exit(1)

    if not os.path.exists(args.sub_annotations):
        print(f"❌ 错误: 子图Annotations目录不存在: {args.sub_annotations}")
        sys.exit(1)

    if not os.path.exists(args.crop_mapping):
        print(f"❌ 错误: crop_mapping.json文件不存在: {args.crop_mapping}")
        sys.exit(1)

    if not os.path.exists(args.original_xml):
        print(f"❌ 错误: 原图XML目录不存在: {args.original_xml}")
        sys.exit(1)

    # 打印参数
    print("\n" + "=" * 70)
    print("🔧 运行参数")
    print("=" * 70)
    print(f"子图images: {args.sub_images}")
    print(f"子图Annotations: {args.sub_annotations}")
    print(f"crop_mapping: {args.crop_mapping}")
    print(f"原图XML目录: {args.original_xml}")
    print(f"输出目录: {args.output}")
    print("=" * 70 + "\n")

    # 确认操作
    print("⚠️  警告: 此操作将直接修改原图XML目录中的文件！")
    print(f"   原图XML目录: {args.original_xml}")
    print(f"   备份将保存到: {args.output}/Annotations_年月日_时间/")
    response = input("\n是否继续？(yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ 操作已取消")
        sys.exit(0)

    # 创建映射器并执行
    mapper = DefectMapper(
        sub_images_dir=args.sub_images,
        sub_annotations_dir=args.sub_annotations,
        crop_mapping_path=args.crop_mapping,
        original_xml_dir=args.original_xml,
        output_dir=args.output
    )

    mapper.run()

    print("\n" + "=" * 70)
    print("🎉 步骤1完成！")
    print("=" * 70)
    print(f"\n✅ 原图XML已更新: {args.original_xml}")
    print(f"✅ 备份和结果保存在: {args.output}")
    print("\n下一步:")
    print(f"1. 检查 {args.output}/mapping_result.json")
    print(f"2. 检查 {args.output}/manual_review_queue.json")
    print("3. 如果需要人工审核，运行步骤2生成审核数据包")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
