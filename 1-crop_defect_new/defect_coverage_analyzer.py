#!/usr/bin/env python3
"""
缺陷漏裁分析共享模块

提供裁切结果与原图缺陷标注的对比分析能力，用于统计原图中未被裁切到的缺陷目标
并按原因归集（部件未覆盖、缺陷不完整/未达阈值、数据异常）。

本模块同时被以下两个入口使用：
1. 统一裁切程序_正样本.py：在裁切流程中内联调用，输出基础漏裁统计报告。
2. 分析漏裁.py：独立运行的全库审计脚本，输出 JSON/CSV/TXT 报告及可视化。
"""

import os
import sys
import json
import csv
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont

import yaml
from tqdm import tqdm


# ==================== 配置加载 ====================

class ComponentConfig:
    """单个部件类型的配置"""
    def __init__(self, config_dict):
        self.component_classes = config_dict.get('component_classes', [])
        self.defect_classes = config_dict.get('defect_classes', [])
        self.overlap_thresholds = config_dict.get('overlap_thresholds', {})
        self.expand_component = config_dict.get('expand_component', False)
        self.expand_long_ratio = config_dict.get('expand_long_ratio', 0.35)
        self.expand_short_ratio = config_dict.get('expand_short_ratio', 0.45)
        self.class_mapping = config_dict.get('class_mapping', {})
        # 读取XML时的过滤集合：需要同时包含映射前的原始类别名，
        # 否则 class_mapping 的 key（如 xcxj_lsqdp）会在读取阶段就被过滤掉，映射永远不会生效
        self.defect_classes_raw = list(set(self.defect_classes) | set(self.class_mapping.keys()))


def load_config(config_path):
    """加载配置文件，返回 (component_configs, default_threshold)"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    component_configs = {}
    for component_type in ['ddx', 'gt', 'jyz', 'gd', 'global', 'jc']:
        if component_type in config:
            component_configs[component_type] = ComponentConfig(config[component_type])

    global_config = config.get('global_settings', {})
    default_threshold = global_config.get('default_threshold', 0.80)

    return component_configs, default_threshold


# ==================== 图像索引 ====================

def load_image_index(pkl_path):
    """加载图片路径索引（pkl 文件方式）"""
    index = {}
    with open(pkl_path, 'rb') as f:
        while True:
            try:
                part = pickle.load(f)
                index.update(part)
            except EOFError:
                break
    return index


def build_image_index(images_dir):
    """从图片文件夹构建图片索引（文件夹扫描方式）

    Returns:
        dict: {图片文件名: 完整路径}
    """
    index = {}
    images_dir = Path(images_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG'}

    for img_path in images_dir.rglob('*'):
        if img_path.is_file() and img_path.suffix in image_extensions:
            index[img_path.name] = str(img_path)

    return index


# ==================== 边界框工具 ====================

def expand_bbox(bbox, img_width, img_height, long_ratio=0.35, short_ratio=0.45):
    """根据长边短边自适应扩充边界框"""
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin

    if width >= height:
        expand_w = width * long_ratio
        expand_h = height * short_ratio
    else:
        expand_w = width * short_ratio
        expand_h = height * long_ratio

    xmin = max(0, int(xmin - expand_w))
    ymin = max(0, int(ymin - expand_h))
    xmax = min(img_width, int(xmax + expand_w))
    ymax = min(img_height, int(ymax + expand_h))

    return (xmin, ymin, xmax, ymax)


def calculate_overlap_ratio(parent_box, child_box):
    """
    计算子框与父框交集面积占子框总面积的比例。
    返回: (比例, 裁剪后的子框坐标) 或 (0, None) 如果无交集。
    """
    pxmin, pymin, pxmax, pymax = parent_box
    cxmin, cymin, cxmax, cymax = child_box

    inter_xmin = max(pxmin, cxmin)
    inter_ymin = max(pymin, cymin)
    inter_xmax = min(pxmax, cxmax)
    inter_ymax = min(pymax, cymax)

    if inter_xmin >= inter_xmax or inter_ymin >= inter_ymax:
        return 0, None

    child_area = (cxmax - cxmin) * (cymax - cymin)
    if child_area <= 0:
        return 0, None

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    ratio = inter_area / child_area

    clipped_box = (inter_xmin, inter_ymin, inter_xmax, inter_ymax)
    return ratio, clipped_box


def calculate_iou(box1, box2):
    """计算两个边界框的 IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

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


def is_valid_bbox(bbox):
    """检查 bbox 是否有效"""
    xmin, ymin, xmax, ymax = bbox
    return xmin < xmax and ymin < ymax


# ==================== XML 工具 ====================

def get_objects_from_xml(xml_path, target_classes=None):
    """从 XML 中提取目标对象，跳过无效 bbox"""
    objs = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            try:
                name = obj.find('name').text
                if target_classes and name not in target_classes:
                    continue
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue

                xmin_elem = bndbox.find('xmin')
                ymin_elem = bndbox.find('ymin')
                xmax_elem = bndbox.find('xmax')
                ymax_elem = bndbox.find('ymax')

                if None in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]:
                    continue

                xmin = int(xmin_elem.text)
                ymin = int(ymin_elem.text)
                xmax = int(xmax_elem.text)
                ymax = int(ymax_elem.text)
                bbox = (xmin, ymin, xmax, ymax)

                if not is_valid_bbox(bbox):
                    print(f"⚠️  跳过无效 bbox: {xml_path} / {name} / {bbox}")
                    continue

                objs.append({'name': name, 'bbox': bbox})
            except (ValueError, AttributeError) as e:
                print(f"⚠️  XML {xml_path} 解析对象时出错: {e}，跳过该对象")
                continue
    except Exception as e:
        print(f"❌ 解析XML文件失败 {xml_path}: {e}")
    return objs


def get_image_size_from_xml(xml_path):
    """从 XML 中获取图像尺寸"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        return width, height
    except Exception:
        return None, None


def _indent_xml(elem, level=0):
    """兼容 Python 3.8 的 XML 缩进辅助函数"""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            _indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def create_xml(filename, width, height, objects, save_path):
    """创建 VOC 格式的 XML 文件"""
    root = ET.Element('annotation')

    ET.SubElement(root, 'folder').text = 'JPEGImages'
    ET.SubElement(root, 'filename').text = filename

    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'

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

    tree = ET.ElementTree(root)
    if hasattr(ET, 'indent'):
        ET.indent(tree, space='  ')
    else:
        _indent_xml(root)
    tree.write(save_path, encoding='utf-8', xml_declaration=True)


# ==================== 类别映射 ====================

def apply_class_mapping(defects, class_mapping):
    """对缺陷列表应用类别映射（原地修改 name）"""
    if not class_mapping:
        return defects
    for defect in defects:
        if defect['name'] in class_mapping:
            defect['name'] = class_mapping[defect['name']]
    return defects


# ==================== crop_mapping 索引 ====================

def find_image_path_for_xml(xml_name, image_index, images_dir, use_pkl_index):
    """
    根据 XML 文件名查找对应的图片路径，支持多种扩展名。

    Args:
        xml_name: XML 文件名（如 IMG_001.xml）
        image_index: 图片索引字典
        images_dir: 原图目录
        use_pkl_index: 是否使用 pkl 索引

    Returns:
        str or None: 图片完整路径
    """
    if not image_index:
        return None

    base_name = os.path.splitext(xml_name)[0]

    for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']:
        img_name = base_name + ext

        if use_pkl_index:
            rel_path = image_index.get(img_name)
            if rel_path:
                full_path = os.path.join(images_dir, rel_path)
                if os.path.exists(full_path):
                    return full_path
        else:
            full_path = image_index.get(img_name)
            if full_path and os.path.exists(full_path):
                return full_path

    return None


def build_crop_mapping_index(crop_mapping):
    """将 crop_mapping（按子图 XML 名索引）转换为按原图 XML 名索引"""
    index = defaultdict(list)
    for sub_xml_name, crop_info in crop_mapping.items():
        original_xml = crop_info.get('original_defect_xml')
        if original_xml:
            index[original_xml].append(crop_info)
    return index


# ==================== 缺陷与 crop_mapping 匹配 ====================

def match_defect_to_crop_info(defect_name, defect_bbox, crop_info_list):
    """
    将单个原图缺陷与 crop_mapping 中该图的所有 crop_info 进行三级匹配。

    匹配顺序：
    1. 精确匹配：类别 + bbox_in_source_original 完全一致
    2. IoU 兜底：类别相同，bbox_in_source_original 与原图 defect bbox 的 IoU >= 0.9
    3. 坐标兜底：类别相同，bbox_in_source_clipped 与原图 defect bbox 至少有 2 个坐标相同

    返回：dict 或 None
    """
    defect_bbox = tuple(defect_bbox)

    # Level 1: 精确匹配
    for crop_info in crop_info_list:
        for d in crop_info.get('defects_at_crop_time', []):
            if d.get('name') == defect_name:
                if tuple(d.get('bbox_in_source_original', [])) == defect_bbox:
                    return {
                        'matched_crop': crop_info,
                        'matched_defect': d,
                        'match_method': 'exact'
                    }

    # Level 2: IoU 兜底
    for crop_info in crop_info_list:
        for d in crop_info.get('defects_at_crop_time', []):
            if d.get('name') == defect_name:
                original_bbox = d.get('bbox_in_source_original')
                if original_bbox and calculate_iou(defect_bbox, tuple(original_bbox)) >= 0.9:
                    return {
                        'matched_crop': crop_info,
                        'matched_defect': d,
                        'match_method': 'iou_fallback'
                    }

    # Level 3: 两坐标兜底（基于裁切后坐标）
    for crop_info in crop_info_list:
        for d in crop_info.get('defects_at_crop_time', []):
            if d.get('name') == defect_name:
                clipped_bbox = d.get('bbox_in_source_clipped') or d.get('bbox_in_source')
                if clipped_bbox:
                    clipped_bbox = tuple(clipped_bbox)
                    matching_coords = sum(1 for i in range(4) if defect_bbox[i] == clipped_bbox[i])
                    if matching_coords >= 2:
                        return {
                            'matched_crop': crop_info,
                            'matched_defect': d,
                            'match_method': 'coordinate_fallback'
                        }

    return None


# ==================== 单图漏裁分析 ====================

def analyze_single_image(img_name, all_defects, components, crop_info_list,
                         config, default_threshold, img_width, img_height):
    """
    分析单张原图的漏裁情况。

    Args:
        img_name: 图片文件名
        all_defects: 该图所有缺陷（已应用 class_mapping）
        components: 该图所有部件框（原始未扩充）
        crop_info_list: 该图对应的所有 crop_info 列表
        config: ComponentConfig
        default_threshold: 默认面积阈值
        img_width, img_height: 图像尺寸

    Returns:
        list: 该图所有漏裁缺陷的详细信息
    """
    missed_defects = []

    # 扩充部件框
    expanded_components = []
    for comp in components:
        bbox = comp['bbox']
        if config.expand_component:
            bbox = expand_bbox(
                bbox, img_width, img_height,
                long_ratio=config.expand_long_ratio,
                short_ratio=config.expand_short_ratio
            )
        expanded_components.append({'name': comp['name'], 'bbox': bbox})

    for defect in all_defects:
        defect_name = defect['name']
        defect_bbox = defect['bbox']

        # 获取该类别的阈值
        threshold = config.overlap_thresholds.get(defect_name, default_threshold)

        # 计算与所有部件框的最大重叠比
        max_overlap_ratio = 0.0
        best_component = None
        all_overlaps = []

        for comp in expanded_components:
            ratio, clipped_bbox = calculate_overlap_ratio(comp['bbox'], defect_bbox)
            if ratio > 0:
                overlap_info = {
                    'class': comp['name'],
                    'bbox': list(comp['bbox']),
                    'overlap_ratio': round(ratio, 4)
                }
                all_overlaps.append(overlap_info)

                if ratio > max_overlap_ratio:
                    max_overlap_ratio = ratio
                    best_component = overlap_info

        # 确定原因
        if max_overlap_ratio == 0:
            reason = 'component_not_covered'
            notes = '未与任何部件框相交'
        elif max_overlap_ratio < threshold:
            reason = 'incomplete_or_filtered'
            notes = f'最大重叠比 {max_overlap_ratio:.4f} 低于阈值 {threshold}'
        else:
            # 理论上应该被裁切，检查是否在 crop_mapping 中
            match_result = match_defect_to_crop_info(defect_name, defect_bbox, crop_info_list)
            if match_result:
                # 正常裁切，不属于漏裁
                continue
            else:
                reason = 'data_inconsistency'
                notes = '按理应被裁切，但未在 crop_mapping 中匹配到记录'

        missed_defects.append({
            'original_img_name': img_name,
            'original_xml_name': os.path.splitext(img_name)[0] + '.xml',
            'original_img_path': None,  # 由上层填充
            'defect_name': defect_name,
            'defect_bbox_original': list(defect_bbox),
            'reason': reason,
            'max_overlap_ratio': round(max_overlap_ratio, 4),
            'threshold': threshold,
            'best_component_bbox': best_component['bbox'] if best_component else None,
            'best_component_class': best_component['class'] if best_component else None,
            'all_overlapping_components': all_overlaps,
            'matched_crop': None,
            'match_method': None,
            'notes': notes
        })

    return missed_defects


# ==================== 汇总报告 ====================

def generate_summary_report(missed_defects, output_path, total_defects=None, total_images=None):
    """生成漏裁统计报告（TXT）"""
    if total_defects is None:
        total_defects = len(missed_defects)

    reason_counts = defaultdict(int)
    class_counts = defaultdict(int)
    image_counts = defaultdict(int)

    for d in missed_defects:
        reason_counts[d['reason']] += 1
        class_counts[d['defect_name']] += 1
        image_counts[d['original_img_name']] += 1

    total_missed = len(missed_defects)

    lines = []
    lines.append("=" * 70)
    lines.append("漏裁缺陷统计报告")
    lines.append("=" * 70)
    if total_images is not None:
        lines.append(f"分析原图数: {total_images}")
    if total_defects is not None:
        lines.append(f"分析缺陷总数: {total_defects}")
    lines.append(f"漏裁总数: {total_missed}")
    if total_defects and total_defects > 0:
        lines.append(f"漏裁比例: {total_missed / total_defects * 100:.2f}%")
    lines.append("")

    lines.append("按原因统计:")
    for reason in ['component_not_covered', 'incomplete_or_filtered', 'data_inconsistency']:
        count = reason_counts.get(reason, 0)
        lines.append(f"  {reason}: {count}")
    lines.append("")

    lines.append("按缺陷类别统计（漏裁 top 10）:")
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for cls, count in sorted_classes:
        lines.append(f"  {cls}: {count}")
    lines.append("")

    lines.append("按图片统计（漏裁最多 top 10）:")
    sorted_images = sorted(image_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for img, count in sorted_images:
        lines.append(f"  {img}: {count}")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    return report_text


def save_missed_defects_json(missed_defects, output_path):
    """保存漏裁缺陷 JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(missed_defects, f, ensure_ascii=False, indent=2)


def save_missed_defects_csv(missed_defects, output_path):
    """保存漏裁缺陷 CSV"""
    if not missed_defects:
        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['original_img_name', 'original_xml_name', 'original_img_path',
                             'defect_name', 'defect_bbox_original', 'reason',
                             'max_overlap_ratio', 'threshold', 'best_component_class',
                             'best_component_bbox', 'match_method', 'notes'])
        return

    fieldnames = ['original_img_name', 'original_xml_name', 'original_img_path',
                  'defect_name', 'defect_bbox_original', 'reason',
                  'max_overlap_ratio', 'threshold', 'best_component_class',
                  'best_component_bbox', 'match_method', 'notes']

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in missed_defects:
            row = {k: d.get(k, '') for k in fieldnames}
            # 将 bbox 列表转为字符串，便于 Excel 查看
            if isinstance(row['defect_bbox_original'], list):
                row['defect_bbox_original'] = str(row['defect_bbox_original'])
            if isinstance(row['best_component_bbox'], (list, type(None))):
                row['best_component_bbox'] = str(row['best_component_bbox']) if row['best_component_bbox'] else ''
            writer.writerow(row)


def save_missed_image_paths(missed_defects, output_dir):
    """
    按原因分别生成漏裁原图路径 txt 文件，每行一张图像的绝对路径，去重。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reason_paths = {
        'component_not_covered': set(),
        'incomplete_or_filtered': set(),
        'data_inconsistency': set()
    }

    for d in missed_defects:
        reason = d.get('reason')
        img_path = d.get('original_img_path')
        if reason in reason_paths and img_path:
            reason_paths[reason].add(img_path)

    for reason, paths in reason_paths.items():
        file_path = output_dir / f'missed_images_{reason}.txt'
        with open(file_path, 'w', encoding='utf-8') as f:
            for path in sorted(paths):
                f.write(f"{path}\n")
        print(f"  ✅ 漏裁原图路径已保存: {file_path}, 共 {len(paths)} 张")


# ==================== 可视化 ====================

REASON_COLORS = {
    'component_not_covered': 'red',
    'incomplete_or_filtered': 'orange',
    'data_inconsistency': 'purple'
}

REASON_LABELS = {
    'component_not_covered': '部件未覆盖',
    'incomplete_or_filtered': '缺陷不完整/未达阈值',
    'data_inconsistency': '数据异常'
}


def get_chinese_font(size=16):
    """获取支持中文的字体"""
    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/source-han-sans/SourceHanSansCN-Regular.otf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "C:\\Windows\\Fonts\\msyh.ttc",
    ]

    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            continue

    return ImageFont.load_default()


def draw_bbox_on_image(img, bbox, label, color, thickness=3):
    """在图像上绘制边界框和标签"""
    draw = ImageDraw.Draw(img)
    xmin, ymin, xmax, ymax = bbox

    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=thickness)

    font = get_chinese_font(size=16)
    try:
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
    except AttributeError:
        text_width, text_height = draw.textsize(label, font=font)

    # 标签背景
    draw.rectangle(
        [xmin, ymin - text_height - 4, xmin + text_width + 4, ymin],
        fill=color
    )
    draw.text((xmin + 2, ymin - text_height - 2), label, fill='white', font=font)


def _draw_dashed_rectangle(draw, bbox, color, step=10, width=2):
    """绘制虚线矩形框"""
    xmin, ymin, xmax, ymax = bbox
    for i in range(xmin, xmax, step):
        draw.line([(i, ymin), (min(i + step // 2, xmax), ymin)], fill=color, width=width)
        draw.line([(i, ymax), (min(i + step // 2, xmax), ymax)], fill=color, width=width)
    for i in range(ymin, ymax, step):
        draw.line([(xmin, i), (xmin, min(i + step // 2, ymax))], fill=color, width=width)
        draw.line([(xmax, i), (xmax, min(i + step // 2, ymax))], fill=color, width=width)


def _visualize_one_image(task_info):
    """
    单张可视化任务，供多线程调用。

    Args:
        task_info: dict，包含 reason, img_name, img_defects, xml_name,
                   images_dir, image_index, use_pkl_index, component_ann_dir,
                   reason_dir

    Returns:
        int: 1 表示成功，0 表示失败
    """
    reason = task_info['reason']
    img_name = task_info['img_name']
    img_defects = task_info['img_defects']
    xml_name = task_info['xml_name']
    images_dir = task_info['images_dir']
    image_index = task_info['image_index']
    use_pkl_index = task_info['use_pkl_index']
    component_ann_dir = task_info['component_ann_dir']
    reason_dir = task_info['reason_dir']

    # 查找原图路径（支持多种扩展名）
    img_path = find_image_path_for_xml(xml_name, image_index, images_dir, use_pkl_index)
    if not img_path or not os.path.exists(img_path):
        print(f"⚠️  可视化跳过，未找到图片: {img_name}")
        return 0

    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"❌ 打开图片失败 {img_path}: {e}")
        return 0

    draw = ImageDraw.Draw(img)

    # 加载并绘制所有部件检测框（青色虚线）
    if component_ann_dir:
        component_xml_path = Path(component_ann_dir) / xml_name
        if component_xml_path.exists():
            try:
                components = get_objects_from_xml(str(component_xml_path))
                for comp in components:
                    _draw_dashed_rectangle(draw, comp['bbox'], color='cyan', step=15, width=2)
            except Exception as e:
                print(f"⚠️  加载部件框失败 {component_xml_path}: {e}")

    # 绘制该原因下的漏裁缺陷
    color = REASON_COLORS.get(reason, 'red')
    for d in img_defects:
        bbox = d['defect_bbox_original']
        label = f"{d['defect_name']} | {d['max_overlap_ratio']:.2f}"
        draw_bbox_on_image(img, bbox, label, color)

    # 保存
    save_path = Path(reason_dir) / os.path.basename(img_path)
    try:
        img.save(save_path, quality=95)
        return 1
    except Exception as e:
        print(f"❌ 保存可视化失败 {save_path}: {e}")
        return 0


def generate_visualizations(missed_defects, images_dir, image_index, use_pkl_index,
                            output_dir, component_ann_dir=None, max_workers=8, max_images=0):
    """
    生成漏裁可视化图，按原因分目录存放，多线程并行加速。
    每张图只高亮该原因下的漏裁缺陷，同时画出所有部件检测框。
    max_images <= 0 表示不限制数量。
    """
    if not missed_defects:
        return

    output_dir = Path(output_dir)
    enable_limit = (max_images > 0)

    # 按原因分组，再按原图分组
    reason_groups = defaultdict(lambda: defaultdict(list))
    for d in missed_defects:
        reason_groups[d['reason']][d['original_img_name']].append(d)

    # 构造任务列表
    tasks = []
    for reason, img_groups in reason_groups.items():
        reason_dir = output_dir / reason
        reason_dir.mkdir(parents=True, exist_ok=True)

        for img_name, img_defects in img_groups.items():
            xml_name = os.path.splitext(img_name)[0] + '.xml'
            tasks.append({
                'reason': reason,
                'img_name': img_name,
                'img_defects': img_defects,
                'xml_name': xml_name,
                'images_dir': images_dir,
                'image_index': image_index,
                'use_pkl_index': use_pkl_index,
                'component_ann_dir': component_ann_dir,
                'reason_dir': str(reason_dir)
            })

    # 如果启用上限，只取前 max_images 个任务
    if enable_limit:
        tasks = tasks[:max_images]

    viz_count = 0
    if max_workers <= 1:
        for task in tasks:
            viz_count += _visualize_one_image(task)
    else:
        import threading
        counter_lock = threading.Lock()

        def wrapped_task(task):
            result = _visualize_one_image(task)
            with counter_lock:
                nonlocal viz_count
                viz_count += result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(wrapped_task, tasks)

    print(f"✅ 已生成 {viz_count} 张漏裁可视化图，保存到: {output_dir}")


# ==================== 独立脚本批量分析入口 ====================

def run_analysis(component_type, defect_ann_dir, component_ann_dir, crop_mapping_path,
                 config_path, output_dir, images_dir=None, image_pkl_path=None,
                 max_workers=24, enable_viz=False, viz_max=0):
    """
    独立脚本的批量分析入口。
    viz_max <= 0 表示不限制可视化图数量。

    Returns:
        tuple: (missed_defects, total_defects, total_images)
    """
    component_configs, default_threshold = load_config(config_path)
    if component_type not in component_configs:
        print(f"❌ 配置文件中未找到部件类型 {component_type}")
        sys.exit(1)

    config = component_configs[component_type]

    # 加载 crop_mapping
    with open(crop_mapping_path, 'r', encoding='utf-8') as f:
        crop_mapping = json.load(f)
    crop_index = build_crop_mapping_index(crop_mapping)

    # 加载图片索引
    use_pkl_index = (image_pkl_path is not None)
    image_index = None
    if images_dir:
        if use_pkl_index:
            image_index = load_image_index(image_pkl_path)
        else:
            image_index = build_image_index(images_dir)

    # 遍历缺陷 XML
    defect_ann_dir = Path(defect_ann_dir)
    defect_xml_files = sorted([f for f in defect_ann_dir.iterdir() if f.suffix == '.xml'])
    print(f"📁 找到 {len(defect_xml_files)} 个缺陷 XML 文件")

    total_defects = 0
    total_images = len(defect_xml_files)
    all_missed_defects = []
    lock = None if max_workers <= 1 else __import__('threading').Lock()

    def process_one(xml_path):
        xml_name = xml_path.name

        # 获取图像尺寸
        img_width, img_height = get_image_size_from_xml(str(xml_path))
        if img_width is None or img_height is None:
            # XML 中没有尺寸，尝试从对应图片读取
            img_path = find_image_path_for_xml(xml_name, image_index, images_dir, use_pkl_index)
            if img_path and os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                except Exception as e:
                    print(f"⚠️  无法从图片获取尺寸 {img_path}: {e}")
                    return [], 0

        if img_width is None or img_height is None:
            print(f"⚠️  无法获取图像尺寸，跳过: {xml_name}")
            return [], 0

        # 加载缺陷（过滤集合需包含映射前的原始类别名，见 defect_classes_raw）
        all_defects = get_objects_from_xml(str(xml_path), config.defect_classes_raw)
        if not all_defects:
            return [], 0

        # 应用类别映射
        apply_class_mapping(all_defects, config.class_mapping)

        # 加载部件 XML
        component_xml_path = Path(component_ann_dir) / xml_name
        components = []
        if component_xml_path.exists():
            components = get_objects_from_xml(str(component_xml_path), config.component_classes)
        else:
            print(f"⚠️  部件 XML 不存在: {component_xml_path}")

        # 获取 crop_info 列表
        crop_info_list = crop_index.get(xml_name, [])

        # 分析
        missed = analyze_single_image(
            img_name=xml_name.replace('.xml', '.jpg'),
            all_defects=all_defects,
            components=components,
            crop_info_list=crop_info_list,
            config=config,
            default_threshold=default_threshold,
            img_width=img_width,
            img_height=img_height
        )

        # 填充原图路径
        full_path = find_image_path_for_xml(xml_name, image_index, images_dir, use_pkl_index)
        for d in missed:
            d['original_img_path'] = full_path
            d['original_img_name'] = os.path.basename(full_path) if full_path else xml_name.replace('.xml', '.jpg')

        return missed, len(all_defects)

    if max_workers <= 1:
        for xml_path in tqdm(defect_xml_files, desc="分析进度", unit="张"):
            missed, n_defects = process_one(xml_path)
            all_missed_defects.extend(missed)
            total_defects += n_defects
    else:
        import threading
        lock = threading.Lock()

        def wrapped_process(xml_path):
            missed, n_defects = process_one(xml_path)
            with lock:
                all_missed_defects.extend(missed)
                nonlocal total_defects
                total_defects += n_defects

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(wrapped_process, defect_xml_files),
                      total=len(defect_xml_files), desc="分析进度", unit="张"))

    # 保存结果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_missed_defects_json(all_missed_defects, output_dir / 'missed_defects.json')
    save_missed_defects_csv(all_missed_defects, output_dir / 'missed_defects.csv')
    save_missed_image_paths(all_missed_defects, output_dir)

    report_text = generate_summary_report(
        all_missed_defects,
        output_dir / '漏裁统计报告.txt',
        total_defects=total_defects,
        total_images=total_images
    )
    print(report_text)

    # 可视化
    if enable_viz and images_dir:
        viz_dir = output_dir / 'viz'
        generate_visualizations(
            all_missed_defects, images_dir, image_index, use_pkl_index,
            viz_dir, component_ann_dir=component_ann_dir,
            max_workers=max_workers, max_images=viz_max
        )

    return all_missed_defects, total_defects, total_images
