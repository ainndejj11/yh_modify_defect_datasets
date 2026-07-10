#!/usr/bin/env python3
import os
import sys
import json
import pickle
import yaml
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm


'''
本版本为 仅裁切正样本（跳过负样本）
支持两种图片索引方式：
1. pkl索引文件（通过 --image-index 参数指定）
2. 文件夹直接扫描（不指定 --image-index 时使用）
'''





# ==================== 线程安全计数器 ====================
counter_lock = threading.Lock()
counter = 0

# 统计每个类别被裁切的缺陷数量（面积达标但超出部件范围需要调整bbox的）
clipped_stats_lock = threading.Lock()
clipped_stats = {}  # {类别名: 裁切次数}

def print_with_count(msg):
    """线程安全打印，每行前加计数"""
    global counter
    with counter_lock:
        counter += 1
        print(f"[{counter}] {msg}")

def increment_clipped_stat(class_name):
    """线程安全地增加某类别的裁切计数"""
    with clipped_stats_lock:
        if class_name not in clipped_stats:
            clipped_stats[class_name] = 0
        clipped_stats[class_name] += 1

# ==================== 配置加载 ====================

class ComponentConfig:
    """单个部件类型的配置"""
    def __init__(self, config_dict):
        self.component_classes = config_dict.get('component_classes', [])
        self.defect_classes = config_dict.get('defect_classes', [])
        self.overlap_thresholds = config_dict.get('overlap_thresholds', {})
        self.expand_component = config_dict.get('expand_component', False)
        self.expand_long_ratio = config_dict.get('expand_long_ratio', 0.35)  # 长边扩充比例
        self.expand_short_ratio = config_dict.get('expand_short_ratio', 0.45)  # 短边扩充比例
        self.class_mapping = config_dict.get('class_mapping', {})

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 解析各部件配置
    component_configs = {}
    for component_type in ['ddx', 'gt', 'jyz', 'gd']:
        if component_type in config:
            component_configs[component_type] = ComponentConfig(config[component_type])

    # 全局配置
    global_config = config.get('global', {})
    default_threshold = global_config.get('default_threshold', 0.80)

    return component_configs, default_threshold

# ==================== 工具函数 ====================

def load_image_index(pkl_path):
    """加载图片路径索引（pkl文件方式）"""
    index = {}
    with open(pkl_path, 'rb') as f:
        while True:
            try:
                part = pickle.load(f)
                index.update(part)
            except EOFError:
                break
    print_with_count(f"✅ 加载完成，共 {len(index)} 条图片索引")
    return index

def build_image_index(images_dir):
    """从图片文件夹构建图片索引（文件夹扫描方式）

    Args:
        images_dir: 图片文件夹路径

    Returns:
        dict: {图片文件名: 完整路径}
    """
    index = {}
    images_dir = Path(images_dir)

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG'}

    # 遍历文件夹（包括子文件夹）
    for img_path in images_dir.rglob('*'):
        if img_path.is_file() and img_path.suffix in image_extensions:
            # 使用文件名作为key，完整路径作为value
            index[img_path.name] = str(img_path)

    print_with_count(f"✅ 扫描完成，共找到 {len(index)} 张图片")
    return index

def expand_bbox(bbox, img_width, img_height, long_ratio=0.35, short_ratio=0.45):
    """根据长边短边自适应扩充边界框

    Args:
        bbox: (xmin, ymin, xmax, ymax) 原始边界框
        img_width: 图像宽度
        img_height: 图像高度
        long_ratio: 长边扩充比例(默认0.35)
        short_ratio: 短边扩充比例(默认0.45)

    Returns:
        扩充后的边界框 (xmin, ymin, xmax, ymax)
    """
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin

    # 判断长边和短边，决定各方向扩充比例
    if width >= height:
        # 宽度是长边，高度是短边
        expand_w = width * long_ratio    # 长边扩充比例
        expand_h = height * short_ratio  # 短边扩充比例
    else:
        # 高度是长边，宽度是短边
        expand_w = width * short_ratio   # 短边扩充比例
        expand_h = height * long_ratio   # 长边扩充比例

    # 扩充边界框，并确保不超出图像范围
    xmin = max(0, int(xmin - expand_w))
    ymin = max(0, int(ymin - expand_h))
    xmax = min(img_width, int(xmax + expand_w))
    ymax = min(img_height, int(ymax + expand_h))

    return (xmin, ymin, xmax, ymax)

def calculate_overlap_ratio(parent_box, child_box):
    """
    计算子框与父框交集面积占子框总面积的比例
    返回: (比例, 裁剪后的子框坐标) 或 (0, None) 如果无交集
    """
    pxmin, pymin, pxmax, pymax = parent_box
    cxmin, cymin, cxmax, cymax = child_box

    # 计算交集坐标
    inter_xmin = max(pxmin, cxmin)
    inter_ymin = max(pymin, cymin)
    inter_xmax = min(pxmax, cxmax)
    inter_ymax = min(pymax, cymax)

    # 无交集
    if inter_xmin >= inter_xmax or inter_ymin >= inter_ymax:
        return 0, None

    # 计算面积
    child_area = (cxmax - cxmin) * (cymax - cymin)
    if child_area <= 0:
        return 0, None

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    ratio = inter_area / child_area

    # 返回比例和裁剪后的坐标（交集坐标就是裁剪后的坐标）
    clipped_box = (inter_xmin, inter_ymin, inter_xmax, inter_ymax)
    return ratio, clipped_box

def filter_and_process_defect(parent_box, defect_obj, overlap_thresholds, default_threshold):
    """
    根据面积阈值判断缺陷是否保留，并返回完整的映射信息
    返回: {
        'name': 类别名,
        'bbox_in_source_original': 原图中的原始坐标,
        'bbox_in_source_clipped': 原图中裁切后的坐标,
        'was_clipped': 是否被裁切,
        'overlap_ratio': 面积占比
    } 或 None（不保留）
    """
    defect_name = defect_obj['name']
    original_bbox = defect_obj['bbox']

    # 获取该类别的阈值
    threshold = overlap_thresholds.get(defect_name, default_threshold)

    # 计算交集比例和裁剪后坐标
    ratio, clipped_bbox = calculate_overlap_ratio(parent_box, original_bbox)

    # 比例低于阈值，不保留
    if ratio < threshold:
        return None

    # 判断是否被裁切
    was_clipped = (clipped_bbox != original_bbox)

    if was_clipped:
        increment_clipped_stat(defect_name)

    # 返回完整信息
    return {
        'name': defect_name,
        'bbox_in_source_original': original_bbox,
        'bbox_in_source_clipped': clipped_bbox,
        'was_clipped': was_clipped,
        'overlap_ratio': round(ratio, 4)
    }

def get_objects_from_xml(xml_path, target_classes=None):
    """从XML中提取目标对象"""
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
                    print_with_count(f"⚠️  XML {xml_path} 中的对象 {name} 缺少 bndbox 元素，跳过")
                    continue

                xmin_elem = bndbox.find('xmin')
                ymin_elem = bndbox.find('ymin')
                xmax_elem = bndbox.find('xmax')
                ymax_elem = bndbox.find('ymax')

                if None in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]:
                    print_with_count(f"⚠️  XML {xml_path} 中的对象 {name} 坐标信息不完整，跳过")
                    continue

                xmin = int(xmin_elem.text)
                ymin = int(ymin_elem.text)
                xmax = int(xmax_elem.text)
                ymax = int(ymax_elem.text)
                objs.append({'name': name, 'bbox': (xmin, ymin, xmax, ymax)})
            except (ValueError, AttributeError) as e:
                print_with_count(f"⚠️  XML {xml_path} 解析对象时出错: {e}，跳过该对象")
                continue
    except Exception as e:
        print_with_count(f"❌ 解析XML文件失败 {xml_path}: {e}")
    return objs

def create_xml(filename, width, height, objects, save_path):
    """
    创建VOC格式的XML文件
    objects: list of {'name': str, 'bbox': (xmin, ymin, xmax, ymax)}
    """
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
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    # 保存XML文件
    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    tree.write(save_path, encoding='utf-8', xml_declaration=True)

# ==================== 处理单张图片 ====================

def process_single_image(img_name, images_dir, image_index, component_ann_dir, defect_ann_dir, out_dir,
                         config, default_threshold, use_pkl_index):
    """
    处理单张图片的裁切
    config: ComponentConfig 对象
    component_ann_dir: 部件检测XML目录
    defect_ann_dir: 缺陷标注XML目录
    use_pkl_index: 是否使用pkl索引（True=使用pkl，False=使用文件夹扫描）
    """
    crop_results = []

    # 获取原图路径
    img_path = image_index.get(img_name)
    if not img_path:
        print_with_count(f"⚠️  图片 {img_name} 未在索引中找到，跳过")
        return crop_results

    # 根据索引方式处理路径
    if use_pkl_index:
        # pkl索引方式：img_path是相对路径，需要拼接
        full_img_path = os.path.join(images_dir, img_path)
    else:
        # 文件夹扫描方式：img_path已经是完整路径
        full_img_path = img_path

    if not os.path.exists(full_img_path):
        print_with_count(f"⚠️  图片文件不存在: {full_img_path}，跳过")
        return crop_results

    # 打开原图
    try:
        img = Image.open(full_img_path)
        img_width, img_height = img.size
    except Exception as e:
        print_with_count(f"❌ 打开图片失败 {full_img_path}: {e}")
        return crop_results

    # 解析部件XML
    xml_name = os.path.splitext(img_name)[0] + '.xml'
    xml_path = os.path.join(component_ann_dir, xml_name)

    if not os.path.exists(xml_path):
        print_with_count(f"⚠️  部件XML不存在: {xml_path}，跳过")
        return crop_results

    # 获取所有部件框
    components = get_objects_from_xml(xml_path, config.component_classes)
    if not components:
        print_with_count(f"⚠️  图片 {img_name} 中未找到目标部件类别，跳过")
        return crop_results

    # 解析缺陷XML
    xml2_path = os.path.join(defect_ann_dir, xml_name)
    all_defects = []
    if os.path.exists(xml2_path):
        all_defects = get_objects_from_xml(xml2_path, config.defect_classes)
        # 应用类别映射（如果有）
        if config.class_mapping:
            for defect in all_defects:
                if defect['name'] in config.class_mapping:
                    defect['name'] = config.class_mapping[defect['name']]

    # 遍历每个部件进行裁切
    for idx, component in enumerate(components, 1):
        comp_bbox = component['bbox']

        # 如果需要扩充部件框
        if config.expand_component:
            comp_bbox_expanded = expand_bbox(
                comp_bbox, img_width, img_height,
                long_ratio=config.expand_long_ratio,
                short_ratio=config.expand_short_ratio
            )
        else:
            comp_bbox_expanded = comp_bbox

        xmin, ymin, xmax, ymax = comp_bbox_expanded

        # 筛选该部件内的缺陷
        defects_in_crop = []
        for defect in all_defects:
            result = filter_and_process_defect(comp_bbox_expanded, defect,
                                               config.overlap_thresholds, default_threshold)
            if result:
                # 转换为子图坐标
                src_xmin, src_ymin, src_xmax, src_ymax = result['bbox_in_source_clipped']
                crop_xmin = src_xmin - xmin
                crop_ymin = src_ymin - ymin
                crop_xmax = src_xmax - xmin
                crop_ymax = src_ymax - ymin

                defects_in_crop.append({
                    'name': result['name'],
                    'bbox_in_crop': (crop_xmin, crop_ymin, crop_xmax, crop_ymax),
                    'bbox_in_source_clipped': result['bbox_in_source_clipped'],
                    'bbox_in_source_original': result['bbox_in_source_original'],
                    'was_clipped': result['was_clipped'],
                    'overlap_ratio': result['overlap_ratio']
                })

        # 只保留正样本（包含缺陷的裁切），负样本直接跳过
        if len(defects_in_crop) == 0:
            continue

        # 裁切子图
        try:
            cropped_img = img.crop((xmin, ymin, xmax, ymax))
        except Exception as e:
            print_with_count(f"❌ 裁切失败 {img_name} 部件 {idx}: {e}")
            continue

        # 如果是RGBA模式，转换为RGB
        if cropped_img.mode == 'RGBA':
            rgb_img = Image.new('RGB', cropped_img.size, (255, 255, 255))
            rgb_img.paste(cropped_img, mask=cropped_img.split()[3])
            cropped_img = rgb_img

        crop_width, crop_height = cropped_img.size

        # 生成新文件名
        base_name = os.path.splitext(img_name)[0]
        new_img_name = f"{base_name}_crop{idx}.jpg"
        new_xml_name = f"{base_name}_crop{idx}.xml"

        # 保存子图
        out_img_path = os.path.join(out_dir, "images", new_img_name)
        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        cropped_img.save(out_img_path, quality=95)

        # 保存XML
        xml_objects = [{'name': d['name'], 'bbox': d['bbox_in_crop']} for d in defects_in_crop]
        out_xml_path = os.path.join(out_dir, "Annotations", new_xml_name)
        os.makedirs(os.path.dirname(out_xml_path), exist_ok=True)
        create_xml(new_img_name, crop_width, crop_height, xml_objects, out_xml_path)

        # 记录裁切信息（增强版：包含缺陷映射关系）
        defects_mapping = []
        for d in defects_in_crop:
            defects_mapping.append({
                'name': d['name'],
                'bbox_in_crop': list(d['bbox_in_crop']),
                'bbox_in_source': list(d['bbox_in_source_clipped']),
                'bbox_in_source_original': list(d['bbox_in_source_original']),
                'was_clipped': d['was_clipped'],
                'overlap_ratio': d['overlap_ratio']
            })

        crop_info = {
            'original_img_path': full_img_path if not use_pkl_index else img_path,
            'original_defect_xml': xml_name,
            'part_bbox': list(comp_bbox_expanded),
            'defects_at_crop_time': defects_mapping  # 新增：裁切时保留的缺陷详情
        }
        crop_results.append((new_xml_name, crop_info))

    return crop_results

# ==================== 多线程处理 ====================

def start_multithread(images_dir, image_pkl_path, component_ann_dir, defect_ann_dir, out_dir,
                      component_type, config_path, max_workers=24):
    """
    多线程批量处理图片
    component_type: 部件类型，如 'ddx', 'gt', 'jyz', 'gd'
    config_path: 配置文件路径
    images_dir: 原图目录
    image_pkl_path: 图片索引文件（可选，None时使用文件夹扫描）
    component_ann_dir: 部件检测XML目录
    defect_ann_dir: 缺陷标注XML目录
    out_dir: 输出目录
    """
    print("=" * 70)
    print(f"🚀 开始处理部件类型: {component_type}")
    print("=" * 70)

    # 加载配置
    component_configs, default_threshold = load_config(config_path)

    if component_type not in component_configs:
        print(f"❌ 配置文件中未找到部件类型 {component_type}")
        print(f"   可用的部件类型: {', '.join(component_configs.keys())}")
        return

    config = component_configs[component_type]

    print(f"\n📋 配置信息:")
    print(f"  部件类别: {', '.join(config.component_classes)}")
    print(f"  缺陷类别数: {len(config.defect_classes)}")
    print(f"  面积阈值配置数: {len(config.overlap_thresholds)}")
    print(f"  扩充部件框: {'是' if config.expand_component else '否'}")
    if config.expand_component:
        print(f"  扩充比例: 长边={config.expand_long_ratio}, 短边={config.expand_short_ratio}")
    if config.class_mapping:
        print(f"  类别映射数: {len(config.class_mapping)}")
    print(f"  模式: 仅正样本（跳过无缺陷的裁切）")
    print()

    # 根据是否提供pkl文件选择索引方式
    use_pkl_index = (image_pkl_path is not None)

    if use_pkl_index:
        print("📦 加载图片索引（pkl文件）...")
        image_index = load_image_index(image_pkl_path)
    else:
        print("📦 扫描图片文件夹...")
        image_index = build_image_index(images_dir)

    # 获取所有XML文件
    xml_files = [f for f in os.listdir(component_ann_dir) if f.endswith('.xml')]
    print(f"📁 找到 {len(xml_files)} 个部件XML文件\n")

    # 创建输出目录
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "Annotations"), exist_ok=True)

    # 多线程处理
    all_crop_info = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for xml_file in xml_files:
            # 根据索引方式查找图片文件名
            base_name = os.path.splitext(xml_file)[0]
            img_name = None

            if use_pkl_index:
                # pkl索引方式：尝试.jpg扩展名
                img_name = base_name + '.jpg'
                if img_name not in image_index:
                    img_name = None
            else:
                # 文件夹扫描方式：尝试多种扩展名
                for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']:
                    candidate = base_name + ext
                    if candidate in image_index:
                        img_name = candidate
                        break

            # 如果没找到，跳过这个XML
            if img_name is None:
                print_with_count(f"⚠️  XML文件 {xml_file} 没有找到对应的图片，跳过")
                continue

            future = executor.submit(
                process_single_image,
                img_name, images_dir, image_index, component_ann_dir, defect_ann_dir, out_dir,
                config, default_threshold, use_pkl_index
            )
            futures.append(future)

        # 使用进度条
        for future in tqdm(futures, desc="处理进度", unit="img"):
            results = future.result()
            all_crop_info.extend(results)

    # 保存裁切映射信息到JSON
    crop_mapping = {}
    for xml_name, info in all_crop_info:
        crop_mapping[xml_name] = info

    json_path = os.path.join(out_dir, "crop_mapping_正样本.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(crop_mapping, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 裁切映射信息已保存到: {json_path}, 共 {len(crop_mapping)} 条记录")

    # 统计信息
    print("\n" + "=" * 70)
    print("📊 裁切统计")
    print("=" * 70)
    print(f"  正样本（包含缺陷）: {len(crop_mapping)} 个")
    print(f"  总计: {len(crop_mapping)} 个子图")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("📊 缺陷bbox被裁切统计（面积达标但超出部件范围需调整bbox）")
    print("=" * 70)
    if clipped_stats:
        total_clipped = 0
        for class_name in sorted(clipped_stats.keys()):
            count = clipped_stats[class_name]
            total_clipped += count
            print(f"  {class_name}: {count} 个")
        print("-" * 70)
        print(f"  总计: {total_clipped} 个缺陷bbox被裁切")
    else:
        print("  无缺陷bbox被裁切")
    print("=" * 70)

# ==================== 主程序入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description='部件缺陷数据集裁切工具（仅正样本）- 支持pkl索引和文件夹扫描两种方式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 方式一：
  #     使用pkl索引文件

  python 统一裁切程序_正样本.py ddx \
    --images-dir /path/to/JPEGImages \
    --image-index /path/to/image_index.pkl \
    --component-ann /path/to/导地线检测xml结果 \
    --defect-ann /path/to/Annotations \
    --output /path/to/output_ddx

    
  # 方式二：
  #     使用文件夹扫描（不指定--image-index）

  python 统一裁切程序_正样本.py gt \
    --images-dir /raid/datasets_defect_2026/全图测试集/images \
    --component-ann /raid/datasets_defect_2026/全图测试集/部件_xml \
    --defect-ann /raid/datasets_defect_2026/全图测试集/Annotations \
    --output /raid/datasets_defect_2026/datasets_val/gt_data/gt_正样本

支持的部件类型: ddx(导地线), gt(杆塔), jyz(绝缘子), gd(挂点)
        """
    )

    parser.add_argument(
        'component_type',
        choices=['ddx', 'gt', 'jyz', 'gd'],
        help='部件类型: ddx(导地线), gt(杆塔), jyz(绝缘子), gd(挂点)'
    )

    parser.add_argument(
        '--images-dir',
        required=True,
        help='原图目录路径'
    )

    parser.add_argument(
        '--image-index',
        required=False,
        default=None,
        help='图片索引pkl文件路径（可选，不指定时自动扫描images-dir文件夹）'
    )

    parser.add_argument(
        '--component-ann',
        required=True,
        help='部件检测XML目录路径'
    )

    parser.add_argument(
        '--defect-ann',
        required=True,
        help='缺陷标注XML目录路径'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='输出目录路径'
    )

    parser.add_argument(
        '--config',
        default='./config.yaml',
        help='配置文件路径 (默认: ./config.yaml)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=20,
        help='线程数 (默认: 24)'
    )

    args = parser.parse_args()

    # 检查路径是否存在
    if not os.path.exists(args.images_dir):
        print(f"❌ 错误: 原图目录不存在: {args.images_dir}")
        sys.exit(1)

    if args.image_index and not os.path.exists(args.image_index):
        print(f"❌ 错误: 图片索引文件不存在: {args.image_index}")
        sys.exit(1)

    if not os.path.exists(args.component_ann):
        print(f"❌ 错误: 部件检测XML目录不存在: {args.component_ann}")
        sys.exit(1)

    if not os.path.exists(args.defect_ann):
        print(f"❌ 错误: 缺陷标注XML目录不存在: {args.defect_ann}")
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"❌ 错误: 配置文件不存在: {args.config}")
        sys.exit(1)

    # 打印参数信息
    print("\n" + "=" * 70)
    print("🔧 运行参数")
    print("=" * 70)
    print(f"  部件类型: {args.component_type}")
    print(f"  原图目录: {args.images_dir}")
    if args.image_index:
        print(f"  图片索引: {args.image_index} (pkl索引方式)")
    else:
        print(f"  图片索引: 文件夹扫描方式")
    print(f"  部件XML目录: {args.component_ann}")
    print(f"  缺陷XML目录: {args.defect_ann}")
    print(f"  输出目录: {args.output}")
    print(f"  配置文件: {args.config}")
    print(f"  线程数: {args.workers}")
    print(f"  模式: 仅正样本")
    print("=" * 70 + "\n")

    # 执行裁切
    start_multithread(
        images_dir=args.images_dir,
        image_pkl_path=args.image_index,
        component_ann_dir=args.component_ann,
        defect_ann_dir=args.defect_ann,
        out_dir=args.output,
        component_type=args.component_type,
        config_path=args.config,
        max_workers=args.workers
    )

    print("\n" + "=" * 70)
    print("🎉 处理完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()