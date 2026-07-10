#!/usr/bin/env python3
import os
import json
import pickle
import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm

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

# ==================== 面积阈值配置 ====================
# 缺陷在部件范围内的面积占比阈值，低于此值则不保留
OVERLAP_THRESHOLDS = {
    'bsp_sh': 0.80,
    'bsp_twbq': 0.80,
    'fc': 0.50,
    'fnc_sh': 0.30,
    'fnc_wdk': 0.40,
    'gt_dyjj': 0.70,
    'gt_qls': 0.80,
    'gt_tcbx': 0.50,
    'gt_yw': 0.15,
    'nc': 0.50,
    'qnq_sh': 0.30,
    'xjqx': 0.80,
}
DEFAULT_THRESHOLD = 0.80  # 默认阈值（未配置的类别使用）

# ==================== 工具函数 ====================

def has_quexian_classes(xml_path, quexian_class):
    """检查XML文件中是否包含指定类别"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is not None and name_elem.text in quexian_class:
                return True
        return False
    except Exception as e:
        print_with_count(f"❌ 解析XML文件 {xml_path} 时出错: {e}")
        return False

def load_image_index(pkl_path):
    """加载图片路径索引"""
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

def is_inside(parent_box, child_box):
    """判断子框是否在父框内"""
    pxmin, pymin, pxmax, pymax = parent_box
    cxmin, cymin, cxmax, cymax = child_box
    return cxmin >= pxmin and cymin >= pymin and cxmax <= pxmax and cymax <= pymax

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

def filter_and_clip_child_obj(parent_box, child_obj):
    """
    根据面积阈值判断缺陷是否保留，并返回裁剪后的坐标
    返回: 修改后的child_obj字典（包含裁剪后的bbox），或None表示不保留
    """
    child_name = child_obj['name']
    child_box = child_obj['bbox']
    
    # 获取该类别的阈值
    threshold = OVERLAP_THRESHOLDS.get(child_name, DEFAULT_THRESHOLD)
    
    # 计算交集比例和裁剪后坐标
    ratio, clipped_box = calculate_overlap_ratio(parent_box, child_box)
    
    # 比例低于阈值，不保留
    if ratio < threshold:
        return None
    
    # 保留，返回裁剪后的对象（注意创建新字典，避免修改原对象）
    return {
        'name': child_name,
        'bbox': clipped_box
    }

def get_objects_from_xml(xml_path, target_classes=None):
    """从XML中提取目标对象"""
    objs = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text
        if target_classes and name not in target_classes:
            continue
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objs.append({'name': name, 'bbox': (xmin, ymin, xmax, ymax)})
    return objs

# ==================== 裁切处理函数 ====================

def process_single_xml(xml_file, ann_dir, ann2_dir, image_index, out_img_dir, out_xml_dir, bujian_class, quexian_class, progress_bar=None):
    """处理单个XML文件：按部件裁切缺陷"""
    crop_info_list = []  # 存储裁切映射信息
    try:
        quexian_xml_path = os.path.join(ann2_dir, xml_file)
        if not has_quexian_classes(quexian_xml_path, quexian_class):
            if progress_bar:
                progress_bar.update(1)
            return crop_info_list

        bujian_xml_path = os.path.join(ann_dir, xml_file)
        if not has_quexian_classes(bujian_xml_path, bujian_class):
            if progress_bar:
                progress_bar.update(1)
            return crop_info_list

        img_name = os.path.splitext(xml_file)[0] + '.jpg'
        img_path = image_index.get(img_name)
        if img_path is None:
            print_with_count(f"⚠️ 未找到图片: {img_name}")
            if progress_bar:
                progress_bar.update(1)
            return crop_info_list

        parent_objs = get_objects_from_xml(bujian_xml_path, bujian_class)
        child_objs = get_objects_from_xml(quexian_xml_path, quexian_class)

        img = Image.open(img_path)

        crop_count = 0
        for idx, parent_obj in enumerate(parent_objs):
            pxmin, pymin, pxmax, pymax = parent_obj['bbox']
            # 过滤并裁剪缺陷框（根据面积阈值判断是否保留，超出部分裁剪到部件边缘）
            valid_child_objs = []
            for c in child_objs:
                result = filter_and_clip_child_obj(parent_obj['bbox'], c)
                if result is not None:
                    # 检查是否发生了裁剪（原始bbox和裁剪后的bbox不同）
                    original_bbox = c['bbox']
                    clipped_bbox = result['bbox']
                    if original_bbox != clipped_bbox:
                        # 统计被裁切的缺陷
                        increment_clipped_stat(c['name'])
                        # 输出裁切前后的bbox坐标
                        print_with_count(f"📐 缺陷被裁剪: 图片={img_name}, 类别={c['name']}, 原始bbox={original_bbox}, 裁剪后bbox={clipped_bbox}")
                    valid_child_objs.append(result)
            if not valid_child_objs:
                continue

            out_img_name = f"{os.path.splitext(img_name)[0]}_{idx}.jpg"
            out_xml_name = f"{os.path.splitext(xml_file)[0]}_{idx}.xml"
            out_img_path = os.path.join(out_img_dir, out_img_name)
            out_xml_path = os.path.join(out_xml_dir, out_xml_name)

            # ======== 裁切保存图片 ========
            cropped_img = img.crop((pxmin, pymin, pxmax, pymax))
            cropped_img.save(out_img_path)

            # ======== 生成对应XML ========
            root = ET.Element('annotation')
            ET.SubElement(root, 'filename').text = out_img_name
            size = ET.SubElement(root, 'size')
            ET.SubElement(size, 'width').text = str(cropped_img.width)
            ET.SubElement(size, 'height').text = str(cropped_img.height)
            ET.SubElement(size, 'depth').text = str(len(cropped_img.getbands()))

            for obj in valid_child_objs:
                cxmin, cymin, cxmax, cymax = obj['bbox']
                adj_box = (cxmin - pxmin, cymin - pymin, cxmax - pxmin, cymax - pymin)
                ob = ET.SubElement(root, 'object')
                ET.SubElement(ob, 'name').text = obj['name']
                bndbox = ET.SubElement(ob, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = str(adj_box[0])
                ET.SubElement(bndbox, 'ymin').text = str(adj_box[1])
                ET.SubElement(bndbox, 'xmax').text = str(adj_box[2])
                ET.SubElement(bndbox, 'ymax').text = str(adj_box[3])

            tree = ET.ElementTree(root)
            tree.write(out_xml_path, encoding='utf-8', xml_declaration=True)

            # ======== 记录裁切映射信息 ========
            crop_info_list.append({
                'cropped_xml': out_xml_name,
                'original_img_path': img_path,
                'original_defect_xml': xml_file,
                'part_bbox': [pxmin, pymin, pxmax, pymax]
            })

            crop_count += 1
            print_with_count(f"✅ 裁切完成: {out_img_path}, 子类数: {len(valid_child_objs)}")

        if progress_bar:
            progress_bar.update(1)
        return crop_info_list
            
    except Exception as e:
        print_with_count(f"❌ 处理 {xml_file} 时出错: {e}")
        if progress_bar:
            progress_bar.update(1)
        return crop_info_list

# ==================== 主函数 ====================
def start_multithread(images_dir, image_pkl_path, ann_dir, ann2_dir, out_dir, bujian_class, quexian_class, max_workers=16):
    """多线程启动"""
    out_img_dir = os.path.join(out_dir, "images")
    out_xml_dir = os.path.join(out_dir, "Annotations")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_xml_dir, exist_ok=True)

    image_index = load_image_index(image_pkl_path)

    xml_files = os.listdir(ann2_dir)
    
    # 用于收集所有裁切映射信息
    all_crop_info = {}
    
    # 创建进度条
    with tqdm(total=len(xml_files), desc="处理XML文件", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = []
            for xml_file in xml_files:
                future = executor.submit(process_single_xml, xml_file, ann_dir, ann2_dir, image_index,
                                        out_img_dir, out_xml_dir, bujian_class, quexian_class, pbar)
                futures.append(future)
            
            # 等待所有任务完成并收集结果
            for future in futures:
                result = future.result()
                if result:
                    for info in result:
                        all_crop_info[info['cropped_xml']] = {
                            'original_img_path': info['original_img_path'],
                            'original_defect_xml': info['original_defect_xml'],
                            'part_bbox': info['part_bbox']
                        }
    
    # 保存JSON文件
    json_path = os.path.join(out_dir, "crop_mapping.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_crop_info, f, ensure_ascii=False, indent=2)
    print_with_count(f"✅ 裁切映射信息已保存到: {json_path}, 共 {len(all_crop_info)} 条记录")
    
    # 输出每个类别被裁切的缺陷统计
    print("\n" + "=" * 60)
    print("📊 缺陷bbox被裁切统计（面积达标但超出部件范围需调整bbox）")
    print("=" * 60)
    if clipped_stats:
        total_clipped = 0
        for class_name in sorted(clipped_stats.keys()):
            count = clipped_stats[class_name]
            total_clipped += count
            print(f"  {class_name}: {count} 个")
        print("-" * 60)
        print(f"  总计: {total_clipped} 个缺陷bbox被裁切")
    else:
        print("  无缺陷bbox被裁切")
    print("=" * 60)

# ==================== 主程序入口 ====================

if __name__ == "__main__":
    images_dir = "/raid/Nas-122/项目数据/输电项目/缺陷/标记样本库/训练集/JPEGImages"
    image_pkl_path = '/raid/wtj/ultralytics-8.4.6/缺陷识别-模型优化/1-总库图像重新进行部件检测/image_indexs_20260114.pkl'
    ann_dir = "/raid/Nas-122/项目数据/输电项目/缺陷/标记样本库/训练集/部件检测xml结果_v5.0"
    ann2_dir = "/raid/Nas-122/项目数据/输电项目/缺陷/标记样本库/训练集/Annotations"
    out_dir = "/raid/datasets_defect_model/gt_data/v5.0"

    bujian_class = ['gt']
    quexian_class = ['bsp_sh', 'bsp_twbq', 'fc', 'fnc_sh', 'fnc_wdk', 'gt_dyjj', 'gt_qls', 'gt_tcbx', 'gt_yw', 'nc', 'qnq_sh', 'xjqx']
    
    start_multithread(images_dir, image_pkl_path, ann_dir, ann2_dir, out_dir,
                      bujian_class, quexian_class, max_workers=10)