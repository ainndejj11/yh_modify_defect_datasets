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
    'ddx_yw': 0.20,
    'fdjx_cbtl': 0.70,
    'fdjx_jxjd': 0.70,
    'fdjx_jxjx': 0.70,
    'fhjyz_dhss': 0.50,
    'fhjyz_sqbx': 0.60,
    'fhjyz_sqfh': 0.40,
    'fhjyz_sqps': 0.50,
    'jyh_azbgf': 0.70,
    'jyh_qx': 0.70,
    'jyh_sh': 0.70,
    'jyh_tl': 0.70,
    'jyh_zs': 0.70,
    'jyz_jjxs': 0.50,
    'jyz_sqzb': 0.60,
    'tcjyz_dhss': 0.50,
    'tcjyz_sqps': 0.50,
    'wtxztc': 0.80,
    'jyh_qs': 0.80,
    'lslmqk': 0.80,
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

def expand_bbox(bbox, img_width, img_height, long_ratio=0.35, short_ratio=0.45):
    """扩充边界框,并确保不超出图像范围
    
    根据部件长宽判断长边和短边：
    - 长边两边各扩充 long_ratio (默认0.2)
    - 短边两边各扩充 short_ratio (默认0.3)
    
    Args:
        bbox: (xmin, ymin, xmax, ymax) 原始边界框
        img_width: 图像宽度
        img_height: 图像高度
        long_ratio: 长边扩充比例(默认0.2)
        short_ratio: 短边扩充比例(默认0.3)
    
    Returns:
        扩充后的边界框 (xmin, ymin, xmax, ymax)
    """
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    
    # 判断长边和短边，决定各方向扩充比例
    if width >= height:
        # 宽度是长边，高度是短边
        expand_w = width * long_ratio    # 长边扩充0.2
        expand_h = height * short_ratio  # 短边扩充0.3
    else:
        # 高度是长边，宽度是短边
        expand_w = width * short_ratio   # 短边扩充0.3
        expand_h = height * long_ratio   # 长边扩充0.2
    
    # 扩充边界框，并确保不超出图像范围
    new_xmin = max(0, int(xmin - expand_w))
    new_ymin = max(0, int(ymin - expand_h))
    new_xmax = min(img_width, int(xmax + expand_w))
    new_ymax = min(img_height, int(ymax + expand_h))
    
    return (new_xmin, new_ymin, new_xmax, new_ymax)

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

def process_single_xml(xml_file, ann_dir, ann2_dir, image_index, out_img_dir, out_xml_dir, 
                       bujian_class, quexian_class, progress_bar=None):
    """处理单个XML文件：按部件裁切缺陷
    
    Args:
        xml_file: 缺陷xml文件名
        ann_dir: 部件xml目录
        ann2_dir: 缺陷xml目录
        image_index: 图片索引
        out_img_dir: 输出图片目录
        out_xml_dir: 输出xml目录
        bujian_class: 部件类别列表
        quexian_class: 缺陷类别列表
        progress_bar: 进度条
    
    Returns:
        crop_info_list: 裁切映射信息列表
    """
    crop_info_list = []  # 存储裁切映射信息
    try:
        quexian_xml_path = os.path.join(ann2_dir, xml_file)
        
        # 获取缺陷xml中的所有缺陷对象
        all_quexian_objs = get_objects_from_xml(quexian_xml_path, target_classes=quexian_class)
        if not all_quexian_objs:
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
        
        img = Image.open(img_path)
        img_width, img_height = img.size
        crop_count = 0
        
        # 获取部件xml
        bujian_xml_path = os.path.join(ann_dir, xml_file)
        if not os.path.exists(bujian_xml_path):
            if progress_bar:
                progress_bar.update(1)
            return crop_info_list
        
        # 获取部件对象
        bujian_objs = get_objects_from_xml(bujian_xml_path, target_classes=bujian_class)
        
        for idx, parent_obj in enumerate(bujian_objs):
            parent_name = parent_obj['name']
            original_bbox = parent_obj['bbox']
            
            # 扩充边界框(长边0.2, 短边0.3)
            expanded_bbox = expand_bbox(original_bbox, img_width, img_height)
            pxmin, pymin, pxmax, pymax = expanded_bbox
            
            # 在扩充后的边界框内根据面积阈值过滤并裁剪缺陷框
            valid_child_objs = []
            for c in all_quexian_objs:
                result = filter_and_clip_child_obj(expanded_bbox, c)
                if result is not None:
                    # 检查是否发生了裁剪（原始bbox和裁剪后的bbox不同）
                    original_defect_bbox = c['bbox']
                    clipped_bbox = result['bbox']
                    if original_defect_bbox != clipped_bbox:
                        # 统计被裁切的缺陷
                        increment_clipped_stat(c['name'])
                        # 输出裁切前后的bbox坐标
                        print_with_count(f"📐 缺陷被裁剪: 图片={img_name}, 类别={c['name']}, 原始bbox={original_defect_bbox}, 裁剪后bbox={clipped_bbox}")
                    valid_child_objs.append(result)
            
            if not valid_child_objs:
                continue
            
            out_img_name = f"{os.path.splitext(img_name)[0]}_{parent_name}_{idx}.jpg"
            out_xml_name = f"{os.path.splitext(xml_file)[0]}_{parent_name}_{idx}.xml"
            out_img_path = os.path.join(out_img_dir, out_img_name)
            out_xml_path = os.path.join(out_xml_dir, out_xml_name)
            
            # 使用扩充后的边界框裁切图片
            cropped_img = img.crop((pxmin, pymin, pxmax, pymax))
            cropped_img.save(out_img_path)
            
            # 生成对应XML
            root = ET.Element('annotation')
            ET.SubElement(root, 'filename').text = out_img_name
            size = ET.SubElement(root, 'size')
            ET.SubElement(size, 'width').text = str(cropped_img.width)
            ET.SubElement(size, 'height').text = str(cropped_img.height)
            ET.SubElement(size, 'depth').text = str(len(cropped_img.getbands()))
            
            for obj in valid_child_objs:
                cxmin, cymin, cxmax, cymax = obj['bbox']
                # 使用扩充后的边界框调整坐标
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
            print_with_count(f"✅ 裁切完成({parent_name}): {out_img_path}, 缺陷数: {len(valid_child_objs)}")
        
        if progress_bar:
            progress_bar.update(1)
        return crop_info_list
            
    except Exception as e:
        print_with_count(f"❌ 处理 {xml_file} 时出错: {e}")
        if progress_bar:
            progress_bar.update(1)
        return crop_info_list

# ==================== 主函数 ====================
def start_multithread(images_dir, image_pkl_path, ann_dir, ann2_dir, out_dir, 
                     bujian_class, quexian_class, max_workers=16):
    """多线程启动
    
    Args:
        images_dir: 图片目录
        image_pkl_path: 图片索引pkl文件路径
        ann_dir: 部件xml目录
        ann2_dir: 缺陷xml目录
        out_dir: 输出目录
        bujian_class: 部件类别列表
        quexian_class: 缺陷类别列表
        max_workers: 最大线程数
    """
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
    image_pkl_path = '/raid/wtj/ultralytics-8.4.6/缺陷识别-模型优化v6.0/1-总库图像重新进行部件检测/image_indexs_20260114.pkl'
    ann_dir = "/raid/Nas-122/项目数据/输电项目/缺陷/标记样本库/训练集/部件检测xml结果_v5.0"
    ann2_dir = "/raid/Nas-122/项目数据/输电项目/缺陷/标记样本库/训练集/Annotations"
    out_dir = "/raid/datasets_defect_model/jyz_data/v5.3"

    # 部件类别
    bujian_class = ['bljyz', 'tcjyz', 'fhjyz', 'dxjyz', 'fwstljyz']
    
    # 缺陷类别
    quexian_class = ['ddx_yw', 'fdjx_cbtl', 'fdjx_jxjd', 'fdjx_jxjx', 'fhjyz_dhss', 'fhjyz_sqbx', 'fhjyz_sqfh', 
                    'fhjyz_sqps', 'jyh_azbgf', 'jyh_qx', 'jyh_sh', 'jyh_tl', 'jyh_zs', 'jyz_jjxs', 
                    'jyz_sqzb', 'tcjyz_dhss', 'tcjyz_sqps', 'wtxztc', 'jyh_qs', 'lslmqk']

    start_multithread(images_dir, image_pkl_path, ann_dir, ann2_dir, out_dir,
                      bujian_class, quexian_class, max_workers=22)