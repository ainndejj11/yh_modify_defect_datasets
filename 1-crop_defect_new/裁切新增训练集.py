#!/usr/bin/env python3
"""
新增训练集裁切。

不判断缺陷，按 config.yaml 里指定的部件类型（ddx / gt / jyz / gd 等）
裁出该类型的全部部件子图。部件类别和扩充比例都从对应配置段读取。

每张子图写入 crop_mapping_<类型>.json，记录部件类型、原图路径、
原图部件框和扩充后的裁切框，供后续把子图坐标映射回原图。
"""
import argparse
import json
import os
import sys
import threading
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP")

DEFAULT_COMPONENT_TYPE = "gd"  # config.yaml 里的段名：ddx / gt / jyz / gd
DEFAULT_IMAGES_DIR = "/raid/wanzaojun/trans_test/gansu/张掖/images"
DEFAULT_XML_DIR = "/raid/wanzaojun/trans_test/gansu/张掖/xmls"
DEFAULT_OUTPUT_DIR = "/raid/wtj/输电项目_新增挂点训练集/images"

stat_lock = threading.Lock()
class_counter = Counter()
saved_count = 0
skip_no_component = 0
skip_no_image = 0
fail_count = 0


def load_crop_params(config_path, component_type):
    """按部件类型读取 component_classes 和扩充比例。"""
    if not config_path or not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    available = [
        name for name, section in config.items()
        if isinstance(section, dict) and section.get("component_classes")
    ]
    section = config.get(component_type)
    if not isinstance(section, dict) or not section.get("component_classes"):
        print(f"❌ 配置中没有可裁切的部件类型: {component_type}")
        print(f"   可选: {', '.join(available) if available else '（无）'}")
        sys.exit(1)

    classes = list(section["component_classes"])
    expand = bool(section.get("expand_component", False))
    long_ratio = float(section.get("expand_long_ratio", 0.0)) if expand else 0.0
    short_ratio = float(section.get("expand_short_ratio", 0.0)) if expand else 0.0
    return classes, long_ratio, short_ratio, expand


def expand_bbox(bbox, img_width, img_height, long_ratio, short_ratio):
    """按长边、短边自适应扩充边界框，并限制在图像范围内。"""
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        return None

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
    if xmax <= xmin or ymax <= ymin:
        return None
    return (xmin, ymin, xmax, ymax)


def get_components(xml_path, target_classes):
    """从 XML 中提取指定类型的部件框。"""
    objs = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    target = set(target_classes)
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        bndbox = obj.find("bndbox")
        if name_elem is None or name_elem.text is None or bndbox is None:
            continue
        name = name_elem.text.strip()
        if name not in target:
            continue
        try:
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
        except (AttributeError, TypeError, ValueError):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        objs.append({"name": name, "bbox": (xmin, ymin, xmax, ymax)})
    return objs


def build_image_index(images_dir):
    """按文件名（不含扩展名）索引图片，同名时优先 .JPG。"""
    index = {}
    images_dir = Path(images_dir)
    for img_path in images_dir.iterdir():
        if not img_path.is_file() or img_path.suffix not in IMAGE_EXTS:
            continue
        stem = img_path.stem
        old = index.get(stem)
        if old is None or img_path.suffix == ".JPG":
            index[stem] = str(img_path)
    return index


def record(kind, class_name=None):
    global saved_count, skip_no_component, skip_no_image, fail_count
    with stat_lock:
        if kind == "saved":
            saved_count += 1
            class_counter[class_name] += 1
        elif kind == "no_component":
            skip_no_component += 1
        elif kind == "no_image":
            skip_no_image += 1
        elif kind == "fail":
            fail_count += 1


def process_one(xml_path, image_index, out_dir, component_type, component_classes, long_ratio, short_ratio):
    """裁切一张原图上指定类型的全部部件，返回 [(子图xml名, crop_info), ...]。"""
    stem = xml_path.stem
    img_path = image_index.get(stem)
    if not img_path:
        record("no_image")
        return []

    try:
        components = get_components(str(xml_path), component_classes)
    except Exception as e:
        print(f"❌ 解析XML失败 {xml_path.name}: {e}")
        record("fail")
        return []

    if not components:
        record("no_component")
        return []

    try:
        img = Image.open(img_path)
        img.load()
        img_width, img_height = img.size
    except Exception as e:
        print(f"❌ 打开图片失败 {img_path}: {e}")
        record("fail")
        return []

    if img.mode == "RGBA":
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
        img = rgb_img
    elif img.mode != "RGB":
        img = img.convert("RGB")

    crop_records = []
    for idx, component in enumerate(components, 1):
        expanded = expand_bbox(
            component["bbox"], img_width, img_height, long_ratio, short_ratio
        )
        if expanded is None:
            continue
        try:
            cropped = img.crop(expanded)
            new_img_name = f"{stem}_crop{idx}_{component['name']}.jpg"
            new_xml_name = f"{stem}_crop{idx}_{component['name']}.xml"
            cropped.save(os.path.join(out_dir, new_img_name), quality=95)
            # 键用子图 xml 名，与统一裁切的 crop_mapping 一致，便于以后按标注文件名回映射。
            # part_bbox 是扩充后的裁切框（子图左上角在原图上的偏移），
            # component_bbox 是 XML 里的原始部件框。
            crop_records.append((new_xml_name, {
                "component_type": component_type,
                "original_img_path": img_path,
                "original_defect_xml": xml_path.name,
                "original_component_xml": xml_path.name,
                "component_name": component["name"],
                "component_bbox": list(component["bbox"]),
                "part_bbox": list(expanded),
                "crop_image": new_img_name,
                "defects_at_crop_time": [],
            }))
            record("saved", component["name"])
        except Exception as e:
            print(f"❌ 裁切失败 {stem} 部件 {idx} {component['name']}: {e}")
            record("fail")
    return crop_records


def mapping_json_path(output_dir, component_type):
    """子图在 images/ 下时，映射文件放在其上一级，文件名带部件类型。"""
    filename = f"crop_mapping_{component_type}.json"
    output_dir = os.path.normpath(output_dir)
    if os.path.basename(output_dir) == "images":
        return os.path.join(os.path.dirname(output_dir), filename)
    return os.path.join(output_dir, filename)


def main():
    parser = argparse.ArgumentParser(
        description="新增训练集裁切：按 config.yaml 中的部件类型裁出全部对应部件子图，不判断缺陷"
    )
    parser.add_argument(
        "--component-type",
        default=DEFAULT_COMPONENT_TYPE,
        help="部件类型，对应 config.yaml 的段名，如 ddx、gt、jyz、gd",
    )
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, help="原图目录")
    parser.add_argument("--xml-dir", default=DEFAULT_XML_DIR, help="部件XML目录")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="子图保存目录")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="统一裁切配置，按部件类型读取部件类别和扩充比例",
    )
    parser.add_argument("--workers", type=int, default=24, help="线程数")
    args = parser.parse_args()

    if not os.path.isdir(args.images_dir):
        print(f"❌ 原图目录不存在: {args.images_dir}")
        sys.exit(1)
    if not os.path.isdir(args.xml_dir):
        print(f"❌ XML目录不存在: {args.xml_dir}")
        sys.exit(1)

    component_classes, long_ratio, short_ratio, expand = load_crop_params(
        args.config, args.component_type
    )
    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print(f"部件裁切（类型 {args.component_type}，不判断缺陷）")
    print("=" * 70)
    print(f"  部件类型: {args.component_type}")
    print(f"  原图目录: {args.images_dir}")
    print(f"  XML目录: {args.xml_dir}")
    print(f"  保存目录: {args.output}")
    print(f"  部件类别: {', '.join(component_classes)}")
    print(f"  是否扩充: {expand}（长边={long_ratio}, 短边={short_ratio}）")
    print(f"  线程数: {args.workers}")
    print("=" * 70)

    print("扫描原图...")
    image_index = build_image_index(args.images_dir)
    xml_files = sorted(Path(args.xml_dir).glob("*.xml"))
    print(f"图片 {len(image_index)} 张，XML {len(xml_files)} 个")

    all_crop_info = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_one,
                xml_path,
                image_index,
                args.output,
                args.component_type,
                component_classes,
                long_ratio,
                short_ratio,
            )
            for xml_path in xml_files
        ]
        for future in tqdm(futures, desc="裁切进度", unit="xml"):
            all_crop_info.extend(future.result() or [])

    crop_mapping = {xml_name: info for xml_name, info in all_crop_info}
    json_path = mapping_json_path(args.output, args.component_type)
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(crop_mapping, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("裁切统计")
    print("=" * 70)
    print(f"  子图数量: {saved_count}")
    for name in component_classes:
        print(f"    {name}: {class_counter.get(name, 0)}")
    extra = [n for n in class_counter if n not in component_classes]
    for name in extra:
        print(f"    {name}: {class_counter[name]}")
    print(f"  无该类型部件跳过: {skip_no_component}")
    print(f"  无对应原图跳过: {skip_no_image}")
    print(f"  失败: {fail_count}")
    print(f"  保存目录: {args.output}")
    print(f"  裁切映射: {json_path}（{len(crop_mapping)} 条）")
    print("=" * 70)


if __name__ == "__main__":
    main()
