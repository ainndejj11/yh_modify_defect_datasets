#!/usr/bin/env python3
"""
缺陷数据集映射脚本 - 步骤2：人工审核辅助工具（优化版）

功能：
1. 读取manual_review_queue.json
2. 逐个展示需要人工审核的案例（原图+子图对比可视化）
3. 提供交互界面：保留修改/放弃修改/跳过
4. 生成审核结果JSON供步骤3使用
5. 支持断点续审、批量操作

输入：
- manual_review_queue.json（步骤1生成）
- 子图数据目录
- 原图数据目录

输出：
- manual_review_decisions.json（人工审核决策）
- 可视化对比图（临时缓存）
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
from datetime import datetime


# ==================== 工具函数 ====================

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


def transform_bbox_to_original(sub_bbox, crop_bbox):
    """将子图坐标转换为原图坐标"""
    crop_xmin, crop_ymin, crop_xmax, crop_ymax = crop_bbox
    sub_xmin, sub_ymin, sub_xmax, sub_ymax = sub_bbox

    original_xmin = sub_xmin + crop_xmin
    original_ymin = sub_ymin + crop_ymin
    original_xmax = sub_xmax + crop_xmin
    original_ymax = sub_ymax + crop_ymin

    return (original_xmin, original_ymin, original_xmax, original_ymax)


def get_chinese_font(size=16):
    """获取支持中文的字体"""
    # 按优先级尝试常见的中文字体
    font_paths = [
        # 文泉驿字体（常见于Linux）
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
        # Noto Sans CJK（Google开源字体）
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        # Droid Sans Fallback（Android字体）
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        # 思源黑体
        "/usr/share/fonts/opentype/source-han-sans/SourceHanSansCN-Regular.otf",
        # 其他可能的中文字体路径
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/System/Library/Fonts/PingFang.ttc",  # macOS
        "C:\\Windows\\Fonts\\msyh.ttc",  # Windows 微软雅黑
    ]

    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue

    # 如果都失败，返回默认字体（中文会显示方框）
    print("⚠️  警告: 未找到支持中文的字体，标签可能显示为方框")
    return ImageFont.load_default()


def draw_bbox_on_image(img, bbox, label, color, thickness=3, label_position='top'):
    """在图像上绘制边界框和标签

    Args:
        label_position: 'top' 表示左上角，'bottom' 表示右下角
    """
    draw = ImageDraw.Draw(img)
    xmin, ymin, xmax, ymax = bbox

    # 绘制矩形框
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=thickness)

    # 获取支持中文的字体
    font = get_chinese_font(size=18)

    # 计算标签大小
    bbox_text = draw.textbbox((0, 0), label, font=font)
    text_width = bbox_text[2] - bbox_text[0]
    text_height = bbox_text[3] - bbox_text[1]

    if label_position == 'bottom':
        # 右下角显示
        draw.rectangle(
            [xmax - text_width - 4, ymax, xmax, ymax + text_height + 4],
            fill=color
        )
        draw.text((xmax - text_width - 2, ymax + 2), label, fill='white', font=font)
    else:
        # 左上角显示（默认）
        draw.rectangle(
            [xmin, ymin - text_height - 4, xmin + text_width + 4, ymin],
            fill=color
        )
        draw.text((xmin + 2, ymin - text_height - 2), label, fill='white', font=font)


def create_comparison_image(original_img_path, sub_img_path,
                           original_bbox, sub_bbox,
                           original_label, sub_label,
                           crop_bbox, reason, save_path):
    """
    创建对比可视化图像

    左侧：原图中的缺陷框（绿色=原始，红色=如果应用修改后）
    右侧：子图中的缺陷框（蓝色=修改后）
    """
    try:
        # 加载图像
        original_img = Image.open(original_img_path).convert('RGB')
        sub_img = Image.open(sub_img_path).convert('RGB')

        # 在原图上绘制原始框（绿色，标签在右下角）
        draw_bbox_on_image(
            original_img,
            original_bbox,
            f"原始: {original_label}",
            color='green',
            label_position='bottom'
        )

        # 在原图上绘制映射后的框（红色虚线）
        mapped_bbox = transform_bbox_to_original(sub_bbox, crop_bbox)
        draw_bbox_on_image(
            original_img,
            mapped_bbox,
            f"映射后: {sub_label}",
            color='red'
        )

        # 在子图上绘制修改后的框（蓝色）
        draw_bbox_on_image(
            sub_img,
            sub_bbox,
            f"子图修改: {sub_label}",
            color='blue'
        )

        # 并排拼接
        total_width = original_img.width + sub_img.width + 20
        max_height = max(original_img.height, sub_img.height)

        comparison = Image.new('RGB', (total_width, max_height + 100), 'white')
        comparison.paste(original_img, (0, 0))
        comparison.paste(sub_img, (original_img.width + 20, 0))

        # 添加说明文字
        draw = ImageDraw.Draw(comparison)
        font = get_chinese_font(size=16)

        info_text = f"原因: {reason}\n原始类别: {original_label} → 修改为: {sub_label}"
        draw.text((10, max_height + 10), info_text, fill='black', font=font)

        # 保存
        comparison.save(save_path, quality=95)
        return True

    except Exception as e:
        print(f"❌ 创建对比图失败: {e}")
        return False


# ==================== 人工审核交互 ====================

class ManualReviewer:
    def __init__(self, manual_review_queue_path, sub_images_dir, sub_annotations_dir,
                 original_images_dir, original_xml_dir, output_dir):
        self.queue_path = Path(manual_review_queue_path)
        self.sub_images_dir = Path(sub_images_dir)
        self.sub_annotations_dir = Path(sub_annotations_dir)
        self.original_images_dir = Path(original_images_dir)
        self.original_xml_dir = Path(original_xml_dir)
        self.output_dir = Path(output_dir)

        # 加载待审核队列
        with open(self.queue_path, 'r', encoding='utf-8') as f:
            self.queue = json.load(f)

        # 审核决策记录
        self.decisions = []

        # 加载已有的审核决策（支持断点续审）
        self.decisions_path = self.output_dir / 'manual_review_decisions.json'
        if self.decisions_path.exists():
            with open(self.decisions_path, 'r', encoding='utf-8') as f:
                self.decisions = json.load(f)
            print(f"📋 加载已有审核决策: {len(self.decisions)} 条")

        # 创建临时可视化目录
        self.viz_dir = self.output_dir / 'manual_review_viz'
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"📋 人工审核队列加载完成")
        print(f"{'='*70}")
        print(f"  待审核数量: {len(self.queue)}")
        print(f"  已审核数量: {len(self.decisions)}")
        print(f"  可视化目录: {self.viz_dir}")
        print(f"{'='*70}\n")

    def get_image_path(self, img_name_or_path, is_original=True):
        """
        查找图片路径（支持多种扩展名）

        参数可能是：
        1. 文件名（如 image_xxx.jpg）
        2. 完整路径（如 /raid/.../image_xxx.jpg）
        """
        # 如果是完整路径且存在，直接返回
        if os.path.isabs(img_name_or_path) and os.path.exists(img_name_or_path):
            return img_name_or_path

        # 否则在指定目录中查找
        base_dir = self.original_images_dir if is_original else self.sub_images_dir
        base_name = os.path.splitext(os.path.basename(img_name_or_path))[0]

        # 尝试常见扩展名
        for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            candidate = base_dir / (base_name + ext)
            if candidate.exists():
                return str(candidate)

        return None

    def review_single_case(self, case, index, total):
        """审核单个案例"""
        print(f"\n{'='*70}")
        print(f"📌 案例 [{index + 1}/{total}]")
        print(f"{'='*70}")

        case_type = case.get('type', 'unknown')
        print(f"  案例类型: {case_type}")

        # 根据不同类型展示不同信息
        if case_type == 'clipped_defect_modification':
            # 被裁切缺陷的修改
            self._review_clipped_defect_case(case, index)
        elif case_type == 'one_to_many_conflict':
            # 一对多冲突
            self._review_conflict_case(case, index)
        else:
            print(f"  ⚠️  未知案例类型: {case_type}")
            return 'skip'

        # 交互式决策
        return self._get_user_decision()

    def _review_clipped_defect_case(self, case, index):
        """审核被裁切缺陷的修改案例"""
        print(f"  子图XML: {case['sub_xml']}")
        print(f"  原图XML: {case['original_xml']}")
        print(f"  操作类型: {case['operation_type']}")
        print(f"  审核原因: {case['reason']}")

        if 'sub_defect' in case and 'matched_original' in case:
            print(f"  原始类别: {case['matched_original']['name']}")
            print(f"  修改类别: {case['sub_defect']['name']}")
            print(f"  是否被裁切: {case['matched_original']['was_clipped']}")

            # 获取图片路径
            sub_img_name = case['sub_xml'].replace('.xml', '.jpg')
            sub_img_path = self.get_image_path(sub_img_name, is_original=False)

            original_img_path = self.get_image_path(
                case.get('original_img_path', case['original_xml'].replace('.xml', '.jpg')),
                is_original=True
            )

            if sub_img_path and original_img_path:
                # 创建对比可视化
                viz_path = self.viz_dir / f"review_{index + 1:04d}_clipped.jpg"

                # 修复字段名：使用 bbox_in_source 而不是 bbox_in_source_clipped
                original_bbox = tuple(case['matched_original']['bbox_in_source_original'])
                sub_bbox = tuple(case['sub_defect']['bbox'])
                original_label = case['matched_original']['name']
                sub_label = case['sub_defect']['name']
                crop_bbox = tuple(case['crop_info']['part_bbox'])

                viz_created = create_comparison_image(
                    original_img_path, sub_img_path,
                    original_bbox, sub_bbox,
                    original_label, sub_label,
                    crop_bbox, case['reason'],
                    str(viz_path)
                )

                if viz_created:
                    print(f"  ✅ 对比图: {viz_path}")
            else:
                print(f"  ⚠️  图片文件未找到")

    def _review_conflict_case(self, case, index):
        """审核一对多冲突案例"""
        conflict_type = case.get('conflict_type', 'unknown')
        print(f"  冲突类型: {conflict_type}")
        print(f"  原图XML: {case['original_xml']}")

        if conflict_type == 'added_defect_conflict':
            # 新增框冲突
            print(f"  冲突原因: 多个子图在重叠位置新增了不同类别的缺陷")
            for i, defect in enumerate(case.get('defects', []), 1):
                print(f"    子图{i}: {defect['sub_xml']} - {defect['name']}")
        else:
            # 原图缺陷的修改冲突
            if 'defect_name' in case:
                print(f"  缺陷类别: {case['defect_name']}")

            modifications = case.get('modifications', [])
            print(f"  冲突修改数: {len(modifications)}")
            for i, mod in enumerate(modifications, 1):
                op_type = mod.get('operation_type', 'unknown')
                sub_xml = mod.get('sub_xml', 'unknown')
                print(f"    修改{i}: {sub_xml} - {op_type}")
                if 'sub_defect' in mod:
                    print(f"           新类别: {mod['sub_defect']['name']}")

    def _get_user_decision(self):
        """获取用户决策"""
        print(f"\n{'─'*70}")
        print("  决策选项:")
        print("    [a] 接受修改 - 将子图的修改映射回原图")
        print("    [r] 拒绝修改 - 保持原图不变")
        print("    [s] 跳过 - 暂不决策，稍后处理")
        print("    [q] 保存并退出审核")
        print(f"{'─'*70}")

        while True:
            choice = input("\n  请输入决策 [a/r/s/q]: ").strip().lower()

            if choice == 'a':
                print("  ✅ 决策: 接受修改")
                return 'accept'
            elif choice == 'r':
                print("  ❌ 决策: 拒绝修改")
                return 'reject'
            elif choice == 's':
                print("  ⏭️  决策: 跳过")
                return 'skip'
            elif choice == 'q':
                return 'quit'
            else:
                print("  ⚠️  无效输入，请重新选择")

    def save_decisions(self):
        """保存审核决策"""
        with open(self.decisions_path, 'w', encoding='utf-8') as f:
            json.dump(self.decisions, f, ensure_ascii=False, indent=2)

    def start_review(self):
        """开始交互式审核流程"""
        total = len(self.queue)

        # 找到已审核的案例ID
        reviewed_ids = {d.get('case_id') for d in self.decisions}

        for i, case in enumerate(self.queue):
            # 跳过已审核的案例
            if i in reviewed_ids:
                print(f"\n⏭️  案例 [{i + 1}/{total}] 已审核，跳过")
                continue

            result = self.review_single_case(case, i, total)

            if result == 'quit':
                print(f"\n⏸️  用户选择退出，已审核 {len(self.decisions)}/{total} 个案例")
                break

            if result != 'skip':
                # 记录决策
                decision_record = {
                    'case_id': i,
                    'case_type': case.get('type', 'unknown'),
                    'decision': result,
                    'timestamp': datetime.now().isoformat(),
                    'original_case': case  # 保存完整的原始案例信息
                }

                self.decisions.append(decision_record)

                # 实时保存（防止意外退出丢失进度）
                self.save_decisions()

        # 最终保存
        self.save_decisions()

        print(f"\n{'='*70}")
        print(f"✅ 审核结果已保存")
        print(f"{'='*70}")
        print(f"  总案例数: {total}")
        print(f"  已审核: {len(self.decisions)}")
        print(f"  接受修改: {sum(1 for d in self.decisions if d['decision'] == 'accept')}")
        print(f"  拒绝修改: {sum(1 for d in self.decisions if d['decision'] == 'reject')}")
        print(f"  跳过: {total - len(self.decisions)}")
        print(f"  结果文件: {self.decisions_path}")
        print(f"{'='*70}\n")


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(
        description='缺陷数据集映射 - 步骤2：人工审核辅助工具（优化版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python 映射脚本_步骤2_人工审核_优化版.py \
    --manual-review-queue /raid/datasets_defect_2026/全图测试集/mapping_results/manual_review_queue.json \
    --sub-images /raid/datasets_defect_2026/datasets_val/dx_data_正样本/images \
    --sub-annotations /raid/datasets_defect_2026/datasets_val/dx_data_正样本/Annotations \
    --original-images /raid/datasets_defect_2026/全图测试集/images \
    --original-xml /raid/datasets_defect_2026/全图测试集/Annotations \
    --output /raid/datasets_defect_2026/全图测试集/mapping_results

说明:
  1. 逐个展示需要人工审核的案例（原图+子图对比）
  2. 交互式决策：接受修改/拒绝修改/跳过
  3. 支持断点续审（实时保存进度）
  4. 生成审核决策JSON供步骤3使用
        """
    )

    parser.add_argument(
        '--manual-review-queue',
        required=True,
        help='步骤1生成的manual_review_queue.json文件路径'
    )

    parser.add_argument(
        '--sub-images',
        required=True,
        help='子图images目录'
    )

    parser.add_argument(
        '--sub-annotations',
        required=True,
        help='子图Annotations目录'
    )

    parser.add_argument(
        '--original-images',
        required=True,
        help='原图images目录'
    )

    parser.add_argument(
        '--original-xml',
        required=True,
        help='原图XML目录'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='输出目录（与步骤1相同）'
    )

    args = parser.parse_args()

    # 检查文件/目录是否存在
    if not os.path.exists(args.manual_review_queue):
        print(f"❌ 错误: 审核队列文件不存在: {args.manual_review_queue}")
        sys.exit(1)

    for path, name in [
        (args.sub_images, '子图images目录'),
        (args.sub_annotations, '子图Annotations目录'),
        (args.original_images, '原图images目录'),
        (args.original_xml, '原图XML目录'),
    ]:
        if not os.path.exists(path):
            print(f"❌ 错误: {name}不存在: {path}")
            sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 开始审核
    reviewer = ManualReviewer(
        manual_review_queue_path=args.manual_review_queue,
        sub_images_dir=args.sub_images,
        sub_annotations_dir=args.sub_annotations,
        original_images_dir=args.original_images,
        original_xml_dir=args.original_xml,
        output_dir=args.output
    )

    reviewer.start_review()

    print("\n✅ 步骤2完成！")
    print(f"   下一步：运行步骤3，应用审核决策到原图XML")


if __name__ == '__main__':
    main()
