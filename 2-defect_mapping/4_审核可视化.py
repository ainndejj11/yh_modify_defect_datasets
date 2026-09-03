#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4_审核可视化.py — 把待人工确认的条目渲染成对比图，并把人工决策回灌成补充变更集

三个子命令：

  render          读 review_queue.json，为每条待确认项渲染「原图局部 + 子图」对比图，
                  并生成带决策界面的 index.html。加 --serve 可渲染完直接起服务。

  serve           启动 HTTP 服务提供审核页面。远程连 Linux 服务器时用这个：
                  决策通过 API 实时存到服务器磁盘上，补充变更集也能在网页上直接生成，
                  不用在 Windows 和 Linux 之间来回倒腾 decisions.json。

  build-changeset 命令行版的决策回灌：读 decisions.json 组装成补充变更集，
                  再交给 2_应用变更集.py 写入。

用法：
  # 渲染完直接起服务（远程审核推荐一步到位）
  python 4_审核可视化.py render \
    --review-queue /raid/datasets_defect_2026/datasets_val/全量_正样本/gd_data/映射原图过程/review_queue.json \
    --dataset-dir /raid/datasets_defect_2026/datasets_val/全量_正样本/gd_data \
    --original-images /raid/Nas-122/项目数据/输电项目/缺陷/标记样本库/全图测试集2026/images \
    --output /raid/datasets_defect_2026/datasets_val/全量_正样本/gd_data/映射原图过程 --serve


  # 之后想重新打开审核页面
  python 4_审核可视化.py serve --review-dir /raid/datasets_defect_2026/datasets_val/全量_正样本/gd_data/映射原图过程


  # 命令行方式回灌决策（等价于网页上点「生成补充变更集」）
  python 4_审核可视化.py build-changeset \
    --review-queue ./输出/gt_data/review_queue.json \
    --decisions ./输出/gt_data/review/decisions.json \
    --base-changeset ./输出/gt_data/changeset.json \
    --output ./输出/gt_data/supplement_changeset.json
"""

from __future__ import annotations

import argparse
import html
import http.server
import json
import math
import os
import re
import shutil
import socketserver
import sys
import time
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mapping_core as mc

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None

IMG_EXTS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP',
            '.tif', '.tiff']

# 与 1_生成变更集.py --dup-iou 默认值一致：补充变更集回灌时合并几乎重合的框
SUPPLEMENT_DUP_IOU = 0.5

# 画框配色
COLOR_ORIGINAL = (0, 200, 0)       # 原图里已有的框
COLOR_PROPOSED = (255, 40, 40)     # 人工改动后 / 待新增的框
COLOR_CROP = (0, 180, 255)         # 部件裁切区域
COLOR_ALT = (255, 170, 0)          # 其他候选
COLOR_CURRENT = (120, 120, 120)    # 方案效果图里的「当前框」（灰色虚线）
COLOR_EDGE = (220, 0, 220)         # 新旧框之间有变化的边（品红加粗）
DIM_RGBA = (15, 23, 42, 120)       # 裁切区以外的压暗遮罩
CLIP_RGBA = (255, 40, 40, 80)      # 被截断部分的红色半透明垫底
PAD_BG = (15, 23, 42)              # 小图加宽时的两侧填充
MIN_STRIP_WIDTH = 440              # 图例/说明带的最小宽度，避免小图把字裁掉
MIN_DRAW_WIDTH = 420               # 过窄的裁切局部先放大再画字
MIN_DRAW_HEIGHT = 260
MAX_UPSCALE_SIDE = 1600            # 放大时最长边上限，避免细长条被拉成巨图

# 页面上的中文原因标题（完整解释仍在 reason_text 里）
REASON_CN = {
    'clipped_deleted': '被裁切的缺陷在子图里被删除',
    'clipped_bbox_modified_border': '被裁切的缺陷改了框，新框仍贴裁切边界',
    'target_not_found': '原图里找不到目标框',
    'target_ambiguous': '原图里存在重复框，无法确定改哪个',
    'added_overlaps_existing': '新增框与原图已有框重叠',
    'added_out_of_scope': '新增类别不在本数据集作用域内',
    'added_invalid_bbox': '新增框坐标非法',
    'cross_crop_inconsistent': '多个子图的清洗结果互相矛盾',
    'cross_crop_added_conflict': '多个子图在同一位置新增了不同类别',
    'ledger_conflict': '与其他数据集的修改冲突',
    'sub_xml_unreadable': '子图 XML 无法读取',
}

FONT_CACHE: Dict[int, object] = {}

FONT_PATHS = [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    '/usr/share/fonts/opentype/source-han-sans/SourceHanSansCN-Regular.otf',
    '/usr/share/fonts/truetype/arphic/uming.ttc',
]


def get_font(size: int):
    if size in FONT_CACHE:
        return FONT_CACHE[size]
    font = None
    for path in FONT_PATHS:
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    FONT_CACHE[size] = font
    return font


def find_image(directory: str, stem: str) -> Optional[str]:
    """按文件名主干在目录中找图片，兼容多种扩展名"""
    if not directory:
        return None
    for ext in IMG_EXTS:
        candidate = os.path.join(directory, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def _wrap_text(text: str, font, max_width: int) -> List[str]:
    """按像素宽度折行，中文按字切，避免图例/标签画出画布被截断"""
    text = text or ''
    if max_width <= 8 or _text_size(text, font)[0] <= max_width:
        return [text or ' ']
    lines: List[str] = []
    current = ''
    for ch in text:
        trial = current + ch
        if current and _text_size(trial, font)[0] > max_width:
            lines.append(current)
            current = '' if ch == ' ' else ch
        else:
            current = trial
    if current:
        lines.append(current)
    return lines or [text]


def _pad_to_width(img, width: int, bg=PAD_BG):
    """把图左右补到指定宽度（内容居中），给顶部图例留出写字空间"""
    if img.width >= width:
        return img
    out = Image.new('RGB', (width, img.height), bg)
    out.paste(img, ((width - img.width) // 2, 0))
    return out


def ensure_readable_size(img, min_w: int = MIN_DRAW_WIDTH,
                         min_h: int = MIN_DRAW_HEIGHT,
                         max_side: int = MAX_UPSCALE_SIDE):
    """
    小缺陷裁出来往往只有几十像素。先放大再画框/写字，
    点开灯箱时图例和边注才不会糊成一截。大图不动。
    返回 (处理后的图, 放大倍数)。
    """
    w, h = img.size
    if w <= 0 or h <= 0:
        return img, 1.0
    scale = 1.0
    if w < min_w:
        scale = max(scale, min_w / w)
    if h < min_h:
        scale = max(scale, min_h / h)
    if max(w, h) * scale > max_side:
        scale = min(scale, max_side / max(w, h))
    if scale <= 1.01:
        return img, 1.0
    return img.resize((max(1, int(round(w * scale))),
                       max(1, int(round(h * scale)))), Image.BILINEAR), scale


def draw_tag_at(draw, cx: float, cy: float, label: str, color,
                font_size: int = 16, bounds: Optional[Tuple[int, int]] = None) -> None:
    """在任意点居中画文字标签（用于标注有变化的边），超出画面时折行并往内收"""
    if bounds:
        max_tw = max(24, int(bounds[0]) - 16)
        while font_size > 11 and _text_size(label, get_font(font_size))[0] > max_tw * 2:
            font_size -= 1
        font = get_font(font_size)
        lines = _wrap_text(label, font, max_tw)
    else:
        font = get_font(font_size)
        lines = [label]
    sizes = [_text_size(ln, font) for ln in lines]
    tw = max(s[0] for s in sizes)
    th = sum(s[1] for s in sizes) + 2 * (len(lines) - 1)
    x, y = cx - tw / 2, cy - th / 2
    if bounds:
        w, h = bounds
        x = min(max(x, 4), max(4.0, w - tw - 4))
        y = min(max(y, 3), max(3.0, h - th - 3))
    draw.rectangle([x - 3, y - 2, x + tw + 3, y + th + 2], fill=color)
    ty = y
    for ln, (_, lh) in zip(lines, sizes):
        draw.text((x, ty), ln, fill=(255, 255, 255), font=font)
        ty += lh + 2


def _dashed_line(draw, x1, y1, x2, y2, color, width: int,
                 dash: int = 14, gap: int = 10) -> None:
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0
    while pos < length:
        seg = min(pos + dash, length)
        draw.line([x1 + dx * pos, y1 + dy * pos,
                   x1 + dx * seg, y1 + dy * seg], fill=color, width=width)
        pos += dash + gap


def draw_dashed_rect(draw, bbox, color, width: int = 3,
                     dash: int = 14, gap: int = 10) -> None:
    x0, y0, x1, y1 = (float(v) for v in bbox)
    _dashed_line(draw, x0, y0, x1, y0, color, width, dash, gap)
    _dashed_line(draw, x1, y0, x1, y1, color, width, dash, gap)
    _dashed_line(draw, x1, y1, x0, y1, color, width, dash, gap)
    _dashed_line(draw, x0, y1, x0, y0, color, width, dash, gap)


def draw_cross(draw, bbox, color, width: int = 4) -> None:
    """在框内画对角叉，表示删除"""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    draw.line([x0, y0, x1, y1], fill=color, width=width)
    draw.line([x0, y1, x1, y0], fill=color, width=width)


def draw_box(draw, bbox: Sequence[float], color, width: int = 3,
             dashed: bool = False) -> None:
    """只画框不写字 —— 文字统一放到图例带里，免得糊住画面内容"""
    xmin, ymin, xmax, ymax = (float(v) for v in bbox)
    if dashed:
        draw_dashed_rect(draw, [xmin, ymin, xmax, ymax], color, width,
                         dash=max(8, width * 4), gap=max(6, width * 3))
    else:
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=width)


def fill_rect_alpha(img, rect, rgba):
    """在图上盖一块半透明矩形（RGB 图 -> 合成后仍返回 RGB）"""
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle([float(v) for v in rect], fill=rgba)
    return Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')


def dim_outside_part(view, part_rel):
    """
    把部件裁切区以外的地方压暗 —— 那是子图里根本看不到的区域，
    是 clipped_* 类冲突的根源。含义由图例带说明，图上不再写字。
    返回 (处理后的图, 是否有遮罩)。
    """
    if not part_rel:
        return view, False
    W, H = view.size
    x0, y0, x1, y1 = (float(v) for v in part_rel)
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(W), x1), min(float(H), y1)
    slabs = [[0, 0, W, y0], [0, y1, W, H], [0, y0, x0, y1], [x1, y0, W, y1]]
    slabs = [s for s in slabs if s[2] - s[0] > 1 and s[3] - s[1] > 1]
    if not slabs:
        return view, False
    overlay = Image.new('RGBA', view.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for s in slabs:
        d.rectangle(s, fill=DIM_RGBA)
    return Image.alpha_composite(view.convert('RGBA'), overlay).convert('RGB'), True


def clipped_slabs(box, part) -> List[List[float]]:
    """box 落在 part（部件裁切区）之外的部分，拆成若干矩形"""
    x0, y0, x1, y1 = (float(v) for v in box)
    px0, py0, px1, py1 = (float(v) for v in part)
    slabs = []
    if y0 < py0:
        slabs.append([x0, y0, x1, min(y1, py0)])
    if y1 > py1:
        slabs.append([x0, max(y0, py1), x1, y1])
    iy0, iy1 = max(y0, py0), min(y1, py1)
    if iy1 > iy0:
        if x0 < px0:
            slabs.append([x0, iy0, min(x1, px0), iy1])
        if x1 > px1:
            slabs.append([max(x0, px1), iy0, x1, iy1])
    return [s for s in slabs if s[2] - s[0] > 0 and s[3] - s[1] > 0]


def draw_edge_diffs(draw, old, new, width: int, font_size: int,
                    bounds: Optional[Tuple[int, int]] = None) -> None:
    """逐边对比新旧框：有变化的边用品红加粗，并在边中点标注 旧值→新值"""
    names = ['左', '上', '右', '下']
    for i, name in enumerate(names):
        if old[i] == new[i]:
            continue
        if i == 0:
            seg = [new[0], new[1], new[0], new[3]]
        elif i == 1:
            seg = [new[0], new[1], new[2], new[1]]
        elif i == 2:
            seg = [new[2], new[1], new[2], new[3]]
        else:
            seg = [new[0], new[3], new[2], new[3]]
        draw.line(seg, fill=COLOR_EDGE, width=width * 3)
        mx, my = (seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2
        draw_tag_at(draw, mx, my, f'{name}边 {old[i]:.0f} → {new[i]:.0f}',
                    COLOR_EDGE, font_size, bounds)


def _text_size(text: str, font):
    probe = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    try:
        tb = probe.textbbox((0, 0), text, font=font)
        return tb[2] - tb[0], tb[3] - tb[1]
    except AttributeError:
        return probe.textsize(text, font=font)


def add_caption_strip(img, text: str, bg=(185, 28, 28), font_scale: float = 1.0):
    """在图片上方接一条说明带；画布过窄时加宽并折行，避免点开后文字被截断"""
    font = get_font(max(14, int(round(19 * font_scale))))
    canvas_w = max(img.width, MIN_STRIP_WIDTH)
    lines = _wrap_text(text, font, canvas_w - 24)
    sizes = [_text_size(ln, font) for ln in lines]
    th = sum(s[1] for s in sizes) + 4 * (len(lines) - 1)
    band_h = th + int(14 * font_scale)
    img = _pad_to_width(img, canvas_w, bg=(80, 20, 20))
    out = Image.new('RGB', (img.width, img.height + band_h), bg)
    out.paste(img, (0, band_h))
    d = ImageDraw.Draw(out)
    y = (band_h - th) / 2
    for ln, (lw, lh) in zip(lines, sizes):
        d.text(((img.width - lw) / 2, y), ln, fill=(255, 255, 255), font=font)
        y += lh + 4
    return out


def add_legend_strip(img, entries: List[Tuple[tuple, str, bool]],
                     font_scale: float = 1.0):
    """
    在图片上方接一条图例带：色块 + 说明文字。

    小图会先加宽到 MIN_STRIP_WIDTH，单条说明超出宽度则折行，
    保证点开灯箱后字能完整看见。
    """
    if not entries:
        return img
    fs = max(13, int(round(16 * font_scale)))
    font = get_font(fs)
    sw = fs + 2
    padx, gap, line_gap = 12, 16, 3
    canvas_w = max(img.width, MIN_STRIP_WIDTH)
    max_text_w = max(40, canvas_w - padx * 2 - sw - 7)

    packed = []
    for color, text, dashed in entries:
        lines = _wrap_text(text, font, max_text_w)
        sizes = [_text_size(ln, font) for ln in lines]
        tw = max(s[0] for s in sizes)
        th = sum(s[1] for s in sizes) + line_gap * (len(lines) - 1)
        packed.append((color, lines, dashed, sw + 7 + tw, max(sw, th), sizes))

    rows: List[List[tuple]] = [[]]
    cur = padx
    for item in packed:
        w = item[3]
        if rows[-1] and cur + w > canvas_w - padx:
            rows.append([])
            cur = padx
        rows[-1].append(item)
        cur += w + gap

    row_heights = [max(h for *_, h, _ in row) + 10 for row in rows]
    strip_h = sum(row_heights) + 8
    img = _pad_to_width(img, canvas_w)
    out = Image.new('RGB', (img.width, img.height + strip_h), (17, 24, 39))
    out.paste(img, (0, strip_h))
    d = ImageDraw.Draw(out)
    y = 4
    for row, rh in zip(rows, row_heights):
        x = padx
        for color, lines, dashed, w, h, sizes in row:
            cy = y + rh // 2
            box = [x, cy - sw // 2, x + sw, cy + sw // 2]
            if dashed:
                d.rectangle(box, outline=color, width=2)
            else:
                d.rectangle(box, fill=color)
            ty = y + (rh - h) // 2
            for ln, (_, lh) in zip(lines, sizes):
                d.text((x + sw + 7, ty), ln, fill=(255, 255, 255), font=font)
                ty += lh + line_gap
            x += w + gap
        y += rh
    return out


def collect_boxes(item: dict) -> List[Tuple[Sequence[int], str, tuple, bool]]:
    """
    根据待确认项的类型，挑出总览图上要画的框 —— 只画冲突本身，
    各方案的框另有独立效果图（render_option_image），不再叠在总览图上。
    返回 [(原图坐标, 标签, 颜色, 是否虚线), ...]
    """
    reason = item['reason']
    detail = item.get('detail') or {}
    boxes: List[Tuple[Sequence[int], str, tuple, bool]] = []

    if reason == 'clipped_deleted':
        boxes.append((item['target_bbox'],
                      f'原图当前框: {item.get("target_name_raw", "?")}（子图里已被人工删除）',
                      COLOR_ORIGINAL, False))

    elif reason == 'clipped_bbox_modified_border':
        boxes.append((item['target_bbox'], f'原图当前框: {item.get("target_name_raw", "?")}',
                      COLOR_ORIGINAL, False))

    elif reason == 'added_overlaps_existing':
        boxes.append((detail.get('existing_bbox', []),
                      f'原图已有: {detail.get("existing_name", "?")}', COLOR_ORIGINAL, False))
        boxes.append((item['add_bbox'], f'人工新增: {item["add_name"]}',
                      COLOR_PROPOSED, False))

    elif reason in ('added_out_of_scope', 'added_invalid_bbox'):
        if item.get('add_bbox'):
            boxes.append((item['add_bbox'], f'人工新增: {item.get("add_name", "?")}',
                          COLOR_PROPOSED, False))

    elif reason == 'cross_crop_inconsistent':
        boxes.append((item['target_bbox'], f'原图: {item.get("target_name_raw", "?")}',
                      COLOR_ORIGINAL, False))
        for cand in detail.get('candidates', []):
            if cand.get('new_bbox'):
                boxes.append((cand['new_bbox'],
                              f'{cand["action"]} @ {cand["sub_xml"][:28]}',
                              COLOR_ALT, True))

    elif reason == 'cross_crop_added_conflict':
        for cand in detail.get('candidates', []):
            n = cand.get('merged_from') or 1
            suffix = f'（{n} 张子图合并）' if n > 1 else ''
            boxes.append((cand['bbox'], f'新增 {cand["name"]}{suffix}',
                          COLOR_PROPOSED, True))

    elif reason in ('target_not_found', 'target_ambiguous', 'ledger_conflict'):
        if item.get('target_bbox'):
            boxes.append((item['target_bbox'],
                          f'目标: {item.get("target_name_raw") or item.get("target_name_mapped", "?")}',
                          COLOR_ORIGINAL, False))

    return [(b, label, color, dashed) for b, label, color, dashed in boxes
            if b and len(b) == 4]


def load_sub_panel(item: dict, sub_images: str):
    """加载对应的子图（多个子图时只取第一个），加载失败返回 None"""
    sub_name = (item.get('sub_xml') or '').split(' / ')[0]
    if not sub_name:
        return None
    sub_path = find_image(sub_images, os.path.splitext(sub_name)[0])
    if not sub_path:
        return None
    try:
        return Image.open(sub_path).convert('RGB')
    except Exception:
        return None


def save_panel(img, out_dir: str, rel_path: str, quality: int) -> str:
    dst = os.path.join(out_dir, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # progressive 让图片在慢连接下能由模糊到清晰地逐步显示，而不是一直空白
    img.save(dst, quality=quality, optimize=True, progressive=True)
    return rel_path


def fit_height(img, max_h: int):
    """把图缩到不超过 max_h 高，返回 (缩放后的图, 绘制时应放大的倍数)"""
    if img.height <= max_h:
        return img, 1.0
    scale = max_h / img.height
    return img.resize((max(1, int(img.width * scale)), max_h), Image.BILINEAR), 1 / scale


def render_locator(item: dict, base, out_dir: str, quality: int,
                   max_side: int = 1100) -> Optional[str]:
    """
    全图定位图：整张原图缩略，标出部件裁切区和缺陷位置。
    局部图为了看清细节会裁得很紧，裁切区框常常伸到视野外；这张图保证两者都完整可见。
    """
    box = item.get('target_bbox') or item.get('add_bbox')
    part = item.get('part_bbox')
    if not box:
        for cand in (item.get('detail') or {}).get('candidates') or []:
            box = cand.get('bbox') or cand.get('new_bbox')
            if box:
                break
    if not box and not part:
        return None

    scale = min(1.0, max_side / max(base.width, base.height))
    view = base.resize((max(1, int(base.width * scale)),
                        max(1, int(base.height * scale))), Image.BILINEAR)
    draw = ImageDraw.Draw(view)
    legend = []

    if part and any(part):
        pr = [v * scale for v in part]
        draw_dashed_rect(draw, pr, COLOR_CROP, width=3, dash=13, gap=9)
        legend.append((COLOR_CROP, '部件裁切区（子图范围）', True))

    if box:
        br = [v * scale for v in box]
        # 缺陷在整图里可能只有几个像素，撑到最小可见尺寸并拉十字线，方便一眼定位
        cx, cy = (br[0] + br[2]) / 2, (br[1] + br[3]) / 2
        if br[2] - br[0] < 16:
            br[0], br[2] = cx - 8, cx + 8
        if br[3] - br[1] < 16:
            br[1], br[3] = cy - 8, cy + 8
        for seg in ([0, cy, br[0], cy], [br[2], cy, view.width, cy],
                    [cx, 0, cx, br[1]], [cx, br[3], cx, view.height]):
            _dashed_line(draw, seg[0], seg[1], seg[2], seg[3],
                         COLOR_PROPOSED, 2, 11, 9)
        draw.rectangle(br, outline=COLOR_PROPOSED, width=3)
        legend.append((COLOR_PROPOSED, '缺陷位置（十字线指向）', False))

    view = add_legend_strip(view, legend)
    rel_path = os.path.join('images', f'{item["id"]:06d}_{item["reason"]}_full.jpg')
    return save_panel(view, out_dir, rel_path, quality)


def render_original_view(item: dict, base, out_dir: str, margin: float,
                         max_height: int, quality: int) -> Optional[str]:
    """原图局部：裁切区外压暗、被截断部分红色高亮"""
    boxes = collect_boxes(item)
    part_bbox = item.get('part_bbox')

    if boxes:
        xs = [b[0] for b, _, _, _ in boxes] + [b[2] for b, _, _, _ in boxes]
        ys = [b[1] for b, _, _, _ in boxes] + [b[3] for b, _, _, _ in boxes]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
    elif part_bbox:
        x0, y0, x1, y1 = part_bbox
    else:
        x0, y0, x1, y1 = 0, 0, base.width, base.height

    pad = max(80, int(max(x1 - x0, y1 - y0) * margin))
    vx0 = max(0, int(x0 - pad))
    vy0 = max(0, int(y0 - pad))
    vx1 = min(base.width, int(x1 + pad))
    vy1 = min(base.height, int(y1 + pad))

    # 裁切区没大太多时就把它整个收进来，让局部图和子图看的是同一片区域，便于左右对照；
    # 大太多则不强求，否则缺陷会被压得看不清（全图定位图里裁切区总是完整的）
    if part_bbox and any(part_bbox):
        if ((part_bbox[2] - part_bbox[0]) <= (vx1 - vx0) * 2.2
                and (part_bbox[3] - part_bbox[1]) <= (vy1 - vy0) * 2.2):
            vx0 = max(0, min(vx0, int(part_bbox[0]) - 24))
            vy0 = max(0, min(vy0, int(part_bbox[1]) - 24))
            vx1 = min(base.width, max(vx1, int(part_bbox[2]) + 24))
            vy1 = min(base.height, max(vy1, int(part_bbox[3]) + 24))

    if vx1 <= vx0 or vy1 <= vy0:
        return None

    view = base.crop((vx0, vy0, vx1, vy1))
    view, up = ensure_readable_size(view)
    # 出图会被压到 max_height，线宽按缩放比反推，免得缩小后细得看不见
    ds = max(1.0, view.height / min(max_height, view.height))
    line_w = max(2, int(round(4 * ds)))

    def rel(b):
        return [(b[0] - vx0) * up, (b[1] - vy0) * up,
                (b[2] - vx0) * up, (b[3] - vy0) * up]

    # 像素级操作先做：裁切区外压暗、被截断部分垫红
    part_rel = None
    if part_bbox and any(part_bbox):
        part_rel = rel(part_bbox)
        view, _ = dim_outside_part(view, part_rel)
    slab_rels = []
    if (item['reason'] in ('clipped_deleted', 'clipped_bbox_modified_border')
            and part_bbox and item.get('target_bbox')):
        slab_rels = [rel(s) for s in clipped_slabs(item['target_bbox'], part_bbox)]
        for s in slab_rels:
            view = fill_rect_alpha(view, s, CLIP_RGBA)

    draw = ImageDraw.Draw(view)
    legend = []

    for bbox, label, color, dashed in boxes:
        draw_box(draw, rel(bbox), color, width=line_w, dashed=dashed)
        legend.append((color, label, dashed))

    if part_rel:
        draw_dashed_rect(draw, part_rel, COLOR_CROP, width=line_w,
                         dash=int(16 * ds), gap=int(11 * ds))
        legend.append((COLOR_CROP, '部件裁切边界（外侧为暗区）', True))

    for s in slab_rels:
        draw.rectangle(s, outline=COLOR_PROPOSED, width=line_w)
    if slab_rels:
        legend.append((COLOR_PROPOSED, '被截断部分：子图里看不到', False))

    view, _ = fit_height(view, max_height)
    view = add_legend_strip(view, legend)
    rel_path = os.path.join('images', f'{item["id"]:06d}_{item["reason"]}_orig.jpg')
    return save_panel(view, out_dir, rel_path, quality)


def render_sub_view(item: dict, sub_images: str, out_dir: str,
                    max_height: int, quality: int) -> Optional[str]:
    """子图（人工清洗后）"""
    sub = load_sub_panel(item, sub_images)
    if sub is None:
        return None
    sub, up = ensure_readable_size(sub)

    ds = max(1.0, sub.height / min(max_height, sub.height))
    line_w = max(2, int(round(4 * ds)))
    draw = ImageDraw.Draw(sub)
    legend = []

    def scale_box(b):
        return [v * up for v in b]

    if item.get('bbox_in_crop'):
        draw_box(draw, scale_box(item['bbox_in_crop']), COLOR_PROPOSED, width=line_w)
        legend.append((COLOR_PROPOSED, f'人工新增: {item.get("add_name", "")}', False))
    elif (item.get('detail') or {}).get('new_bbox_in_crop'):
        draw_box(draw, scale_box(item['detail']['new_bbox_in_crop']), COLOR_PROPOSED,
                 width=line_w)
        legend.append((COLOR_PROPOSED, '人工修改后的框', False))

    sub, _ = fit_height(sub, max_height)
    sub = add_legend_strip(sub, legend)
    if item['reason'] == 'clipped_deleted':
        sub = add_caption_strip(sub, '子图中该框已被人工删除')
    rel_path = os.path.join('images', f'{item["id"]:06d}_{item["reason"]}_sub.jpg')
    return save_panel(sub, out_dir, rel_path, quality)


def render_item(item: dict, original_images: str, sub_images: str,
                out_dir: str, margin: float,
                max_height: int = 900, quality: int = 82) -> List[dict]:
    """
    渲染一条待确认项的三张图，各存一个文件、页面上并排显示、可各自放大：
    全图定位 / 原图局部 / 子图。返回 [{'title', 'path'}, ...]。
    """
    original_stem = os.path.splitext(item['original_xml'])[0]
    original_path = find_image(original_images, original_stem)
    if original_path is None:
        return []
    try:
        base = Image.open(original_path).convert('RGB')
    except Exception:
        return []

    # 旧版本把三张图拼成一张，重渲染后那个文件就成了孤儿，顺手清掉
    stale = os.path.join(out_dir, 'images',
                         f'{item["id"]:06d}_{item["reason"]}.jpg')
    if os.path.exists(stale):
        try:
            os.remove(stale)
        except OSError:
            pass

    panels = []
    for title, path in (
            ('全图定位（整张原图）',
             render_locator(item, base, out_dir, quality)),
            ('原图局部（冲突处）',
             render_original_view(item, base, out_dir, margin, max_height, quality)),
            ('子图（人工清洗后）',
             render_sub_view(item, sub_images, out_dir, max_height, quality))):
        if path:
            panels.append({'title': title, 'path': path})
    return panels


# ============================================================
#  方案效果图：每个决策选项一张图，点图即选
# ============================================================

def option_marks(option: dict) -> List[Tuple[Optional[list], list, str, bool]]:
    """从方案里提取要画的框：[(旧框或None, 结果框, 标签, 是否删除), ...]"""
    change = option.get('change') or {}
    changes = change.get('_multi') or [change]
    marks = []
    for ch in changes:
        act = ch.get('action')
        tb, nb = ch.get('target_bbox'), ch.get('new_bbox')
        if act == 'delete':
            if tb:
                marks.append((None, tb, '删除此框', True))
        elif act == 'add':
            if nb:
                marks.append((None, nb, f'新增: {ch.get("new_name", "")}', False))
        elif nb or tb:
            box = nb or tb
            label = f'采纳后: {ch["new_name"]}' if ch.get('new_name') else '采纳后'
            marks.append((tb, box, label, False))
    return marks


def render_option_image(item: dict, option: dict, base, out_dir: str,
                        margin: float = 0.45, max_height: int = 520,
                        quality: int = 82) -> Optional[str]:
    """
    为单个方案渲染效果图：旧框灰色虚线（当前），结果框红色实线（采纳后），
    新旧框之间有差异的边用品红加粗并标注数值变化 —— union/replace 这类
    只差几条边的方案，不标出来肉眼根本看不出区别。
    """
    marks = option_marks(option)
    if not marks:
        return None
    all_boxes = [b for pair in marks for b in pair[:2] if b]
    xs = [b[0] for b in all_boxes] + [b[2] for b in all_boxes]
    ys = [b[1] for b in all_boxes] + [b[3] for b in all_boxes]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

    # 方案图收紧裁剪窗口，让边与边的细微差异尽量显眼
    pad = max(60, int(max(x1 - x0, y1 - y0) * margin))
    vx0 = max(0, int(x0 - pad))
    vy0 = max(0, int(y0 - pad))
    vx1 = min(base.width, int(x1 + pad))
    vy1 = min(base.height, int(y1 + pad))
    if vx1 <= vx0 or vy1 <= vy0:
        return None

    view = base.crop((vx0, vy0, vx1, vy1))
    view, up = ensure_readable_size(view)
    target_h = min(max_height, view.height)
    ds = max(1.0, view.height / target_h)
    line_w = max(2, int(round(3 * ds)))
    font_sz = max(12, int(round(15 * ds)))

    def rel(b):
        return [(b[0] - vx0) * up, (b[1] - vy0) * up,
                (b[2] - vx0) * up, (b[3] - vy0) * up]

    part_bbox = item.get('part_bbox')
    has_part = bool(part_bbox) and any(part_bbox)
    if has_part:
        view, _ = dim_outside_part(view, rel(part_bbox))

    draw = ImageDraw.Draw(view)
    if has_part:
        draw_dashed_rect(draw, rel(part_bbox), COLOR_CROP,
                         width=max(2, int(2 * ds)),
                         dash=int(14 * ds), gap=int(10 * ds))

    legend = []
    for old, new_box, label, is_delete in marks:
        if old and not is_delete:
            draw_box(draw, rel(old), COLOR_CURRENT, width=line_w, dashed=True)
            legend.append((COLOR_CURRENT, '当前框（改动前）', True))
        draw_box(draw, rel(new_box), COLOR_PROPOSED, width=line_w)
        legend.append((COLOR_PROPOSED, label, False))
        if is_delete:
            draw_cross(draw, rel(new_box), COLOR_PROPOSED, width=line_w)
        elif old and list(old) != list(new_box):
            draw_edge_diffs(draw, rel(old), rel(new_box), line_w, font_sz,
                            view.size)
            legend.append((COLOR_EDGE, '有变化的边（标注在边上）', False))

    scale = target_h / view.height
    view = view.resize((max(1, int(view.width * scale)), target_h), Image.BILINEAR)
    view = add_legend_strip(view, legend)

    key = re.sub(r'[^A-Za-z0-9_-]+', '_', str(option.get('key', 'opt')))
    rel_path = os.path.join('images',
                            f'{item["id"]:06d}_{item["reason"]}_opt_{key}.jpg')
    dst = os.path.join(out_dir, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    view.save(dst, quality=quality, optimize=True, progressive=True)
    return rel_path


def render_option_images(item: dict, original_images: str, out_dir: str,
                         quality: int = 82) -> None:
    """为每个决策方案渲染独立效果图，路径写回 option['_image']"""
    options = item.get('options') or []
    if not options:
        return
    original_stem = os.path.splitext(item['original_xml'])[0]
    original_path = find_image(original_images, original_stem)
    if not original_path:
        return
    try:
        base = Image.open(original_path).convert('RGB')
    except Exception:
        return
    for opt in options:
        try:
            opt['_image'] = render_option_image(item, opt, base, out_dir,
                                                quality=quality)
        except Exception:
            opt['_image'] = None


# ============================================================
#  HTML 审核页
# ============================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>缺陷映射人工确认 - __DATASET__</title>
<style>
 body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;margin:0;background:#f5f6f8;color:#222}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 20px;z-index:10;
        box-shadow:0 1px 4px rgba(0,0,0,.06)}
 h1{font-size:17px;margin:0 0 8px}
 .bar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;font-size:13px}
 button{padding:6px 14px;border:1px solid #bbb;background:#fff;border-radius:5px;cursor:pointer;font-size:13px}
 button:hover{background:#f1f5f9}
 button.primary{background:#2563eb;color:#fff;border-color:#2563eb}
 button.primary:hover{background:#1d4ed8}
 button.danger{color:#b91c1c;border-color:#fca5a5}
 select{padding:5px;font-size:13px;border-radius:5px;border:1px solid #bbb}
 #status{font-size:12px;padding:3px 9px;border-radius:10px}
 #status.online{background:#dcfce7;color:#166534}
 #status.offline{background:#fef3c7;color:#92400e}
 #saveHint{font-size:12px;color:#16a34a;min-width:74px}
 .item{background:#fff;margin:14px 20px;border-radius:8px;padding:14px 18px;
       box-shadow:0 1px 3px rgba(0,0,0,.08);scroll-margin-top:110px}
 .item.done{border-left:5px solid #16a34a}
 .item.rejected{border-left:5px solid #9ca3af;opacity:.6}
 .item.current{outline:2px solid #2563eb}
 .reason{display:inline-block;background:#fee2e2;color:#991b1b;padding:2px 9px;border-radius:4px;
         font-size:12px;font-weight:600}
 .meta{font-size:12px;color:#666;margin:8px 0;line-height:1.7;word-break:break-all}
 .meta b{color:#333}
 img{max-width:100%;border:1px solid #ddd;border-radius:5px;margin:8px 0;cursor:zoom-in}
 .panels{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}
 figure.panel{margin:0;border:1px solid #cbd5e1;border-radius:7px;overflow:hidden;
              background:#0f172a}
 figure.panel img{display:block;max-height:360px;max-width:640px;width:auto;margin:0;
                  border:none;border-radius:0;cursor:zoom-in;object-fit:contain}
 figure.panel figcaption{font-size:12px;color:#e2e8f0;background:#1e293b;padding:4px 9px}
 #lightbox{position:fixed;inset:0;background:rgba(0,0,0,.95);z-index:300;display:none}
 #lightbox.on{display:flex;align-items:center;justify-content:center}
 #lightbox.on.natural{display:block;overflow:auto}
 #lbimg{margin:12px auto;border:none;border-radius:0;cursor:zoom-in;
        max-width:100vw;max-height:92vh;object-fit:contain}
 #lightbox.natural #lbimg{max-width:none;max-height:none;cursor:zoom-out}
 #lbcap{position:fixed;left:0;right:0;bottom:0;background:rgba(15,23,42,.92);color:#fff;
        font-size:13px;padding:8px 16px;text-align:center}
 .reason-cn{font-size:15px;font-weight:700;color:#991b1b}
 .reason-key{font-size:11px;color:#94a3b8;margin-left:8px;font-family:monospace}
 .info{display:grid;grid-template-columns:70px 1fr;gap:3px 12px;font-size:13px;margin:10px 0;
       background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px 14px}
 .info dt{color:#64748b;white-space:nowrap}
 .info dd{margin:0;color:#1e293b;word-break:break-all}
 details.raw{font-size:12px;color:#64748b;margin:4px 0}
 details.raw summary{cursor:pointer}
 details.raw pre{white-space:pre-wrap;word-break:break-all;background:#f8fafc;padding:8px;
                 border-radius:5px;max-height:220px;overflow:auto}
 .opts{display:flex;gap:12px;flex-wrap:wrap;margin-top:12px}
 .optcard{width:280px;border:2px solid #e2e8f0;border-radius:8px;background:#fff;
          position:relative;overflow:hidden}
 .optcard:hover{border-color:#93c5fd;box-shadow:0 2px 8px rgba(37,99,235,.15)}
 .optcard.reject{cursor:pointer}
 .optcard .optlabel{cursor:pointer}
 .optcard .optlabel:hover{background:#eff6ff}
 .optcard.sel{border-color:#2563eb;box-shadow:0 0 0 3px #bfdbfe}
 .optcard.sel::after{content:'✓ 已选';position:absolute;top:6px;right:6px;background:#2563eb;
          color:#fff;font-size:11px;padding:2px 8px;border-radius:10px}
 .optcard.reject.sel{border-color:#6b7280;box-shadow:0 0 0 3px #d1d5db}
 .optcard.reject.sel::after{background:#6b7280;content:'✓ 已驳回'}
 .optcard .thumb{width:100%;height:200px;object-fit:contain;background:#1e293b;display:block;
          margin:0;border:none;border-radius:0;cursor:zoom-in}
 .optcard .noimg{height:200px;display:flex;align-items:center;justify-content:center;text-align:center;
          background:#f1f5f9;color:#64748b;font-size:13px;padding:0 14px;line-height:1.8}
 .optcard .optlabel{padding:8px 10px;font-size:12.5px;line-height:1.55;border-top:1px solid #eef2f7}
 .kbd{display:inline-block;min-width:15px;text-align:center;background:#e2e8f0;border-radius:3px;
      padding:0 4px;margin-right:7px;font-size:11px;color:#475569;font-family:monospace}
 .legend{font-size:12px;color:#555}
 .sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin:0 4px 0 10px;vertical-align:middle}
 .hidden{display:none}
 #toast{position:fixed;right:20px;bottom:20px;background:#111827;color:#fff;padding:12px 18px;
        border-radius:7px;font-size:13px;max-width:520px;white-space:pre-wrap;z-index:200;
        box-shadow:0 4px 14px rgba(0,0,0,.3);display:none;line-height:1.6}
</style></head><body>
<header>
  <h1>缺陷映射人工确认 &mdash; __DATASET__ &mdash; 共 __COUNT__ 条</h1>
  <div class="bar">
    <span id="status" class="offline">检测中…</span>
    <span id="progress"></span>
    <span id="saveHint"></span>
    <select id="filter">
      <option value="all">全部</option>
      <option value="todo">仅未决策</option>
      <option value="done">仅已决策</option>
      __REASON_OPTIONS__
    </select>
    <button onclick="gotoNext()">跳到下一条未决策 (n)</button>
    <button class="primary" id="buildBtn" onclick="buildChangeset()">生成补充变更集</button>
    <button onclick="exportJSON()">下载 decisions.json</button>
    <button class="danger" onclick="clearAll()">清空决策</button>
  </div>
  <div class="bar" style="margin-top:6px">
    <span class="legend">
      <span class="sw" style="background:#00c800"></span>原图已有框
      <span class="sw" style="background:#ff2828"></span>冲突点/采纳后
      <span class="sw" style="background:#787878"></span>当前框(虚线)
      <span class="sw" style="background:#dc00dc"></span>有变化的边
      <span class="sw" style="background:#ffaa00"></span>其他候选
      <span class="sw" style="background:#00b4ff"></span>部件裁切边界
      &nbsp;&nbsp;快捷键：<span class="kbd">←</span>/<span class="kbd">→</span> 上下条
      <span class="kbd">1</span>~<span class="kbd">9</span> 选方案
      <span class="kbd">0</span> 驳回
      <span class="kbd">n</span> 下一条未决策
      &nbsp;&nbsp;点卡片文字条选方案；图片单击放大，放大后再单击看 1:1 原始像素
    </span>
  </div>
</header>
<div id="list"></div>
<div id="lightbox" onclick="closeLightbox()">
  <img id="lbimg" onclick="toggleNatural(event)"><div id="lbcap"></div>
</div>
<div id="toast"></div>
<script>
const DATA = __DATA__;
const DATASET = __DATASET_JSON__;
const IMG_V = '__IMG_V__';
const KEY = 'defect_mapping_decisions_' + DATASET;
function imgSrc(p){
  if(!p) return '';
  return p + (p.indexOf('?') >= 0 ? '&' : '?') + 'v=' + IMG_V;
}
let decisions = {};
let serverOn = false;
let current = 0;

function toast(msg, ms){
  const el = document.getElementById('toast');
  el.textContent = msg; el.style.display = 'block';
  clearTimeout(el._t); el._t = setTimeout(() => el.style.display = 'none', ms || 4000);
}

// ---------- 决策的读写：优先服务端，退化到 localStorage ----------

// 带超时的 fetch。没有超时的话，一旦服务端没正常响应，页面就会永远停在「检测中」。
async function fetchTimeout(url, opts, ms){
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms || 6000);
  try{
    return await fetch(url, Object.assign({signal: ctl.signal, cache:'no-store'}, opts || {}));
  } finally { clearTimeout(timer); }
}

function setStatus(cls, text, title){
  const st = document.getElementById('status');
  st.className = cls; st.textContent = text; st.title = title || '';
}

function offlineMode(reason){
  serverOn = false;
  setStatus('offline', '未连接服务器 · 决策仅存在本浏览器',
            reason + '（用 serve 子命令起服务可把决策直接存到服务器磁盘）');
  const btn = document.getElementById('buildBtn');
  btn.disabled = true;
  btn.title = '需要连接服务器才能直接生成补充变更集';
}

// 关键：先用本地决策把列表画出来，再去后台探测服务器。
// 绝不能让内容的显示依赖网络请求，否则服务端一慢，整个页面就是空白。
function boot(){
  decisions = JSON.parse(localStorage.getItem(KEY) || '{}');
  render();
  probeServer();
}

async function probeServer(){
  let info;
  try{
    const r = await fetchTimeout('api/ping', {}, 6000);
    if(!r.ok) throw new Error('HTTP ' + r.status);
    info = await r.json();
  }catch(e){
    offlineMode(e.name === 'AbortError' ? '探测服务器超时' : ('探测失败: ' + e.message));
    return;
  }

  serverOn = true;
  setStatus('online', '已连接服务器 · 决策实时保存',
            '决策文件: ' + (info.decisions_file || ''));
  if(!info.original_xml_dir_exists){
    const btn = document.getElementById('buildBtn');
    btn.disabled = true;
    btn.title = '原图 XML 目录不可用: ' + info.original_xml_dir;
  }

  // 服务端的决策才是权威版本，拉回来后合并并重画选中状态
  try{
    const r = await fetchTimeout('api/decisions', {}, 10000);
    const d = await r.json();
    const remote = d.decisions || {};
    const localCount = Object.keys(decisions).length;
    const remoteCount = Object.keys(remote).length;
    if(remoteCount || !localCount){
      decisions = remote;
      applyChecked();
      refresh();
      if(remoteCount) toast(`已从服务器载入 ${remoteCount} 条既有决策`, 3000);
    }else if(localCount){
      // 本地有、服务端没有：把本地的推上去，避免丢失
      save();
      toast(`已把本浏览器的 ${localCount} 条决策同步到服务器`, 4000);
    }
  }catch(e){
    toast('读取服务器决策失败：' + e.message, 6000);
  }
}

function applyChecked(){ refresh(); }

let saveTimer = null;
function save(){
  localStorage.setItem(KEY, JSON.stringify(decisions));
  refresh();
  if(!serverOn) return;
  document.getElementById('saveHint').textContent = '保存中…';
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    try{
      const r = await fetchTimeout('api/decisions', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({decisions: decisions})
      }, 10000);
      const j = await r.json();
      document.getElementById('saveHint').textContent = j.ok ? '已保存 ✓' : '保存失败!';
    }catch(e){
      document.getElementById('saveHint').textContent = '保存失败!';
      toast('决策保存到服务器失败：' + e.message
          + '\\n决策仍保留在本浏览器里，可点「下载 decisions.json」导出。', 8000);
    }
  }, 350);
}

function choose(id, val){
  if(val === '') delete decisions[id]; else decisions[id] = val;
  save();
}

function clearAll(){
  if(!confirm('确定清空所有决策？')) return;
  decisions = {};
  localStorage.removeItem(KEY);
  save();
}

async function buildChangeset(){
  if(!confirm('把当前决策生成为补充变更集？\\n（只写变更集文件，不会修改原图）')) return;
  const btn = document.getElementById('buildBtn');
  btn.disabled = true; btn.textContent = '生成中…';
  try{
    const r = await fetchTimeout('api/build-changeset', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({decisions: decisions})
    }, 180000);
    const j = await r.json();
    if(j.ok){
      const acts = Object.entries(j.actions || {}).map(([k,v]) => k+'='+v).join('  ');
      toast(`补充变更集已生成\\n采纳 ${j.accepted} 条，驳回 ${j.rejected} 条\\n`
          + `涉及 ${j.file_count} 个原图 XML\\n${acts}\\n\\n输出: ${j.output}\\n`
          + `下一步在服务器上执行（先预演）:\\n${j.next_command}`, 20000);
    }else{
      toast('生成失败：' + j.error, 12000);
    }
  }catch(e){ toast('生成失败：' + e, 12000); }
  btn.disabled = false; btn.textContent = '生成补充变更集';
}

function exportJSON(){
  const out = { dataset: DATASET, exported_at: new Date().toISOString(),
                decisions: decisions };
  const blob = new Blob([JSON.stringify(out, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'decisions.json';
  a.click();
}

// ---------- 渲染 ----------

function escapeHtml(s){ return String(s).replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function fmtBbox(b){
  if(!b || b.length !== 4) return '-';
  return '[' + b.map(v => Math.round(v)).join(', ') + ']';
}

// ---------- 图片放大（灯箱） ----------

function openLightbox(src, cap){
  const lb = document.getElementById('lightbox');
  lb.className = 'on';
  document.getElementById('lbimg').src = src;
  document.getElementById('lbcap').textContent =
    cap + '　·　单击图片切换 1:1 原始像素　·　Esc 或点背景关闭';
}
function closeLightbox(){ document.getElementById('lightbox').className = ''; }
function lightboxOpen(){ return document.getElementById('lightbox').className !== ''; }
function toggleNatural(e){
  e.stopPropagation();
  document.getElementById('lightbox').classList.toggle('natural');
}
// 说明文字走 data 属性而不是拼进 onclick 里：属性值会被 HTML 解码后再交给 JS 解析，
// 标签里只要出现引号就会把内联脚本弄坏
function zoom(el){ openLightbox(el.src, el.dataset.cap || ''); }

function render(){
  const list = document.getElementById('list');
  list.innerHTML = '';
  DATA.forEach(it => {
    const div = document.createElement('div');
    div.className = 'item'; div.id = 'item-' + it.id;

    // 决策方案渲染成图片卡片。图片单击只负责放大预览，点底部文字条才选中方案 ——
    // 如果点图即选，双击放大会先触发两次单击把方案选上，职责分开就没这个问题
    let cards = `<div class="optcard reject" data-key="reject" onclick="choose(${it.id},'reject')">
        <div class="noimg">原图保持不变<br>不采纳任何改动</div>
        <div class="optlabel"><span class="kbd">0</span>驳回</div></div>`;
    (it.options || []).forEach((o, i) => {
      const cap = escapeHtml('#' + it.id + ' 方案 ' + (i+1) + '：' + o.label);
      const img = o.image
        ? `<img class="thumb" src="${imgSrc(o.image)}" loading="lazy" title="单击放大预览"
             data-cap="${cap}" onclick="zoom(this)">`
        : '<div class="noimg">（该方案无预览图）</div>';
      cards += `<div class="optcard" data-key="${o.key}">
        ${img}<div class="optlabel" title="点击选择该方案" onclick="choose(${it.id},'${o.key}')"><span class="kbd">${i+1}</span>${escapeHtml(o.label)}</div></div>`;
    });

    const nameHtml = it.add_name
      ? `新增类别 <b>${escapeHtml(it.add_name)}</b>`
      : `缺陷类别 <b>${escapeHtml(it.target_name_raw || '-')}</b>`
        + ((it.target_name_mapped && it.target_name_mapped !== it.target_name_raw)
           ? `（映射后 ${escapeHtml(it.target_name_mapped)}）` : '');

    div.innerHTML = `<span class="reason-cn">${escapeHtml(it.reason_cn || it.reason)}</span>
      <span class="reason-key">${escapeHtml(it.reason)} · #${it.id}</span>
      <div class="meta">${escapeHtml(it.reason_text || '')}</div>
      <dl class="info">
        <dt>类别</dt><dd>${nameHtml}</dd>
        <dt>框坐标</dt><dd>${fmtBbox(it.target_bbox || it.add_bbox)}</dd>
        <dt>原图</dt><dd>${escapeHtml(it.original_xml)}</dd>
        <dt>子图</dt><dd>${escapeHtml(it.sub_xml || '-')}</dd>
      </dl>
      ${(it.panels || []).length
        ? `<div class="panels">` + it.panels.map(p =>
            `<figure class="panel"><img src="${imgSrc(p.path)}" loading="lazy" title="单击放大"
               data-cap="${escapeHtml('#' + it.id + ' ' + p.title)}" onclick="zoom(this)">
             <figcaption>${escapeHtml(p.title)}</figcaption></figure>`).join('') + `</div>`
        : '<div class="meta">（无法渲染对比图：未找到对应图片）</div>'}
      <details class="raw"><summary>原始详情 JSON</summary><pre>${escapeHtml(JSON.stringify(it.detail || {}, null, 1))}</pre></details>
      <div class="opts">${cards}</div>`;
    list.appendChild(div);
  });
  refresh();
}

function refresh(){
  const done = Object.keys(decisions).length;
  document.getElementById('progress').textContent = `已决策 ${done} / ${DATA.length}`;
  const f = document.getElementById('filter').value;
  DATA.forEach((it, idx) => {
    const el = document.getElementById('item-' + it.id);
    const d = decisions[it.id];
    el.className = 'item' + (d === 'reject' ? ' rejected' : (d ? ' done' : ''))
                 + (idx === current ? ' current' : '');
    el.querySelectorAll('.optcard').forEach(c => {
      c.classList.toggle('sel', !!d && c.dataset.key === d);
    });
    let show = true;
    if(f === 'todo') show = !d;
    else if(f === 'done') show = !!d;
    else if(f !== 'all') show = (it.reason === f);
    el.classList.toggle('hidden', !show);
  });
}

// ---------- 键盘操作 ----------

function focusItem(idx){
  if(idx < 0 || idx >= DATA.length) return;
  current = idx;
  refresh();
  const el = document.getElementById('item-' + DATA[idx].id);
  if(el && !el.classList.contains('hidden')) el.scrollIntoView({block:'start', behavior:'smooth'});
}

function step(dir){
  let i = current + dir;
  while(i >= 0 && i < DATA.length){
    const el = document.getElementById('item-' + DATA[i].id);
    if(el && !el.classList.contains('hidden')){ focusItem(i); return; }
    i += dir;
  }
}

function gotoNext(){
  for(let i = 0; i < DATA.length; i++){
    const idx = (current + 1 + i) % DATA.length;
    if(!decisions[DATA[idx].id]){ focusItem(idx); return; }
  }
  toast('所有条目都已决策完毕');
}

function pick(n){
  const it = DATA[current];
  if(!it) return;
  const key = (n === 0) ? 'reject' : ((it.options || [])[n-1] || {}).key;
  if(!key) return;
  choose(it.id, key);
  setTimeout(gotoNext, 120);
}

document.addEventListener('keydown', e => {
  if(e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if(e.key === 'Escape'){ closeLightbox(); return; }
  // 放大态下不让上下条/选方案的快捷键生效，避免看图时误操作
  if(lightboxOpen()) return;
  if(e.key === 'ArrowRight'){ step(1); e.preventDefault(); }
  else if(e.key === 'ArrowLeft'){ step(-1); e.preventDefault(); }
  else if(e.key === 'n'){ gotoNext(); e.preventDefault(); }
  else if(e.key >= '0' && e.key <= '9'){ pick(parseInt(e.key)); e.preventDefault(); }
});

document.getElementById('filter').addEventListener('change', () => { current = 0; refresh(); });
boot();
</script></body></html>
"""


def build_html(items: List[dict], dataset: str, out_dir: str) -> str:
    slim = []
    for item in items:
        slim.append({
            'id': item['id'],
            'reason': item['reason'],
            'reason_cn': REASON_CN.get(item['reason'], item['reason']),
            'reason_text': item.get('reason_text', ''),
            'original_xml': item.get('original_xml', ''),
            'sub_xml': item.get('sub_xml', ''),
            'target_name_raw': item.get('target_name_raw'),
            'target_name_mapped': item.get('target_name_mapped'),
            'add_name': item.get('add_name'),
            'target_bbox': item.get('target_bbox'),
            'add_bbox': item.get('add_bbox'),
            'detail': item.get('detail', {}),
            'options': [{'key': o['key'], 'label': o['label'], 'image': o.get('_image')}
                        for o in (item.get('options') or [])],
            'panels': item.get('_panels') or [],
        })

    counts = Counter(item['reason'] for item in items)
    reason_options = ''.join(
        f'<option value="{html.escape(r)}">只看 {html.escape(REASON_CN.get(r, r))}'
        f'（{n}）</option>' for r, n in counts.most_common())

    page = (HTML_TEMPLATE
            .replace('__DATA__', json.dumps(slim, ensure_ascii=False))
            .replace('__DATASET_JSON__', json.dumps(dataset, ensure_ascii=False))
            .replace('__DATASET__', html.escape(dataset))
            .replace('__COUNT__', str(len(items)))
            .replace('__REASON_OPTIONS__', reason_options)
            .replace('__IMG_V__', str(int(time.time()))))

    path = os.path.join(out_dir, 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page)
    return path


# ============================================================
#  子命令：render
# ============================================================

def resolve_review_context(review_queue: str, base_changeset: Optional[str],
                           original_xml: Optional[str]) -> dict:
    """
    推断审核所需的上下文：基础变更集在哪、原图 XML 目录在哪、数据集叫什么。
    默认约定 changeset.json 与 review_queue.json 同目录。
    """
    queue_dir = os.path.dirname(os.path.abspath(review_queue))
    if not base_changeset:
        candidate = os.path.join(queue_dir, 'changeset.json')
        base_changeset = candidate if os.path.exists(candidate) else None

    meta = {}
    if base_changeset and os.path.exists(base_changeset):
        try:
            meta = mc.load_json(base_changeset).get('meta', {})
        except Exception:
            meta = {}

    return {
        'dataset': meta.get('dataset'),
        'component_type': meta.get('component_type'),
        'review_queue': os.path.abspath(review_queue),
        'base_changeset': os.path.abspath(base_changeset) if base_changeset else None,
        'original_xml_dir': original_xml or meta.get('original_xml_dir'),
        'ledger': meta.get('ledger'),
        'supplement_output': os.path.join(queue_dir, 'supplement_changeset.json'),
    }


def cmd_render(args) -> int:
    if Image is None:
        print('❌ 需要 Pillow 才能渲染对比图: pip install pillow')
        return 1

    items = mc.load_json(args.review_queue)
    if not items:
        print('待确认队列为空，无需审核。')
        return 0

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    sub_images = args.sub_images or (str(dataset_dir / 'images') if dataset_dir else '')
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    if args.reason:
        items = [it for it in items if it['reason'] in args.reason]
        print(f'按原因过滤后剩余 {len(items)} 条')
    if args.limit:
        items = items[:args.limit]

    print(f'开始渲染 {len(items)} 条待确认项 ...')
    rendered = failed = 0
    for i, item in enumerate(items, 1):
        if i % 200 == 0:
            print(f'  {i}/{len(items)}')
        try:
            panels = render_item(item, args.original_images, sub_images, out_dir,
                                 args.margin, args.max_height, args.quality)
            render_option_images(item, args.original_images, out_dir, args.quality)
        except Exception as exc:
            panels = []
            if failed < 5:
                print(f'  ⚠️  渲染失败 #{item["id"]}: {exc}')
        item['_panels'] = panels
        if panels:
            rendered += 1
        else:
            failed += 1

    path = build_html(items, args.dataset or (dataset_dir.name if dataset_dir else 'review'),
                      out_dir)

    # 写出审核上下文，serve 子命令据此定位队列、基础变更集和原图目录
    context = resolve_review_context(args.review_queue, args.base_changeset,
                                     args.original_xml)
    context['created_at'] = mc.now_iso()
    context['decisions'] = os.path.join(os.path.abspath(out_dir), 'decisions.json')
    if not context.get('dataset'):
        context['dataset'] = args.dataset or (dataset_dir.name if dataset_dir else 'review')
    mc.dump_json(context, os.path.join(out_dir, 'review_meta.json'))

    counter = Counter(item['reason'] for item in items)
    print(f'\n渲染完成: 成功 {rendered}，失败 {failed}')
    print('按原因分布:')
    for reason, count in counter.most_common():
        print(f'  {reason:32s} {count}')
    print(f'\n审核页面: {path}')

    if args.serve:
        return serve_review(out_dir, host=args.host, port=args.port,
                            open_browser=args.open_browser)

    print('\n两种打开方式:')
    print(f'  1) 直接用浏览器打开 {path}（决策存在浏览器 localStorage，需手动导出）')
    print('  2) 起服务，远程也能用（推荐）:')
    print(f'     python 4_审核可视化.py serve --review-dir {out_dir}')
    return 0


# ============================================================
#  子命令：build-changeset
# ============================================================

def _merge_change_meta(dst: dict, src: dict) -> None:
    sources = list(dst.get('sources') or [])
    for s in (src.get('sources') or []):
        if s and s not in sources:
            sources.append(s)
    dst['sources'] = sources
    ids = list(dst.get('review_ids') or ([dst['review_id']] if 'review_id' in dst else []))
    for rid in (src.get('review_ids') or ([src['review_id']] if 'review_id' in src else [])):
        if rid not in ids:
            ids.append(rid)
    if ids:
        dst['review_ids'] = ids


def _modify_outcome(change: dict) -> Tuple:
    bbox = change.get('new_bbox')
    return (change.get('action'), change.get('new_name'),
            tuple(bbox) if bbox else None)


def dedupe_supplement_changes(xml_name: str, changes: List[dict],
                              stats: Counter, problems: List[str],
                              dup_iou: float = SUPPLEMENT_DUP_IOU) -> List[dict]:
    """
    旧审核队列里，同一位置被多个子图各出一张卡。两张都选 add / retag
    时会写出重复变更。回灌时再收一次：同类重叠只留一条，retag 后不再加同名框。
    """
    if len(changes) <= 1:
        return changes

    modifies = [c for c in changes
                if c.get('action') in ('set_name', 'set_bbox', 'set_name_bbox', 'delete')]
    adds = [c for c in changes if c.get('action') == 'add']
    others = [c for c in changes
              if c.get('action') not in ('set_name', 'set_bbox', 'set_name_bbox', 'delete', 'add')]

    collapsed: List[dict] = []
    by_target: Dict[Tuple, List[dict]] = {}
    order: List[Tuple] = []
    for change in modifies:
        key = (change.get('target_name'), tuple(change.get('target_bbox') or ()))
        if key not in by_target:
            order.append(key)
            by_target[key] = []
        by_target[key].append(change)

    for key in order:
        group = by_target[key]
        keeper = dict(group[0])
        outcomes = {_modify_outcome(c) for c in group}
        if len(group) > 1 and len(outcomes) > 1:
            problems.append(
                f'{xml_name} 同一原框 {key} 有互相矛盾的审核决策 {sorted(outcomes)}，'
                f'已只保留条目 #{keeper.get("review_id")} 的方案')
        for extra in group[1:]:
            _merge_change_meta(keeper, extra)
            stats['supplement_merged_modify'] += 1
        collapsed.append(keeper)

    merged_adds: List[dict] = []
    if adds:
        groups = mc.cluster_indices(len(adds), lambda i, j: (
            adds[i].get('new_name') == adds[j].get('new_name')
            and mc.iou(adds[i].get('new_bbox') or [0, 0, 0, 0],
                       adds[j].get('new_bbox') or [0, 0, 0, 0]) >= dup_iou
        ))
        for idxs in groups:
            keeper = dict(adds[idxs[0]])
            for k in idxs[1:]:
                _merge_change_meta(keeper, adds[k])
                stats['supplement_merged_add'] += 1
            merged_adds.append(keeper)

    retags = [c for c in collapsed if c.get('action') in ('set_name', 'set_name_bbox')]
    kept_adds: List[dict] = []
    for add in merged_adds:
        add_bbox = tuple(add.get('new_bbox') or ())
        add_name = add.get('new_name')
        dropped = False
        if add_bbox and len(add_bbox) == 4:
            for retag in retags:
                target = tuple(retag.get('target_bbox') or ())
                if (len(target) == 4 and add_name == retag.get('new_name')
                        and mc.iou(add_bbox, target) >= dup_iou):
                    _merge_change_meta(retag, add)
                    stats['supplement_dropped_add_after_retag'] += 1
                    dropped = True
                    break
        if not dropped:
            kept_adds.append(add)

    return collapsed + kept_adds + others


def build_supplement_changeset(review_queue_path: str, decisions: dict,
                               original_dir: str, base_meta: dict,
                               decisions_path: str = '') -> Tuple[dict, Counter, List[str]]:
    """
    把人工决策组装成补充变更集。

    被命令行的 build-changeset 子命令和审核服务器的 /api/build-changeset 共用，
    保证两条路径产出完全一致的结果。
    """
    items = {item['id']: item for item in mc.load_json(review_queue_path)}
    files: Dict[str, dict] = {}
    stats: Counter = Counter()
    problems: List[str] = []

    for raw_id, choice in decisions.items():
        try:
            item_id = int(raw_id)
        except (TypeError, ValueError):
            problems.append(f'无法解析的条目编号: {raw_id}')
            continue

        item = items.get(item_id)
        if item is None:
            problems.append(f'条目 #{item_id} 不在待确认队列中，已跳过')
            continue

        if choice == 'reject':
            stats['rejected'] += 1
            continue

        option = next((o for o in (item.get('options') or []) if o['key'] == choice), None)
        if option is None:
            problems.append(f'条目 #{item_id} 选择了不存在的方案 "{choice}"')
            continue

        change = option['change']
        changes = change.get('_multi') or [change]
        xml_name = item['original_xml']
        path = os.path.join(original_dir, xml_name)
        if not os.path.exists(path):
            problems.append(f'条目 #{item_id} 对应的原图 XML 不存在: {xml_name}')
            continue

        bucket = files.setdefault(xml_name, {'md5': mc.md5_of(path), 'changes': []})
        for one in changes:
            entry = {k: v for k, v in one.items() if k != '_multi'}
            entry['review_id'] = item_id
            bucket['changes'].append(entry)
        stats['accepted'] += 1

    for xml_name, bucket in files.items():
        bucket['changes'] = dedupe_supplement_changes(
            xml_name, bucket['changes'], stats, problems)

    for action in ('set_name', 'set_bbox', 'set_name_bbox', 'delete', 'add'):
        stats[action] = 0
    for bucket in files.values():
        for ch in bucket['changes']:
            stats[ch.get('action', '')] += 1

    changeset = {
        'version': 1,
        'meta': {
            'created_at': mc.now_iso(),
            'dataset': base_meta.get('dataset'),
            'component_type': base_meta.get('component_type'),
            'original_xml_dir': os.path.abspath(original_dir),
            'ledger': base_meta.get('ledger') or mc.ledger_path_for(original_dir),
            'source': '人工审核决策（4_审核可视化.py）',
            'review_queue': os.path.abspath(review_queue_path),
            'decisions': os.path.abspath(decisions_path) if decisions_path else None,
        },
        'files': files,
    }
    return changeset, stats, problems


def cmd_build_changeset(args) -> int:
    raw = mc.load_json(args.decisions)
    decisions = raw.get('decisions', raw)

    base_meta = {}
    original_dir = args.original_xml
    if args.base_changeset and os.path.exists(args.base_changeset):
        base = mc.load_json(args.base_changeset)
        base_meta = base.get('meta', {})
        original_dir = original_dir or base_meta.get('original_xml_dir')
    if not base_meta.get('dataset'):
        base_meta['dataset'] = raw.get('dataset')

    if not original_dir or not os.path.isdir(original_dir):
        print(f'❌ 原图 XML 目录不可用: {original_dir}')
        print('   请用 --original-xml 指定，或提供 --base-changeset')
        return 1

    changeset, stats, problems = build_supplement_changeset(
        args.review_queue, decisions, original_dir, base_meta, args.decisions)
    mc.dump_json(changeset, args.output)

    print('=' * 72)
    print('人工决策 -> 补充变更集')
    print('=' * 72)
    print(f'  决策总数    : {len(decisions)}')
    print(f'  采纳        : {stats["accepted"]}')
    print(f'  驳回        : {stats["rejected"]}')
    for action in ('set_name', 'set_bbox', 'set_name_bbox', 'delete', 'add'):
        if stats.get(action):
            print(f'  {action:12s} {stats[action]}')
    if stats.get('supplement_merged_add') or stats.get('supplement_merged_modify') \
            or stats.get('supplement_dropped_add_after_retag'):
        print(f'  回灌去重    : 合并 add {stats.get("supplement_merged_add", 0)}，'
              f'合并改/删 {stats.get("supplement_merged_modify", 0)}，'
              f'retag 后丢掉的同名 add {stats.get("supplement_dropped_add_after_retag", 0)}')
    print(f'  受影响 XML  : {len(changeset["files"])}')
    if problems:
        print(f'\n⚠️  {len(problems)} 个问题:')
        for msg in problems[:20]:
            print(f'    - {msg}')
    print(f'\n已写出: {args.output}')
    print(f'下一步: python 2_应用变更集.py --changeset {args.output}   (先预演)')
    return 0


# ============================================================
#  审核服务器
# ============================================================

class ReviewHandler(http.server.BaseHTTPRequestHandler):
    """
    审核页面的本地 HTTP 服务。

    远程连 Linux 服务器做审核时，「浏览器导出 decisions.json 再传回服务器」这一步很别扭，
    所以决策通过 /api/decisions 直接落到服务器磁盘上，补充变更集也能在网页上直接生成。
    """

    review_dir: Path = Path('.')
    context: dict = {}
    # 原图目录是否可用，在服务启动时探测一次并缓存。
    # 该目录通常在 NFS 上，不能每次 ping 都去 stat。
    original_dir_ok: bool = False

    # 用 HTTP/1.1 开启长连接。审核页面要拉几十上百张对比图，
    # 走 SSH 端口转发时每张图都重新握手一次会非常慢。
    protocol_version = 'HTTP/1.1'

    CONTENT_TYPES = {
        '.html': 'text/html; charset=utf-8',
        '.json': 'application/json; charset=utf-8',
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png', '.bmp': 'image/bmp',
        '.css': 'text/css; charset=utf-8',
        '.js': 'application/javascript; charset=utf-8',
    }

    def log_message(self, fmt, *args):
        # 只打印非静态资源请求，避免上千张图片刷屏
        if '/api/' in str(args[0] if args else ''):
            print(f'[审核服务] {self.address_string()} - {fmt % args}')

    # ---------- 基础工具 ----------

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status: int):
        """HTTP/1.1 下没有响应体也必须给出 Content-Length，否则客户端会一直等"""
        self.send_response(status)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def _decisions_path(self) -> Path:
        return Path(self.context.get('decisions') or (self.review_dir / 'decisions.json'))

    def _load_decisions(self) -> dict:
        path = self._decisions_path()
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
            return raw.get('decisions', raw) or {}
        except Exception as exc:
            print(f'[审核服务] 读取决策文件失败: {exc}')
            return {}

    def _save_decisions(self, decisions: dict) -> None:
        path = self._decisions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'dataset': self.context.get('dataset'),
                   'saved_at': mc.now_iso(),
                   'decisions': decisions}
        # 原子写，避免边写边崩把决策弄丢
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(str(tmp), str(path))

    def _safe_path(self, url_path: str) -> Optional[Path]:
        """把 URL 映射到 review_dir 下的文件，杜绝路径穿越"""
        rel = urllib.parse.unquote(url_path).lstrip('/')
        if not rel:
            rel = 'index.html'
        target = (self.review_dir / rel).resolve()
        try:
            target.relative_to(self.review_dir.resolve())
        except ValueError:
            return None
        return target if target.is_file() else None

    # ---------- GET ----------

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == '/api/ping':
            self._send_json({
                'ok': True,
                'dataset': self.context.get('dataset'),
                'review_queue': self.context.get('review_queue'),
                'base_changeset': self.context.get('base_changeset'),
                'original_xml_dir': self.context.get('original_xml_dir') or '',
                'original_xml_dir_exists': self.original_dir_ok,
                'supplement_output': self.context.get('supplement_output'),
                'decisions_file': str(self._decisions_path()),
            })
            return

        if path == '/api/decisions':
            self._send_json({'ok': True, 'decisions': self._load_decisions()})
            return

        target = self._safe_path(path)
        if target is None:
            self._send_empty(404)
            return

        content_type = self.CONTENT_TYPES.get(target.suffix.lower(),
                                              'application/octet-stream')
        size = target.stat().st_size
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(size))
        if target.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
            # 对比图会随 render 重画，不能缓存一天，否则刷新页面仍是旧图
            self.send_header('Cache-Control', 'no-cache')
        else:
            self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        with open(target, 'rb') as f:
            shutil.copyfileobj(f, self.wfile, length=256 * 1024)

    # ---------- POST ----------

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            length = int(self.headers.get('Content-Length', 0))
            payload = json.loads(self.rfile.read(length)) if length else {}
        except Exception as exc:
            self._send_json({'ok': False, 'error': f'请求体解析失败: {exc}'}, status=400)
            return

        if path == '/api/decisions':
            decisions = payload.get('decisions', {})
            try:
                self._save_decisions(decisions)
                self._send_json({'ok': True, 'count': len(decisions)})
            except Exception as exc:
                self._send_json({'ok': False, 'error': str(exc)}, status=500)
            return

        if path == '/api/build-changeset':
            self._handle_build_changeset(payload)
            return

        self._send_empty(404)

    def _handle_build_changeset(self, payload: dict):
        try:
            decisions = payload.get('decisions')
            if decisions is None:
                decisions = self._load_decisions()
            else:
                self._save_decisions(decisions)

            review_queue = self.context.get('review_queue')
            original_dir = self.context.get('original_xml_dir')
            if not review_queue or not os.path.exists(review_queue):
                raise FileNotFoundError(f'待确认队列不存在: {review_queue}')
            if not original_dir or not os.path.isdir(original_dir):
                raise NotADirectoryError(f'原图 XML 目录不可用: {original_dir}')

            base_meta = {}
            base_changeset = self.context.get('base_changeset')
            if base_changeset and os.path.exists(base_changeset):
                base_meta = mc.load_json(base_changeset).get('meta', {})
            base_meta.setdefault('dataset', self.context.get('dataset'))

            changeset, stats, problems = build_supplement_changeset(
                review_queue, decisions, original_dir, base_meta,
                str(self._decisions_path()))

            output = self.context.get('supplement_output') or str(
                self.review_dir / 'supplement_changeset.json')
            mc.dump_json(changeset, output)

            print(f'[审核服务] 已生成补充变更集: {output} '
                  f'(采纳 {stats["accepted"]}，驳回 {stats["rejected"]}，'
                  f'{len(changeset["files"])} 个 XML'
                  f'{"" if not stats.get("supplement_merged_add") and not stats.get("supplement_merged_modify") and not stats.get("supplement_dropped_add_after_retag") else "，已合并重复决策"}'
                  f')')

            self._send_json({
                'ok': True,
                'output': output,
                'accepted': stats['accepted'],
                'rejected': stats['rejected'],
                'file_count': len(changeset['files']),
                'actions': {k: v for k, v in stats.items()
                            if k in ('set_name', 'set_bbox', 'set_name_bbox',
                                     'delete', 'add')},
                'deduped': {
                    'merged_add': stats.get('supplement_merged_add', 0),
                    'merged_modify': stats.get('supplement_merged_modify', 0),
                    'dropped_add_after_retag': stats.get(
                        'supplement_dropped_add_after_retag', 0),
                },
                'problems': problems[:20],
                'next_command': f'python 2_应用变更集.py --changeset {output}',
            })
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._send_json({'ok': False, 'error': str(exc)}, status=500)


class _ReuseThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve_review(review_dir: str, host: str = '127.0.0.1', port: int = 8770,
                 open_browser: bool = False) -> int:
    # nohup / 重定向到文件时 stdout 默认是全缓冲的，会导致看不到实际监听的端口号
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    review_path = Path(review_dir).resolve()
    index = review_path / 'index.html'
    if not index.exists():
        print(f'❌ 审核页面不存在: {index}')
        print('   请先运行 render 子命令生成审核页面')
        return 1

    meta_path = review_path / 'review_meta.json'
    context = mc.load_json(str(meta_path)) if meta_path.exists() else {}
    if not context:
        print(f'⚠️  没找到 {meta_path}，网页上的「生成补充变更集」按钮将不可用')

    original_dir = context.get('original_xml_dir') or ''
    handler = type('BoundReviewHandler', (ReviewHandler,), {
        'review_dir': review_path,
        'context': context,
        'original_dir_ok': bool(original_dir) and os.path.isdir(original_dir),
    })

    # 端口被占用就自动往后找
    server = None
    for candidate in range(port, port + 50):
        try:
            server = _ReuseThreadingHTTPServer((host, candidate), handler)
            port = candidate
            break
        except OSError:
            print(f'端口 {candidate} 被占用，尝试 {candidate + 1} ...')
    if server is None:
        print(f'❌ 从 {port} 起连续 50 个端口都被占用')
        return 1

    url = f'http://{"localhost" if host in ("127.0.0.1", "0.0.0.0") else host}:{port}/'
    print()
    print('=' * 72)
    print('审核服务已启动')
    print('=' * 72)
    print(f'  地址        : {url}')
    print(f'  审核目录    : {review_path}')
    print(f'  数据集      : {context.get("dataset")}')
    print(f'  决策保存到  : {context.get("decisions") or review_path / "decisions.json"}')
    print(f'  原图 XML    : {context.get("original_xml_dir")}')
    print('-' * 72)
    print('  远程连 Linux 时：Cursor / VSCode 会自动把这个端口转发到你的 Windows，')
    print(f'  直接在 Windows 浏览器里打开 {url} 即可。')
    print('  若没有自动转发，可手动在本机执行：')
    print(f'      ssh -L {port}:localhost:{port} <用户名>@<服务器地址>')
    print('  决策每次点击都会实时存到服务器磁盘上，关掉浏览器也不会丢。')
    print('  按 Ctrl+C 停止服务。')
    print('=' * 72)
    print()

    if open_browser:
        import webbrowser
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n审核服务已停止')
        server.shutdown()
    return 0


def cmd_serve(args) -> int:
    return serve_review(args.review_dir, host=args.host, port=args.port,
                        open_browser=args.open_browser)


# ============================================================
#  入口
# ============================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='待人工确认条目的可视化审核与决策回灌',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p_render = sub.add_parser('render', help='渲染对比图并生成审核页面')
    p_render.add_argument('--review-queue', required=True, help='review_queue.json 路径')
    p_render.add_argument('--output', required=True, help='输出目录')
    p_render.add_argument('--original-images', required=True, help='原图图片目录')
    p_render.add_argument('--dataset-dir', default=None,
                          help='子图数据集根目录（用于定位 images/）')
    p_render.add_argument('--sub-images', default=None,
                          help='子图图片目录（默认 <dataset-dir>/images）')
    p_render.add_argument('--dataset', default=None, help='数据集名称，用于页面标题')
    p_render.add_argument('--reason', action='append', default=None,
                          help='只渲染指定原因的条目，可重复指定')
    p_render.add_argument('--limit', type=int, default=0, help='最多渲染多少条（调试用）')
    p_render.add_argument('--margin', type=float, default=0.6,
                          help='原图局部视图向外留白的比例（默认 0.6）')
    p_render.add_argument('--max-height', type=int, default=900,
                          help='每张面板图的最大高度像素，越小传输越快（默认 900）')
    p_render.add_argument('--quality', type=int, default=82,
                          help='对比图 JPEG 质量（默认 82）')
    p_render.add_argument('--base-changeset', default=None,
                          help='对应的 changeset.json（默认取 review_queue.json 同目录下的）')
    p_render.add_argument('--original-xml', default=None,
                          help='原图 XML 目录（默认从 base-changeset 的 meta 里取）')
    p_render.add_argument('--serve', action='store_true',
                          help='渲染完直接启动审核服务（远程审核推荐）')
    p_render.add_argument('--host', default='127.0.0.1',
                          help='服务监听地址；需要局域网直连时用 0.0.0.0（默认 127.0.0.1）')
    p_render.add_argument('--port', type=int, default=8770, help='服务端口（默认 8770）')
    p_render.add_argument('--open-browser', action='store_true',
                          help='启动后尝试在本机打开浏览器（远程服务器上通常没用）')
    p_render.set_defaults(func=cmd_render)

    p_serve = sub.add_parser('serve', help='启动审核服务，浏览器远程即可操作')
    p_serve.add_argument('--review-dir', required=True,
                         help='render 生成的审核目录（含 index.html / review_meta.json）')
    p_serve.add_argument('--host', default='127.0.0.1',
                         help='监听地址；需要局域网直连时用 0.0.0.0（默认 127.0.0.1）')
    p_serve.add_argument('--port', type=int, default=8770, help='端口（默认 8770）')
    p_serve.add_argument('--open-browser', action='store_true',
                         help='启动后尝试在本机打开浏览器（远程服务器上通常没用）')
    p_serve.set_defaults(func=cmd_serve)

    p_build = sub.add_parser('build-changeset', help='把人工决策组装成补充变更集')
    p_build.add_argument('--review-queue', required=True, help='review_queue.json 路径')
    p_build.add_argument('--decisions', required=True, help='审核页面导出的 decisions.json')
    p_build.add_argument('--output', required=True, help='输出的补充变更集 json 路径')
    p_build.add_argument('--base-changeset', default=None,
                         help='原始 changeset.json，用于继承 meta 信息')
    p_build.add_argument('--original-xml', default=None, help='原图 XML 目录')
    p_build.set_defaults(func=cmd_build_changeset)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
