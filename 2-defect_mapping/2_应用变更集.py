#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_应用变更集.py — 把 1_生成变更集.py 产出的 changeset.json 真正写入原图总库

安全设计（原图库是总库，务必逐条遵守）：
  1. 默认 dry-run。不加 --apply 只做预演，一个字节都不会写。
  2. 写入前先做完整的 preflight：逐个文件校验 md5 与生成变更集时一致、
     逐条变更校验目标框在原图中恰好命中一次。任何一项不通过就整体中止，
     绝不出现「改了一半」的局面。
  3. 只备份将被修改的那些 XML（不是整库复制），并写 manifest.json，
     配套 3_回滚.py 可一键还原。
  4. 每个文件用「临时文件 + os.replace」原子写入。
  5. 成功后把每一条变更追加到 mapping_ledger.jsonl，供后续数据集做冲突检测与幂等。

用法：
  # 预演（推荐先跑）
  python 2_应用变更集.py --changeset /raid/datasets_defect_2026/datasets_val/全量_正样本/gd_data/映射原图过程/changeset.json

  # 真正写入
  python 2_应用变更集.py --changeset /raid/datasets_defect_2026/datasets_val/全量_正样本/gd_data/映射原图过程/changeset.json --apply
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mapping_core as mc

ACT_SET_NAME = 'set_name'
ACT_SET_BBOX = 'set_bbox'
ACT_SET_NAME_BBOX = 'set_name_bbox'
ACT_DELETE = 'delete'
ACT_ADD = 'add'
MODIFY_ACTIONS = (ACT_SET_NAME, ACT_SET_BBOX, ACT_SET_NAME_BBOX)


class Problem:
    __slots__ = ('xml_name', 'kind', 'message', 'change')

    def __init__(self, xml_name: str, kind: str, message: str, change: Optional[dict] = None):
        self.xml_name = xml_name
        self.kind = kind
        self.message = message
        self.change = change

    def __str__(self):
        return f'[{self.kind}] {self.xml_name}: {self.message}'


def resolve_after(change: dict) -> Tuple[Optional[str], Optional[List[int]]]:
    """把一条变更折算成「这个框最终的类名和坐标」，用于写台账"""
    if change['action'] == ACT_DELETE:
        return None, None
    if change['action'] == ACT_ADD:
        return change['new_name'], list(change['new_bbox'])
    name = change.get('new_name', change['target_name'])
    bbox = list(change.get('new_bbox', change['target_bbox']))
    return name, bbox


def apply_changes_to_doc(doc: mc.VocXml, changes: List[dict],
                         xml_name: str, problems: List[Problem]) -> int:
    """
    把一组变更施加到一份已解析的 XML 上（只改内存，不落盘）。
    返回成功施加的条数；任何定位失败都记入 problems。

    先处理「改/删」再处理「增」，避免新增的框干扰对已有框的定位。
    """
    applied = 0
    objects = doc.objects()

    modify_and_delete = [c for c in changes if c['action'] != ACT_ADD]
    additions = [c for c in changes if c['action'] == ACT_ADD]

    # 先把所有目标一次性定位好，避免边改边找导致互相干扰
    resolved: List[Tuple[dict, mc.XmlObject]] = []
    claimed = set()
    for change in modify_and_delete:
        target = (change['target_name'], tuple(change['target_bbox']))
        hits = [o for o in objects if o.key() == target and id(o) not in claimed]
        if len(hits) == 0:
            problems.append(Problem(xml_name, 'target_not_found',
                                    f'找不到目标框 {target}', change))
            continue
        if len([o for o in objects if o.key() == target]) > 1:
            problems.append(Problem(xml_name, 'target_ambiguous',
                                    f'原图中存在多个相同的框 {target}', change))
            continue
        claimed.add(id(hits[0]))
        resolved.append((change, hits[0]))

    for change, obj in resolved:
        action = change['action']
        if action == ACT_DELETE:
            doc.remove(obj)
        elif action == ACT_SET_NAME:
            doc.set_name(obj, change['new_name'])
        elif action == ACT_SET_BBOX:
            doc.set_bbox(obj, tuple(change['new_bbox']))
        elif action == ACT_SET_NAME_BBOX:
            doc.set_name(obj, change['new_name'])
            doc.set_bbox(obj, tuple(change['new_bbox']))
        else:
            problems.append(Problem(xml_name, 'unknown_action',
                                    f'未知的变更类型 {action}', change))
            continue
        applied += 1

    for change in additions:
        bbox = tuple(int(v) for v in change['new_bbox'])
        if not mc.is_valid_bbox(bbox):
            problems.append(Problem(xml_name, 'invalid_bbox',
                                    f'新增框坐标非法 {bbox}', change))
            continue
        doc.add_object(change['new_name'], bbox)
        applied += 1

    return applied


def preflight(changeset: dict, original_dir: str, check_md5: bool
              ) -> Tuple[List[Problem], Counter, Dict[str, int]]:
    """
    完整预演：解析每个文件、校验 md5、施加变更（只在内存里），统计结果。
    不写任何文件。
    """
    problems: List[Problem] = []
    counter: Counter = Counter()
    per_file: Dict[str, int] = {}

    files = changeset['files']
    total = len(files)
    for i, (xml_name, block) in enumerate(sorted(files.items()), 1):
        if i % 2000 == 0:
            print(f'  预演进度 {i}/{total}')

        path = os.path.join(original_dir, xml_name)
        if not os.path.exists(path):
            problems.append(Problem(xml_name, 'file_missing', f'原图 XML 不存在: {path}'))
            continue

        if check_md5 and block.get('md5'):
            actual = mc.md5_of(path)
            if actual != block['md5']:
                problems.append(Problem(
                    xml_name, 'md5_mismatch',
                    f'文件自生成变更集后已被改动（期望 {block["md5"][:8]}，实际 {actual[:8]}）'))
                continue

        try:
            doc = mc.VocXml(path)
        except Exception as exc:
            problems.append(Problem(xml_name, 'parse_error', f'解析失败: {exc}'))
            continue

        before = len(doc.objects())
        applied = apply_changes_to_doc(doc, block['changes'], xml_name, problems)
        after = len(doc.objects())

        per_file[xml_name] = applied
        for change in block['changes']:
            counter[change['action']] += 1
        counter['objects_before'] += before
        counter['objects_after'] += after
        counter['applied'] += applied

    return problems, counter, per_file


def do_backup(changeset: dict, original_dir: str, backup_dir: str,
              extra_meta: Optional[dict] = None) -> str:
    """只备份将被修改的 XML，并写 manifest.json 供回滚使用"""
    backup_files_dir = os.path.join(backup_dir, 'files')
    os.makedirs(backup_files_dir, exist_ok=True)

    entries = []
    names = sorted(changeset['files'])
    for i, xml_name in enumerate(names, 1):
        if i % 2000 == 0:
            print(f'  备份进度 {i}/{len(names)}')
        src = os.path.join(original_dir, xml_name)
        dst = os.path.join(backup_files_dir, xml_name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        entries.append({'xml': xml_name, 'md5_before': mc.md5_of(src)})

    manifest = {
        'created_at': mc.now_iso(),
        'original_xml_dir': os.path.abspath(original_dir),
        'dataset': changeset['meta'].get('dataset'),
        'changeset_created_at': changeset['meta'].get('created_at'),
        'file_count': len(entries),
        'files': entries,
    }
    if extra_meta:
        manifest.update(extra_meta)
    mc.dump_json(manifest, os.path.join(backup_dir, 'manifest.json'))
    return os.path.join(backup_dir, 'manifest.json')


def build_ledger_records(changeset: dict, changeset_path: str,
                         applied_at: Optional[str] = None) -> List[dict]:
    meta = changeset['meta']
    stamp = applied_at or mc.now_iso()
    records = []
    for xml_name, block in sorted(changeset['files'].items()):
        for change in block['changes']:
            after_name, after_bbox = resolve_after(change)
            records.append({
                'applied_at': stamp,
                'dataset': meta.get('dataset'),
                'component_type': meta.get('component_type'),
                'changeset': os.path.abspath(changeset_path),
                'original_xml': xml_name,
                'action': change['action'],
                'before_name': change.get('target_name'),
                'before_bbox': change.get('target_bbox'),
                'after_name': after_name,
                'after_bbox': after_bbox,
                'sources': change.get('sources', []),
            })
    return records


def print_problems(problems: List[Problem], limit: int = 30) -> None:
    grouped: Dict[str, List[Problem]] = {}
    for p in problems:
        grouped.setdefault(p.kind, []).append(p)
    for kind, items in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        print(f'  {kind}: {len(items)} 条')
        for p in items[:limit]:
            print(f'    - {p.xml_name}: {p.message}')
        if len(items) > limit:
            print(f'    ... 另有 {len(items) - limit} 条')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='把变更集写入原图总库（默认只预演，须显式 --apply 才写入）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预演，不写任何文件
  python 2_应用变更集.py --changeset /raid/datasets_defect_2026/datasets_val/全量_正样本/gd_data/映射原图过程/changeset.json

  # 确认无误后真正写入
  python 2_应用变更集.py --changeset /raid/datasets_defect_2026/datasets_val/全量_正样本/gd_data/映射原图过程/changeset.json --apply
""")
    parser.add_argument('--changeset', required=True, help='changeset.json 路径')
    parser.add_argument('--apply', action='store_true',
                        help='真正写入原图 XML；不加则只预演')
    parser.add_argument('--original-xml', default=None,
                        help='覆盖变更集里记录的原图 XML 目录（原图库换了位置时使用）')
    parser.add_argument('--backup-dir', default=None,
                        help='备份目录（默认 <changeset 所在目录>/backup_<时间戳>）')
    parser.add_argument('--ledger', default=None,
                        help='变更台账路径（默认取变更集里记录的位置）')
    parser.add_argument('--no-md5-check', action='store_true',
                        help='跳过 md5 校验（不推荐；只在确知原图被无关改动过时使用）')
    parser.add_argument('--skip-invalid', action='store_true',
                        help='遇到有问题的文件时跳过该文件而不是整体中止（不推荐）')
    parser.add_argument('--yes', action='store_true', help='跳过交互确认')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.changeset):
        print(f'❌ 变更集不存在: {args.changeset}')
        return 1

    changeset = mc.load_json(args.changeset)
    meta = changeset.get('meta', {})
    original_dir = args.original_xml or meta.get('original_xml_dir')
    if not original_dir or not os.path.isdir(original_dir):
        print(f'❌ 原图 XML 目录不可用: {original_dir}')
        print('   若原图库换了位置，请用 --original-xml 指定新路径')
        return 1

    ledger_path = args.ledger or meta.get('ledger') or mc.ledger_path_for(original_dir)
    total_changes = sum(len(b['changes']) for b in changeset['files'].values())

    print('=' * 72)
    print('应用变更集' + ('（真正写入）' if args.apply else '（预演，不写任何文件）'))
    print('=' * 72)
    print(f'  变更集      : {args.changeset}')
    print(f'  数据集      : {meta.get("dataset")}  (部件类型 {meta.get("component_type")})')
    print(f'  生成于      : {meta.get("created_at")}')
    print(f'  原图 XML    : {original_dir}')
    print(f'  台账        : {ledger_path}')
    print(f'  待改文件数  : {len(changeset["files"])}')
    print(f'  变更总条数  : {total_changes}')
    print('=' * 72)

    if not changeset['files']:
        print('\n变更集为空，无事可做。')
        return 0

    # ---------- preflight ----------
    print('\n[1/4] 预演全部变更（含 md5 校验与目标定位校验）...')
    problems, counter, per_file = preflight(changeset, original_dir, not args.no_md5_check)

    print('\n预演结果:')
    for action in (ACT_SET_NAME, ACT_SET_BBOX, ACT_SET_NAME_BBOX, ACT_DELETE, ACT_ADD):
        print(f'  {action:18s} {counter.get(action, 0)}')
    print(f'  {"成功施加":16s} {counter.get("applied", 0)} / {total_changes}')
    print(f'  原图框总数 {counter.get("objects_before", 0)} -> {counter.get("objects_after", 0)}'
          f'  (净变化 {counter.get("objects_after", 0) - counter.get("objects_before", 0):+d})')

    if problems:
        print(f'\n⚠️  预演发现 {len(problems)} 个问题:')
        print_problems(problems)
        if not args.skip_invalid:
            print('\n❌ 已中止，未写入任何文件。')
            print('   请重新运行 1_生成变更集.py 生成最新的变更集，')
            print('   或在确知风险的前提下加 --skip-invalid 跳过这些文件。')
            return 2
        print('\n⚠️  --skip-invalid 已开启，将跳过有问题的部分继续执行。')
    else:
        print('\n✅ 预演全部通过，没有发现任何问题。')

    if not args.apply:
        print('\n这是预演模式，没有写入任何文件。')
        print('确认无误后加 --apply 参数再执行一次即可真正写入。')
        return 0

    # ---------- 确认 ----------
    if not args.yes:
        print('\n' + '!' * 72)
        print('即将修改原图总库的 XML 文件。修改前会先备份，可用 3_回滚.py 还原。')
        print(f'目标目录: {original_dir}')
        print('!' * 72)
        answer = input('确认执行？请输入 yes: ').strip().lower()
        if answer not in ('yes', 'y'):
            print('已取消，未写入任何文件。')
            return 0

    # ---------- 备份 ----------
    # applied_at 与随后写入的台账行共用同一时间戳，回滚才能精确撤回这次 apply
    applied_at = mc.now_iso()
    backup_dir = args.backup_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.changeset)), f'backup_{mc.now_stamp()}')
    print(f'\n[2/4] 备份将被修改的 {len(changeset["files"])} 个 XML 到 {backup_dir} ...')
    manifest_path = do_backup(changeset, original_dir, backup_dir, extra_meta={
        'changeset': os.path.abspath(args.changeset),
        'ledger': os.path.abspath(ledger_path),
        'applied_at': applied_at,
        'ledger_record_count': total_changes,
    })
    print(f'  ✅ 备份完成，清单: {manifest_path}')

    # ---------- 写入 ----------
    print('\n[3/4] 写入原图 XML ...')
    write_problems: List[Problem] = []
    written = 0
    skipped = 0
    names = sorted(changeset['files'])
    for i, xml_name in enumerate(names, 1):
        if i % 2000 == 0:
            print(f'  写入进度 {i}/{len(names)}')
        block = changeset['files'][xml_name]
        path = os.path.join(original_dir, xml_name)
        if not os.path.exists(path):
            skipped += 1
            continue
        try:
            doc = mc.VocXml(path)
        except Exception as exc:
            write_problems.append(Problem(xml_name, 'parse_error', str(exc)))
            skipped += 1
            continue

        before = len(write_problems)
        apply_changes_to_doc(doc, block['changes'], xml_name, write_problems)
        if len(write_problems) > before and not args.skip_invalid:
            skipped += 1
            continue
        if doc.dirty:
            doc.save()
            written += 1
        else:
            skipped += 1

    print(f'  ✅ 已写入 {written} 个文件，跳过 {skipped} 个')
    if write_problems:
        print(f'  ⚠️  写入阶段出现 {len(write_problems)} 个问题:')
        print_problems(write_problems, limit=10)

    # ---------- 台账 ----------
    print('\n[4/4] 追加变更台账 ...')
    records = build_ledger_records(changeset, args.changeset, applied_at=applied_at)
    count = mc.append_ledger(ledger_path, records)
    print(f'  ✅ 已追加 {count} 条记录到 {ledger_path}')

    print('\n' + '=' * 72)
    print('完成。')
    print(f'  备份目录: {backup_dir}')
    print(f'  如需还原: python 3_回滚.py --manifest {manifest_path}')
    print('  （回滚会同时撤回这次写入对应的台账行）')
    print('=' * 72)
    return 0


if __name__ == '__main__':
    sys.exit(main())
