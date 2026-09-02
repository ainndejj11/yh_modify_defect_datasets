#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3_回滚.py — 用 2_应用变更集.py 生成的备份把原图 XML 还原回去，并撤回对应台账

manifest.json 里记录了每个被修改文件在修改前的 md5，回滚时会逐个核对：
  - 现场文件的 md5 == md5_before  -> 本来就没改过或已回滚过，跳过
  - 备份文件的 md5 != md5_before  -> 备份自身已损坏，中止
还原同样采用「临时文件 + os.replace」原子写入。

台账：只撤回「这次备份对应那一次 --apply」写入的行，其它数据集、更早的写入都留下。
若这些 XML 在本次备份之后又被改过（例如补充变更集、下一个数据集），默认中止，
必须先回滚更晚的那次；加 --force 会连同这些更晚的台账一并撤回（因为整文件还原已经把它们盖掉了）。

用法：
  # 预演
  python 3_回滚.py --manifest ./输出/gd_data/backup_20260813_143000/manifest.json

  # 真正还原
  python 3_回滚.py --manifest ./输出/gd_data/backup_20260813_143000/manifest.json --apply
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mapping_core as mc


def atomic_copy(src: str, dst: str) -> None:
    """原子还原：先复制到目标目录下的临时文件，再 os.replace"""
    dst_dir = os.path.dirname(os.path.abspath(dst)) or '.'
    os.makedirs(dst_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.rollback_tmp_', suffix='.xml', dir=dst_dir)
    os.close(fd)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='按备份清单把原图 XML 还原回修改前的状态，并撤回对应台账（默认只预演）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 3_回滚.py --manifest ./输出/gd_data/backup_20260813_143000/manifest.json
  python 3_回滚.py --manifest ./输出/gd_data/backup_20260813_143000/manifest.json --apply
""")
    parser.add_argument('--manifest', required=True, help='备份目录下的 manifest.json')
    parser.add_argument('--apply', action='store_true', help='真正还原；不加则只预演')
    parser.add_argument('--original-xml', default=None,
                        help='覆盖 manifest 里记录的原图 XML 目录')
    parser.add_argument('--ledger', default=None, help='覆盖台账路径')
    parser.add_argument('--skip-ledger', action='store_true',
                        help='只还原 XML，不改台账（不推荐；下次生成变更集会把这批框当成已应用）')
    parser.add_argument('--force', action='store_true',
                        help='现场文件与备份对不上、或存在更晚的台账写入时仍强制执行')
    parser.add_argument('--yes', action='store_true', help='跳过交互确认')
    return parser.parse_args(argv)


def resolve_ledger_path(args, manifest: dict, original_dir: str) -> str:
    if args.ledger:
        return os.path.abspath(args.ledger)
    default = mc.ledger_path_for(original_dir)
    recorded = manifest.get('ledger')
    if os.path.exists(default):
        return os.path.abspath(default)
    if recorded and os.path.exists(recorded):
        return os.path.abspath(recorded)
    return os.path.abspath(default)


def print_record_preview(records: List[dict], indices: Sequence[int],
                         limit: int = 8) -> None:
    for i in indices[:limit]:
        rec = records[i]
        xml_name = rec.get('original_xml')
        action = rec.get('action')
        dataset = rec.get('dataset')
        when = rec.get('applied_at')
        print(f'    - [{dataset}] {action} {xml_name} @ {when}')
    extra = len(indices) - limit
    if extra > 0:
        print(f'    ... 另有 {extra} 条')


def plan_ledger(args, manifest: dict, original_dir: str
                ) -> Tuple[Optional[dict], Optional[str]]:
    """
    算出要撤回哪些台账行。返回 (plan, error_message)。
    error_message 非空表示默认必须中止；--force / --skip-ledger 由调用方处理。
    """
    ledger_path = resolve_ledger_path(args, manifest, original_dir)
    records = mc.read_ledger(ledger_path)
    changeset = os.path.abspath(manifest['changeset']) if manifest.get('changeset') else None

    plan = {
        'ledger_path': ledger_path,
        'records': records,
        'changeset': changeset,
        'this_indices': [],
        'later_indices': [],
        'retract_indices': [],
        'keep_indices': list(range(len(records))),
        'match_method': '',
        'already_clean': False,
    }

    if args.skip_ledger:
        plan['match_method'] = 'skip'
        return plan, None

    if not manifest.get('applied_at') or not manifest.get('changeset'):
        return plan, (
            '这份备份的 manifest 缺少 changeset / applied_at，'
            '不是当前版本 2_应用变更集.py --apply 生成的，无法安全撤回台账。'
        )

    this_indices, method = mc.match_ledger_indices_for_rollback(records, manifest)
    later_indices = mc.later_ledger_indices(records, this_indices, manifest)
    retract = set(this_indices)
    if args.force:
        retract |= set(later_indices)
    keep_indices = [i for i in range(len(records)) if i not in retract]

    plan['this_indices'] = this_indices
    plan['later_indices'] = later_indices
    plan['retract_indices'] = sorted(retract)
    plan['keep_indices'] = keep_indices
    plan['match_method'] = method
    plan['already_clean'] = not this_indices and not later_indices

    expected = manifest.get('ledger_record_count')
    if expected is not None and this_indices and len(this_indices) != expected:
        print(f'  ⚠️  匹配到 {len(this_indices)} 条台账，与备份记录的 {expected} 条不一致'
              f'（可能台账已被部分改过，将按实际匹配到的行撤回）')

    if later_indices and not args.force:
        return plan, (
            f'这些 XML 在本次备份之后又有 {len(later_indices)} 条台账写入。'
            '整文件还原会把后来的改动一起盖掉。请先回滚更晚的那次备份；'
            '确认要连同后续写入一起作废时再加 --force。'
        )
    return plan, None


def apply_ledger_retract(plan: dict, backup_dir: str) -> int:
    """把撤回前的台账快照放到本次备份目录，再原子写回保留行。"""
    records: List[dict] = plan['records']
    retract = set(plan['retract_indices'])
    if not retract:
        return 0

    ledger_path = plan['ledger_path']
    if os.path.exists(ledger_path):
        snap = os.path.join(backup_dir, f'ledger_before_rollback_{mc.now_stamp()}.jsonl')
        shutil.copy2(ledger_path, snap)
        print(f'  台账快照    : {snap}')

    remaining = [records[i] for i in range(len(records)) if i not in retract]
    mc.write_ledger(ledger_path, remaining)
    return len(retract)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.manifest):
        print(f'❌ 备份清单不存在: {args.manifest}')
        return 1

    manifest = mc.load_json(args.manifest)
    backup_dir = os.path.dirname(os.path.abspath(args.manifest))
    backup_files_dir = os.path.join(backup_dir, 'files')
    original_dir = args.original_xml or manifest.get('original_xml_dir')

    if not os.path.isdir(backup_files_dir):
        print(f'❌ 备份文件目录不存在: {backup_files_dir}')
        return 1
    if not original_dir or not os.path.isdir(original_dir):
        print(f'❌ 原图 XML 目录不可用: {original_dir}')
        return 1

    if not args.skip_ledger and (not manifest.get('applied_at') or not manifest.get('changeset')):
        print('❌ 这份备份的 manifest 缺少 changeset / applied_at，')
        print('   不是当前版本 2_应用变更集.py --apply 生成的，无法安全撤回台账。')
        return 1

    entries = manifest.get('files', [])
    print('=' * 72)
    print('回滚' + ('（真正还原）' if args.apply else '（预演，不写任何文件）'))
    print('=' * 72)
    print(f'  备份清单    : {args.manifest}')
    print(f'  备份产生于  : {manifest.get("created_at")}')
    print(f'  对应数据集  : {manifest.get("dataset")}')
    print(f'  原图 XML    : {original_dir}')
    print(f'  备份文件数  : {len(entries)}')
    print('=' * 72)

    stats: Counter = Counter()
    problems: List[str] = []
    todo: List[tuple] = []

    print('\n[1/3] 校验备份完整性 ...')
    for i, item in enumerate(entries, 1):
        if i % 2000 == 0:
            print(f'  校验进度 {i}/{len(entries)}')
        xml_name = item['xml']
        md5_before = item.get('md5_before')
        backup_path = os.path.join(backup_files_dir, xml_name)
        target_path = os.path.join(original_dir, xml_name)

        if not os.path.exists(backup_path):
            problems.append(f'备份文件缺失: {xml_name}')
            stats['backup_missing'] += 1
            continue

        if md5_before and mc.md5_of(backup_path) != md5_before:
            problems.append(f'备份文件已损坏（md5 与清单不符）: {xml_name}')
            stats['backup_corrupted'] += 1
            continue

        if not os.path.exists(target_path):
            todo.append((backup_path, target_path, xml_name))
            stats['target_missing_will_restore'] += 1
            continue

        current = mc.md5_of(target_path)
        if md5_before and current == md5_before:
            stats['already_original'] += 1
            continue

        todo.append((backup_path, target_path, xml_name))
        stats['will_restore'] += 1

    print(f'  需要还原      : {stats["will_restore"]}')
    print(f'  已是原始状态  : {stats["already_original"]} (跳过)')
    if stats['target_missing_will_restore']:
        print(f'  现场文件缺失  : {stats["target_missing_will_restore"]} (将从备份补回)')
    if problems:
        print(f'\n⚠️  发现 {len(problems)} 个问题:')
        for msg in problems[:20]:
            print(f'    - {msg}')
        if len(problems) > 20:
            print(f'    ... 另有 {len(problems) - 20} 条')
        if not args.force:
            print('\n❌ 已中止，未还原任何文件。确认无误可加 --force 强制执行。')
            return 2

    print('\n[2/3] 核对台账 ...')
    plan, ledger_error = plan_ledger(args, manifest, original_dir)
    assert plan is not None
    print(f'  台账        : {plan["ledger_path"]}')
    if args.skip_ledger:
        print('  ⚠️  --skip-ledger：只还原 XML，不改台账')
    else:
        print(f'  匹配方式    : {plan["match_method"]}')
        if plan.get('changeset'):
            print(f'  对应变更集  : {plan["changeset"]}')
        print(f'  将撤回本次  : {len(plan["this_indices"])} 条')
        if plan['this_indices']:
            print_record_preview(plan['records'], plan['this_indices'])
        print(f'  更晚写入    : {len(plan["later_indices"])} 条'
              + ('（--force 将一并撤回）' if plan['later_indices'] and args.force else ''))
        if plan['later_indices']:
            print_record_preview(plan['records'], plan['later_indices'])
        print(f'  将保留      : {len(plan["keep_indices"])} 条（其他数据集或更早写入）')
        if plan['already_clean'] and not plan['this_indices']:
            print('  本次写入在台账中已不存在，无需再撤')

    if ledger_error:
        print(f'\n❌ {ledger_error}')
        return 2

    if not todo and not plan['retract_indices']:
        print('\n没有需要还原的文件，台账也无需撤回。')
        return 0

    if not args.apply:
        print('\n这是预演模式，没有写入任何文件。')
        print('确认无误后加 --apply 参数再执行一次即可真正还原。')
        return 0

    if not args.yes:
        print('\n' + '!' * 72)
        print(f'即将用备份覆盖 {len(todo)} 个原图 XML 文件。')
        print(f'并将撤回台账 {len(plan["retract_indices"])} 条。')
        print(f'目标目录: {original_dir}')
        print('!' * 72)
        answer = input('确认执行？请输入 yes: ').strip().lower()
        if answer not in ('yes', 'y'):
            print('已取消，未还原任何文件。')
            return 0

    restored = failed = 0
    if todo:
        print('\n[3/3] 还原 XML ...')
        for i, (backup_path, target_path, xml_name) in enumerate(todo, 1):
            if i % 2000 == 0:
                print(f'  还原进度 {i}/{len(todo)}')
            try:
                atomic_copy(backup_path, target_path)
                restored += 1
            except Exception as exc:
                failed += 1
                print(f'  ❌ 还原失败 {xml_name}: {exc}')
        print(f'  ✅ 已还原 {restored} 个文件' + (f'，失败 {failed} 个' if failed else ''))
    else:
        print('\n[3/3] XML 已是备份前状态，跳过文件还原')

    if args.skip_ledger:
        print('  未改台账（--skip-ledger）')
        return 0 if not failed else 1

    retracted = apply_ledger_retract(plan, backup_dir)
    print(f'  ✅ 已撤回 {retracted} 条台账，保留 {len(plan["keep_indices"])} 条')
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
