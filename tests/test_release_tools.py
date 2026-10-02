"""Offline regression tests for release tooling, not model-quality experiments."""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import pytest
from gen_harness.datasets.evolution import build_partiprompts_evolution_split, _load_prompt_rows
from scripts.prepare_evolution import ROOT, prepare, validate_splits
from scripts.snapshot_harness import snapshot, verify


def test_jsonl_prompt_source_supports_multiple_objects():
    rows = _load_prompt_rows('{"prompt":"a fox"}\n{"prompt":"a panda"}\n')
    assert [r['prompt'] for r in rows] == ['a fox', 'a panda']


def test_json_source_accepts_one_prompt_object():
    assert _load_prompt_rows('{"prompt":"a fox"}')[0]['prompt'] == 'a fox'


def test_json_source_rejects_non_object_rows():
    with pytest.raises(ValueError, match='prompt objects'):
        _load_prompt_rows('["a fox"]')


@pytest.mark.parametrize('value', [-1, 1.2, True])
def test_evolution_builder_rejects_invalid_counts(tmp_path, value):
    with pytest.raises(ValueError, match='non-negative integer'):
        build_partiprompts_evolution_split('does-not-need-to-exist.tsv', tmp_path/'out', target_count=value)


def test_evolution_builder_rejects_empty_total(tmp_path):
    with pytest.raises(ValueError, match='at least one'):
        build_partiprompts_evolution_split('unused.tsv', tmp_path/'out', target_count=0, heldout_count=0, preservation_count=0)


def test_jsonl_builder_preserves_original_output_path_contract(tmp_path):
    source = tmp_path/'source.jsonl'
    source.write_text('\n'.join(json.dumps({'prompt': p}) for p in ['a fox', 'a panda', 'a cat'])+'\n')
    report = build_partiprompts_evolution_split(str(source), tmp_path/'out', target_count=1, heldout_count=1, preservation_count=1)
    assert (tmp_path/'out/partiprompts_p2_target500.jsonl').is_file()
    assert report['target_count'] == 1


def test_release_inputs_are_valid_and_pairwise_disjoint():
    rows, report = validate_splits()
    assert len(rows) == 700
    assert report['counts'] == {'target': 500, 'heldout': 100, 'preservation': 100}
    assert len(report['benchmark_checks']) == 1
    assert report['benchmark_checks'][0]['normalized_exact_overlaps'] == 0


def test_preparation_is_idempotent(tmp_path):
    path = tmp_path/'all.jsonl'
    first = prepare(path)
    second = prepare(path)
    assert first == second
    assert len(path.read_text().splitlines()) == 700
    assert path.with_suffix('.manifest.json').exists()


def test_preparation_refuses_destructive_overwrite(tmp_path):
    path = tmp_path/'all.jsonl';path.write_text('keep this\n')
    with pytest.raises(FileExistsError):
        prepare(path)
    assert path.read_text() == 'keep this\n'


def test_data_integrity_rejects_prompt_tampering(tmp_path):
    root = tmp_path/'repo'
    shutil.copytree(ROOT/'data', root/'data')
    path = root/'data/evolution/partiprompts_p2_target500.jsonl'
    text = path.read_text();row = json.loads(text.splitlines()[0]);row['prompt'] += ' changed'
    lines=text.splitlines();lines[0]=json.dumps(row);path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError, match='checksum'):
        validate_splits(root)


def test_snapshot_roundtrip_preserves_source(tmp_path):
    source = ROOT/'examples/visual_harness'
    output = tmp_path/'snapshot'
    report = snapshot(source, output)
    assert report['status'] == 'locally_created_snapshot'
    assert verify(output)['passed']
    assert (output/'policy/visual_contract.json').read_bytes() == (source/'policy/visual_contract.json').read_bytes()


def test_snapshot_refuses_existing_destination(tmp_path):
    output = tmp_path/'exists';output.mkdir()
    with pytest.raises(FileExistsError):
        snapshot(ROOT/'examples/visual_harness', output)


def test_snapshot_verification_detects_mutation(tmp_path):
    output = tmp_path/'snapshot';snapshot(ROOT/'examples/visual_harness', output)
    p=output/'policy/visual_contract.json';p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='changed'):
        verify(output)


def test_snapshot_rejects_symlinks(tmp_path):
    source=tmp_path/'source';shutil.copytree(ROOT/'examples/visual_harness',source)
    (source/'policy/linked.json').symlink_to(source/'policy/visual_contract.json')
    with pytest.raises(ValueError, match='symlinks'):
        snapshot(source,tmp_path/'output')
