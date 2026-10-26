"""Persistent edits must stay local and temporary edits must roll back exactly."""
import pytest

from gen_harness.repository import HarnessRepository
from gen_harness.schema import PatchManifest


@pytest.mark.parametrize("artifact", ["", ".", "..", "../outside.json", "/tmp/outside.json", "nested/file.json"])
def test_repository_rejects_artifact_path_escape(tmp_path, artifact):
    repo = HarnessRepository(tmp_path / "harness")
    with pytest.raises(ValueError, match="filename"):
        repo.save_artifact("policy", artifact, {"changed": True})
    assert not (tmp_path / "harness").exists()


def test_repository_rejects_artifact_symlink(tmp_path):
    repo = HarnessRepository(tmp_path / "harness")
    repo.component_dir("policy").mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text('{}\n', encoding="utf-8")
    (repo.component_dir("policy") / "contract.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlinks"):
        repo.save_artifact("policy", "contract.json", {"changed": True})
    assert outside.read_text(encoding="utf-8") == '{}\n'


def test_temporary_memory_patch_removes_new_file_on_exception(tmp_path):
    repo = HarnessRepository(tmp_path / "harness")
    patch = PatchManifest(
        patch_id="temporary", target_component="memory", changed_artifact="prompt_guidelines.jsonl",
        changed_fields=["prompt_guidelines"], operation="append_jsonl_unique",
        payload={"row": {"id": "new", "guideline": "Preserve explicit counts."}, "unique_key": "id"},
        supporting_experience=[], predicted_improvement="count fidelity", possible_regression="unknown",
        target_validation_set="target", preservation_set="preservation", held_out_set="heldout",
        rollback_boundary="memory/prompt_guidelines.jsonl",
    )
    path = repo.artifact_path("memory", patch.changed_artifact)
    with pytest.raises(RuntimeError, match="generation failed"):
        with repo.temporary_patch(patch):
            assert path.is_file()
            raise RuntimeError("generation failed")
    assert not path.exists()
