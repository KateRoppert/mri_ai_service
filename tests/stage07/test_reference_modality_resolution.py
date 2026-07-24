"""
Bug: stage 07 (inverse_transform) got its reference modality from a
hardcoded --reference-modality value in pipeline_config.yaml, completely
independent from stage 05's own preprocessing_config.yaml, which actually
decides what reference modality was registered to the atlas.

Both files are shared across lesion types (one pipeline_config.yaml
template for every run, per backend/config.py's pipeline_config_template),
so a per-lesion-type reference_modality (t1c for glioblastoma, t1 for
multiple_sclerosis) cannot be expressed as a single static value in
pipeline_config.yaml — changing preprocessing_config.yaml's glioblastoma
setting from t1 to t1c broke stage 07 for every glioblastoma run:
"Atlas transform not found: ..._t1_to_atlas.mat" (the actual file was
..._t1c_to_atlas.mat).

Fix: resolve_reference_modality() mirrors 05_preprocessing.py's own
precedence — an explicit --reference-modality wins; otherwise derive it
from --preprocessing-config (the same file stage 05 reads); otherwise
fall back to lesion_types.yaml; otherwise 't1'.
"""
import sys
import importlib.util
import textwrap
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


inverse_transform = _load_module("07_inverse_transform.py", "inverse_transform_ref_modality")
resolve_reference_modality = inverse_transform.resolve_reference_modality


def _write_preprocessing_config(tmp_path, reference_modality):
    config_path = tmp_path / "preprocessing_config.yaml"
    config_path.write_text(textwrap.dedent(f"""
        steps:
          - name: reorient
            enabled: true
          - name: registration
            enabled: true
            params:
              reference_modality: "{reference_modality}"
    """))
    return config_path


class TestResolveReferenceModality:
    def test_explicit_value_always_wins(self, tmp_path):
        config_path = _write_preprocessing_config(tmp_path, "t1c")
        result = resolve_reference_modality(
            explicit="t1",
            preprocessing_config_path=config_path,
            lesion_type="glioblastoma",
        )
        assert result == "t1"

    def test_derives_from_preprocessing_config_when_not_explicit(self, tmp_path):
        """The exact scenario the bug broke: glioblastoma's
        preprocessing_config.yaml says t1c, nothing explicit given."""
        config_path = _write_preprocessing_config(tmp_path, "t1c")
        result = resolve_reference_modality(
            explicit=None,
            preprocessing_config_path=config_path,
            lesion_type="glioblastoma",
        )
        assert result == "t1c"

    def test_ms_preprocessing_config_still_resolves_to_t1(self, tmp_path):
        """Regression: MS's own preprocessing config (t1) must keep
        resolving correctly — this is the case a blind global value change
        to pipeline_config.yaml would have silently broken."""
        config_path = _write_preprocessing_config(tmp_path, "t1")
        result = resolve_reference_modality(
            explicit=None,
            preprocessing_config_path=config_path,
            lesion_type="multiple_sclerosis",
        )
        assert result == "t1"

    def test_falls_back_to_lesion_types_yaml_when_no_preprocessing_config(self):
        result = resolve_reference_modality(
            explicit=None,
            preprocessing_config_path=None,
            lesion_type="glioblastoma",
        )
        assert result == "t1c"  # configs/lesion_types.yaml's current value

    def test_falls_back_to_t1c_when_preprocessing_config_missing_registration_step(self, tmp_path):
        config_path = tmp_path / "preprocessing_config.yaml"
        config_path.write_text("steps:\n  - name: reorient\n    enabled: true\n")
        result = resolve_reference_modality(
            explicit=None,
            preprocessing_config_path=config_path,
            lesion_type="glioblastoma",
        )
        # No registration step in the file -> falls through to lesion_types.yaml
        assert result == "t1c"

    def test_nonexistent_preprocessing_config_path_falls_back_gracefully(self, tmp_path):
        result = resolve_reference_modality(
            explicit=None,
            preprocessing_config_path=tmp_path / "does_not_exist.yaml",
            lesion_type="glioblastoma",
        )
        assert result == "t1c"
