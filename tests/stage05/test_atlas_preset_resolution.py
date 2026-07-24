"""
Config footgun: atlas.name (a label: "SRI24"/"MNI152_FSL"/"MNI152_ICBM")
and atlas.filename (the actual cached file to load) were two independent
fields. prepare_atlas() only ever read `filename` — `name` was pure
documentation with no effect on which file was loaded. Changing only
`name` (as the comment beside it implies you can) silently kept using
whatever file `filename` still pointed to.

Fix: `name` is resolved to a filename via ATLAS_PRESETS. An explicit
`filename` in config still overrides it (for an atlas not yet in the
preset list), but is no longer required for the three known presets.
"""
import sys
import importlib.util
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


preprocessing = _load_module("05_preprocessing.py", "preprocessing_atlas_preset")
resolve_atlas_filename = preprocessing.resolve_atlas_filename


class TestResolveAtlasFilename:
    def test_name_only_resolves_to_preset_filename(self):
        """The exact scenario the footgun broke: only `name` set."""
        assert resolve_atlas_filename({"name": "MNI152_FSL"}) == "MNI152_T1_1mm.nii.gz"

    def test_icbm_name_resolves_correctly(self):
        assert resolve_atlas_filename({"name": "MNI152_ICBM"}) == "mni_icbm152_t1_tal_nlin_sym_09a.nii"

    def test_sri24_name_resolves_correctly(self):
        assert resolve_atlas_filename({"name": "SRI24"}) == "sri24_t1.nii.gz"

    def test_explicit_filename_overrides_name(self):
        """Escape hatch for a custom atlas not yet in ATLAS_PRESETS."""
        config = {"name": "MNI152_FSL", "filename": "custom_template.nii.gz"}
        assert resolve_atlas_filename(config) == "custom_template.nii.gz"

    def test_unknown_name_without_filename_falls_back_to_legacy_default(self):
        """Preserves prior behavior for configs missing both fields."""
        assert resolve_atlas_filename({}) == "sri24_t1.nii.gz"
        assert resolve_atlas_filename({"name": "TypoedName"}) == "sri24_t1.nii.gz"
