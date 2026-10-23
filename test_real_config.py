"""
The repository's pipeline_config.yaml loads, resolves, and points every
enabled stage at a script that exists.

This used to be a print-only script whose entry point was named
test_real_config(config_path) — pytest took `config_path` for a fixture and
reported an error on every run (KI-063). The printout is still available:

    python test_real_config.py pipeline_config.yaml
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.config_loader import (  # noqa: E402
    get_enabled_stages,
    get_stage_input_dir,
    get_stage_output_dir,
    load_config,
    resolve_paths,
)

CONFIG = ROOT / "pipeline_config.yaml"


def _resolved(path=CONFIG):
    return resolve_paths(load_config(Path(path)))


def test_repo_config_loads_and_resolves():
    config = _resolved()
    assert config["general"]["root_output_dir"]
    assert get_enabled_stages(config), "no stage enabled"


def test_every_enabled_stage_has_its_script_and_dirs():
    config = _resolved()
    for stage in get_enabled_stages(config):
        script = config["stages"][stage]["script"]
        assert (ROOT / script).exists() or (ROOT / "scripts" / Path(script).name).exists(), \
            f"{stage}: script {script} not found"
        assert get_stage_output_dir(stage, config), f"{stage}: no output dir"
        get_stage_input_dir(stage, config)  # must not raise


def print_config(path):
    config = _resolved(path)
    general = config["general"]
    print(f"Config: {path}")
    print(f"  root_input_dir:  {general['root_input_dir']}")
    print(f"  root_output_dir: {general['root_output_dir']}")
    print(f"  max_subjects: {general.get('max_subjects')}  skip_existing: {general.get('skip_existing')}")
    enabled = set(get_enabled_stages(config))
    for stage, cfg in config["stages"].items():
        mark = "✓" if stage in enabled else "✗"
        print(f"  {mark} {stage}: {cfg.get('script')}")
        if stage in enabled:
            print(f"      in:  {get_stage_input_dir(stage, config)}")
            print(f"      out: {get_stage_output_dir(stage, config)}")
            for k, v in (cfg.get("args") or {}).items():
                print(f"      --{k}: {v}")


if __name__ == "__main__":
    print_config(sys.argv[1] if len(sys.argv) > 1 else CONFIG)
