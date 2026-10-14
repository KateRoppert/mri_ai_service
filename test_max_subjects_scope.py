"""The patient limit belongs to stage 01 alone.

`general.max_subjects` was injected into every stage, and each stage then
applied it to whatever it happened to be iterating. Two things went wrong
at once on an MS run of one patient with two sessions:

  - stage 05 counts (subject, session) pairs, so a limit of 1 took one
    SESSION and the patient's second session never reached preprocessing,
    segmentation or the longitudinal view;
  - nothing downstream needs the limit anyway. If stage 01 admits one
    patient, every later stage sees one patient by construction.

So the limit is a statement about how many patients ENTER the pipeline, and
it is injected only where that is decided — which is stage 01 when stage 01
runs, and every enabled stage when it does not, because then the run has no
entrance and nothing would limit it at all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import build_command


def _config(max_subjects=3):
    return {
        'general': {
            'root_input_dir': Path('/tmp/in'),
            'root_output_dir': Path('/tmp/out'),
            'max_subjects': max_subjects,
            'skip_existing': True,
            'lesion_type': 'multiple_sclerosis',
        },
        'output_structure': {
            'stage_01': 'bids', 'stage_02': 'metadata', 'stage_03': 'nifti',
            'stage_04': 'quality', 'stage_05_data': 'preprocessed',
            'stage_05_transforms': 'transforms', 'stage_06': 'segmentation',
            'stage_07': 'segmentation', 'stage_08': 'segmentation',
            'logs': 'logs', 'reports': 'reports',
        },
        'stages': {
            # build_command reads stage_config['args'] directly, so a real
            # config always has the key.
            name: {'script': f'scripts/{name}.py', 'args': {}, 'enabled': True}
            for name in (
                'stage_01_reorganize', 'stage_03_convert', 'stage_04_quality',
                'stage_05_preprocessing', 'stage_06_segmentation',
                'stage_07_inverse_transform', 'stage_08_anatomical_analysis',
            )
        },
    }


def test_stage_01_still_gets_the_limit():
    cmd = build_command('stage_01_reorganize', _config(), Path('/project'))

    assert '--max-subjects' in cmd
    assert '3' in cmd


def test_no_later_stage_gets_the_limit():
    """Stage 05 is the one that bit us — it counts sessions, so a limit of 1
    dropped the second session of a single patient. But none of them need
    it: they process what stage 01 admitted."""
    config = _config()
    for stage in (
        'stage_03_convert', 'stage_04_quality', 'stage_05_preprocessing',
        'stage_06_segmentation', 'stage_07_inverse_transform',
        'stage_08_anatomical_analysis',
    ):
        cmd = build_command(stage, config, Path('/project'))
        assert '--max-subjects' not in cmd, f"{stage} всё ещё получает лимит"


def test_no_limit_configured_means_no_flag_anywhere():
    config = _config(max_subjects=None)

    for stage in ('stage_01_reorganize', 'stage_05_preprocessing'):
        cmd = build_command(stage, config, Path('/project'))
        assert '--max-subjects' not in cmd


def test_a_stage_declaring_its_own_limit_is_left_alone():
    """The existing escape hatch: an explicit args.max_subjects wins, so a
    stage can still be limited on purpose."""
    config = _config()
    config['stages']['stage_01_reorganize']['args'] = {'max_subjects': None}

    cmd = build_command('stage_01_reorganize', config, Path('/project'))

    assert '--max-subjects' not in cmd


def test_later_stages_get_the_limit_when_stage_01_is_disabled():
    """A re-run from a later stage has no entrance, so each stage must apply
    the limit itself or the config's limit silently does nothing — on a 252
    patient dataset in a 20g container that is how stage 05 OOMs the host."""
    config = _config()
    config['stages']['stage_01_reorganize']['enabled'] = False

    for stage in ('stage_03_convert', 'stage_05_preprocessing',
                  'stage_08_anatomical_analysis'):
        cmd = build_command(stage, config, Path('/project'))
        assert '--max-subjects' in cmd, f"{stage} остался без лимита"
        assert cmd[cmd.index('--max-subjects') + 1] == '3'


def test_a_missing_enabled_key_keeps_the_limit_at_the_entrance():
    """Widening the limit to every stage is the branch with the hazard, so it
    takes an explicit `enabled: false` — not a key that happens to be absent."""
    config = _config()
    del config['stages']['stage_01_reorganize']['enabled']

    assert '--max-subjects' not in build_command(
        'stage_05_preprocessing', config, Path('/project'))


def test_the_dashed_key_is_recognised_as_a_stage_s_own_limit():
    """The orchestrator passes arg names through verbatim, so `max-subjects`
    is the spelling that actually reaches argparse. The guard used to check
    only `max_subjects`, which let stage 01 be handed the flag twice."""
    config = _config()
    config['stages']['stage_01_reorganize']['args'] = {'max-subjects': 5}

    cmd = build_command('stage_01_reorganize', config, Path('/project'))

    assert cmd.count('--max-subjects') == 1
    assert cmd[cmd.index('--max-subjects') + 1] == '5'
