import pytest
import kappa_dataset_resolver as resolver


def test_highest_number_of_an_empty_dataset_is_zero():
    assert resolver.highest_subject_number([]) == 0


def test_highest_number_reads_entity_names():
    entities = [
        {"dsEntityName": "sub-001_ses-001"},
        {"dsEntityName": "sub-007_ses-002"},
        {"dsEntityName": "sub-003_ses-001"},
    ]

    assert resolver.highest_subject_number(entities) == 7


def test_highest_number_ignores_names_that_are_not_bids():
    """Someone may have uploaded by hand. Unparseable names must not crash the
    run or be read as a number."""
    entities = [{"dsEntityName": "notes.txt"}, {"dsEntityName": "sub-004_ses-001"},
                {"dsEntityName": None}, {}]

    assert resolver.highest_subject_number(entities) == 4


@pytest.mark.asyncio
async def test_resolve_returns_the_mapped_dataset_without_creating(monkeypatch):
    monkeypatch.setattr(resolver, "get_dataset_id", lambda *a: 337)
    async def fail_create(**kwargs):
        raise AssertionError("must not create when the mapping already has one")
    monkeypatch.setattr(resolver, "create_dataset", fail_create)

    got = await resolver.resolve_or_create(
        token="t", user_id=26, user_type_id=3,
        lesion_type="glioblastoma", preprocessing_id="1099b9cd")

    assert got == 337


@pytest.mark.asyncio
async def test_resolve_creates_and_registers_when_unmapped(monkeypatch):
    monkeypatch.setattr(resolver, "get_dataset_id", lambda *a: None)
    async def fake_create(**kwargs):
        return 350
    recorded = {}
    monkeypatch.setattr(resolver, "create_dataset", fake_create)
    monkeypatch.setattr(resolver, "set_dataset_id",
                        lambda u, l, p, d: recorded.update(user=u, lesion=l, prep=p, ds=d))

    got = await resolver.resolve_or_create(
        token="t", user_id=52, user_type_id=4,
        lesion_type="glioblastoma", preprocessing_id="1099b9cd")

    assert got == 350
    assert recorded == {"user": 52, "lesion": "glioblastoma",
                        "prep": "1099b9cd", "ds": 350}


@pytest.mark.asyncio
async def test_resolve_without_create_returns_none_when_unmapped(monkeypatch):
    """Run start uses create=False when Kappa is unreachable — it must report
    'no dataset' rather than raise."""
    monkeypatch.setattr(resolver, "get_dataset_id", lambda *a: None)

    got = await resolver.resolve_or_create(
        token="t", user_id=52, user_type_id=4, lesion_type="glioblastoma",
        preprocessing_id="1099b9cd", create=False)

    assert got is None


@pytest.mark.asyncio
async def test_floor_is_none_when_kappa_cannot_be_reached(monkeypatch):
    """None means 'unknown', which the caller must not confuse with 0 —
    0 would restart numbering at sub-001 in a populated dataset."""
    async def boom(**kwargs):
        raise OSError("connection refused")
    monkeypatch.setattr(resolver, "get_dataset_entities", boom)

    assert await resolver.dataset_floor("t", 26, 3, 337) is None


@pytest.mark.asyncio
async def test_new_dataset_tags_include_predefined_ml_tag(monkeypatch):
    """Kappa rejects dataset creation unless datasetTags has a predefined ML
    tag. "Image Segmentation" is the predefined tag (Computer Vision is the ML
    task type, a separate field, not a tag). Moved here from
    test_kappa_uploader_resolve.py when creation moved into this module."""
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return 321

    monkeypatch.setattr(resolver, "get_dataset_id", lambda *a: None)
    monkeypatch.setattr(resolver, "set_dataset_id", lambda *a, **k: None)
    monkeypatch.setattr(resolver, "create_dataset", fake_create)

    result = await resolver.resolve_or_create(
        token="t", user_id=52, user_type_id=4,
        lesion_type="glioblastoma", preprocessing_id="abc12345")

    assert result == 321
    assert "Image Segmentation" in captured["dataset_tags"]
