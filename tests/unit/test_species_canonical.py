"""`SpeciesRegistry.resolve_name` / `canonical_name` and the insert gate in
`add_annotations`.

The registry is built from the test env's class map (`write_test_species_registry`
in conftest), so the names below are the ones it knows: seven species plus the
`fish` / `bait` buckets, and the synonyms in `TEST_SYNONYMS`.

An unknown name never raises: it is kept as written and logged at ERROR.
"""

import pytest
import yaml

from spyfish.config.base import NULL_DEPLOYMENT
from spyfish.config.species import species_registry
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from tests.conftest import TEST_SPECIES, TEST_SYNONYMS, write_test_species_registry

DROP = "KSF_20240124_BUV_KSF_085_01"


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_project_root", tmp_path)
    write_test_species_registry(tmp_path)
    return species_registry()


def _row(species, **kw):
    return {
        "drop_id": DROP,
        "scientific_name": species,
        "time_of_max": "00:00:10",
        "max_interval": 1,
        "annotated_by": "expert",
        "interval_annotation": None,
        "confidence_agreement": None,
        "external_id": "legacy",
        **kw,
    }


def _errors(caplog):
    return [r for r in caplog.records if r.levelname == "ERROR"]


# ── resolution ─────────────────────────────────────────────────────────────


def test_registry_name_is_returned_unchanged(registry):
    assert registry.canonical_name("Chrysophrys auratus") == "Chrysophrys auratus"
    assert registry.canonical_name("fish") == "fish"


def test_synonym_resolves_to_canonical(registry):
    assert registry.canonical_name("Pseudocaranx dentex") == "Pseudocaranx georgianus"
    assert registry.canonical_name("Parika scaber") == "Meuschenia scabra"


def test_synonym_chain_follows_to_fixed_point(registry):
    # "Unidentified fish" -> "Unidentified" -> "fish"
    assert registry.canonical_name("Unidentified fish") == "fish"
    assert registry.canonical_name("Sp1") == "fish"


def test_alias_and_biigle_label_forms_resolve_to_scientific(registry):
    assert registry.canonical_name("Fish: final") == "fish"
    assert (
        registry.canonical_name("Snapper - Chrysophrys auratus")
        == "Chrysophrys auratus"
    )


def test_whitespace_is_stripped(registry):
    assert registry.canonical_name("  Chrysophrys auratus ") == "Chrysophrys auratus"


def test_absence_markers_pass_through(registry):
    assert registry.canonical_name(None) is None
    assert registry.canonical_name("") is None
    assert registry.canonical_name(NULL_DEPLOYMENT) == NULL_DEPLOYMENT


def test_every_test_species_round_trips(registry):
    for sci, _common, _aphia in TEST_SPECIES:
        assert registry.resolve_name(sci) == (sci, None)


# The BIIGLE tree still carries "White Trevally - Pseudocaranx dentex" (label
# 477433) next to "Silver trevally - Pseudocaranx georgianus". P. dentex is an
# Atlantic species: every NZ record under that name is georgianus.
RETIRED_TREVALLY = ("Pseudocaranx dentex", "White Trevally", 126812)


def test_label_form_of_a_retired_name_lands_on_the_current_name(tmp_path, monkeypatch):
    """The label resolves to its scientific name, which is a synonym key, so
    the chain must keep going rather than stop at the registry entry."""
    monkeypatch.setattr(config, "_project_root", tmp_path)
    write_test_species_registry(tmp_path, species=TEST_SPECIES + [RETIRED_TREVALLY])
    reg = species_registry()
    assert reg.resolve_name("White Trevally - Pseudocaranx dentex") == (
        "Pseudocaranx georgianus",
        None,
    )
    assert reg.canonical_name("Pseudocaranx dentex") == "Pseudocaranx georgianus"
    # The retired entry stays a registry member: YOLO label files written
    # against this class map still decode through its class_id.
    assert reg.get("Pseudocaranx dentex").class_id is not None


def test_zoo_choice_matches_a_class_map_alias(tmp_path, monkeypatch):
    """Volunteers pick TREVALLY; the tree says 'Silver trevally'. The alias on
    the class-map entry answers for the choice, the common name still does too."""
    monkeypatch.setattr(config, "_project_root", tmp_path)
    write_test_species_registry(
        tmp_path, aliases={"Pseudocaranx georgianus": ["Trevally"]}
    )
    reg = species_registry()
    assert reg.from_zoo_choice("TREVALLY").scientific_name == "Pseudocaranx georgianus"
    assert (
        reg.from_zoo_choice("SILVER TREVALLY").scientific_name
        == "Pseudocaranx georgianus"
    )
    assert reg.from_zoo_choice("WHITE TREVALLY") is None


def test_zoo_choice_normaliser_drops_apostrophes_and_slashes(tmp_path, monkeypatch):
    """SANDAGERSWRASSE and GREENBONEBUTTERFISH are how the workflow spells
    "Sandager's wrasse" and "Greenbone/butterfish"; both fell to `fish`."""
    monkeypatch.setattr(config, "_project_root", tmp_path)
    write_test_species_registry(
        tmp_path,
        species=TEST_SPECIES
        + [
            ("Coris sandeyeri", "Sandager's wrasse", 273563),
            ("Odax pullus", "Greenbone/butterfish", 281815),
        ],
    )
    reg = species_registry()
    assert reg.from_zoo_choice("SANDAGERSWRASSE").scientific_name == "Coris sandeyeri"
    assert reg.from_zoo_choice("GREENBONEBUTTERFISH").scientific_name == "Odax pullus"


def test_biigle_label_id_is_reachable_under_the_canonical_name(tmp_path, monkeypatch):
    """BIIGLE still says 'Snapper - Pagrus auratus' while the registry says
    Chrysophrys auratus. The label id must follow the synonym, or snapper
    frames upload under the default fish label."""
    import pandas as pd

    monkeypatch.setattr(config, "_project_root", tmp_path)
    write_test_species_registry(
        tmp_path, synonyms={**TEST_SYNONYMS, "Pagrus auratus": "Chrysophrys auratus"}
    )
    labels = config.species_labels_csv_path
    labels.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"id": 477416, "name": "Snapper - Pagrus auratus"}]).to_csv(
        labels, index=False
    )
    reg = species_registry()
    assert reg.biigle_label_id_for("Chrysophrys auratus") == 477416
    assert reg.biigle_label_id_for("Snapper - Chrysophrys auratus") == 477416
    assert reg.canonical_name("Snapper - Pagrus auratus") == "Chrysophrys auratus"


def test_label_join_prefers_the_same_genus_spelling_when_two_labels_fold(
    tmp_path, monkeypatch, caplog
):
    """The tree has both the misspelt 'Meuschenia scaber' label and the
    retired lookup-error label 'Oligoplites saurus'; both are synonyms of
    M. scabra. The spelling variant is the real label; the other must not win
    just because it comes first in the CSV (it did, 2026-11-04)."""
    import pandas as pd

    monkeypatch.setattr(config, "_project_root", tmp_path)
    write_test_species_registry(
        tmp_path,
        synonyms={
            **TEST_SYNONYMS,
            "Meuschenia scaber": "Meuschenia scabra",
            "Oligoplites saurus": "Meuschenia scabra",
        },
    )
    labels = config.species_labels_csv_path
    labels.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"id": 477462, "name": "Leather jack - Oligoplites saurus"},
            {"id": 477365, "name": "Smooth leatherjacket - Meuschenia scaber"},
        ]
    ).to_csv(labels, index=False)
    with caplog.at_level("WARNING"):
        reg = species_registry()
    assert reg.biigle_label_id_for("Meuschenia scabra") == 477365
    assert "fold onto 'Meuschenia scabra'" in caplog.text


def test_zoo_choice_on_a_retired_name_is_written_canonical(tmp_path, monkeypatch):
    """A choice whose key is a retired common name must not put the retired
    scientific name in the Zooniverse MaxN CSV."""
    from spyfish.zooniverse.parse_classifications import _zoo_choice_to_scientific

    monkeypatch.setattr(config, "_project_root", tmp_path)
    write_test_species_registry(tmp_path, species=TEST_SPECIES + [RETIRED_TREVALLY])
    assert _zoo_choice_to_scientific("WHITE TREVALLY") == "Pseudocaranx georgianus"
    assert _zoo_choice_to_scientific("SNAPPER") == "Chrysophrys auratus"
    assert _zoo_choice_to_scientific("NOT A CHOICE") == "fish"
    assert _zoo_choice_to_scientific("") is None


# ── unknown names: kept, logged, never raised ──────────────────────────────


def test_unknown_name_is_kept_and_logged_with_both_sources_named(registry, caplog):
    with caplog.at_level("ERROR"):
        assert registry.canonical_name("Oligoplites saurus") == "Oligoplites saurus"
    (err,) = _errors(caplog)
    assert "Oligoplites saurus" in err.message
    assert "class_map.json" in err.message and "species_synonyms" in err.message


def test_resolve_name_reports_the_problem_without_logging(registry, caplog):
    with caplog.at_level("ERROR"):
        canonical, problem = registry.resolve_name("Oligoplites saurus")
    assert canonical == "Oligoplites saurus"
    assert "not in the species registry" in problem
    assert _errors(caplog) == []
    assert registry.resolve_name("Pseudocaranx dentex") == (
        "Pseudocaranx georgianus",
        None,
    )


def test_synonym_pointing_at_unknown_name_is_kept_and_shows_chain(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(config, "_project_root", tmp_path)
    write_test_species_registry(
        tmp_path, synonyms={"Coris sandageri": "Coris sandeyeri"}
    )
    with caplog.at_level("ERROR"):
        assert species_registry().canonical_name("Coris sandageri") == "Coris sandageri"
    assert "Coris sandageri -> Coris sandeyeri" in caplog.text


def test_synonym_cycle_is_kept_and_logged(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(config, "_project_root", tmp_path)
    write_test_species_registry(tmp_path, synonyms={"A": "B", "B": "A"})
    with caplog.at_level("ERROR"):
        assert species_registry().canonical_name("A") == "A"
    assert "cycle" in caplog.text


def test_species_absent_from_class_map_is_kept_and_logged(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(config, "_project_root", tmp_path)
    write_test_species_registry(tmp_path, species=[])
    # Only the buckets (and training.class_order orphans) remain.
    reg = species_registry()
    assert reg.canonical_name("fish") == "fish"
    with caplog.at_level("ERROR"):
        assert reg.canonical_name("Notolabrus fucicola") == "Notolabrus fucicola"
    assert "Notolabrus fucicola" in caplog.text


# ── the map file: missing is tolerated by readers, refused by ingest ───────


def test_missing_synonyms_file_warns_readers_and_blocks_ingest(
    tmp_path, monkeypatch, caplog
):
    """process_files/ is gitignored, so a fresh checkout has no map. Dashboards
    and uploads must still resolve names (like they do without class_map.json);
    add_annotations, the one writer the map protects, refuses to run."""
    monkeypatch.setattr(config, "_project_root", tmp_path)
    write_test_species_registry(tmp_path)
    config.species_synonyms_path.unlink()
    with caplog.at_level("WARNING"):
        reg = species_registry()
    assert reg.synonyms_missing and reg.synonyms == {}
    assert "species_synonyms" in caplog.text
    assert reg.canonical_name("Chrysophrys auratus") == "Chrysophrys auratus"

    db = AnnotationDatabaseManager(db_path=str(tmp_path / "a.db"))
    with pytest.raises(FileNotFoundError, match="species_synonyms"):
        db.add_annotations([_row("Chrysophrys auratus")])
    assert db.get_annotations_for_drop(DROP) == []


def test_synonyms_file_must_be_a_flat_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_project_root", tmp_path)
    write_test_species_registry(tmp_path)
    config.species_synonyms_path.write_text(yaml.safe_dump(["not", "a", "map"]))
    with pytest.raises(ValueError, match="flat"):
        species_registry()


# ── add_annotations gate ───────────────────────────────────────────────────


def test_add_annotations_stores_canonical_names(registry, tmp_path):
    db = AnnotationDatabaseManager(db_path=str(tmp_path / "a.db"))
    db.add_annotations(
        [
            _row("Pseudocaranx dentex"),
            _row("Sp1"),
            _row("Pagrus auratus"),  # synonym key, stored under the WoRMS name
            _row(NULL_DEPLOYMENT, max_interval=0),
        ]
    )
    stored = sorted(
        str(r["scientific_name"]) for r in db.get_annotations_for_drop(DROP)
    )
    assert stored == [
        "Chrysophrys auratus",
        NULL_DEPLOYMENT,
        "Pseudocaranx georgianus",
        "fish",
    ]


def test_add_annotations_keeps_unknown_names_and_logs_one_error_per_name(
    registry, tmp_path, caplog
):
    db = AnnotationDatabaseManager(db_path=str(tmp_path / "a.db"))
    with caplog.at_level("ERROR"):
        db.add_annotations(
            [
                _row("Chrysophrys auratus"),
                _row("Thalassoma sp."),
                _row("Thalassoma sp."),
            ]
        )
    stored = sorted(
        str(r["scientific_name"]) for r in db.get_annotations_for_drop(DROP)
    )
    assert stored == ["Chrysophrys auratus", "Thalassoma sp.", "Thalassoma sp."]
    (err,) = _errors(caplog)
    assert "Thalassoma sp." in err.message and "2 row(s)" in err.message
