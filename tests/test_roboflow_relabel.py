from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parents[1] / "tools" / "roboflow_relabel"))

from core import Box, extract_predictions, propose_insulator_additions, propose_matches  # noqa: E402


def test_extract_predictions_from_nested_workflow_result():
    result = [{"output": {"predictions": [{
        "x": 50,
        "y": 40,
        "width": 60,
        "height": 20,
        "confidence": 0.9,
        "class": "insulator",
        "class_id": 0,
    }]}}]

    boxes = extract_predictions(result)

    assert boxes == [Box(20, 30, 80, 50, 0, 0.9, "insulator")]


def test_proposal_requires_full_box_to_contain_broken_box_center():
    old = Box(40, 35, 60, 45, 1)
    full = Box(10, 25, 90, 55, 0, 0.95, "insulator")
    unrelated = Box(100, 100, 180, 140, 0, 0.99, "insulator")

    proposals = propose_matches([old], [full, unrelated], {"insulator"}, 0.5, 1.1, 20)

    assert proposals[0]["status"] == "proposed"
    assert proposals[0]["match_count"] == 1
    assert proposals[0]["new_box"]["x1"] == 10


def test_proposal_marks_multiple_containing_boxes_ambiguous():
    old = Box(40, 35, 60, 45, 1)
    predictions = [
        Box(10, 25, 90, 55, 0, 0.95, "insulator"),
        Box(20, 20, 80, 60, 0, 0.90, "insulator"),
    ]

    proposals = propose_matches([old], predictions, {"insulator"}, 0.5, 1.1, 20)

    assert proposals[0]["status"] == "ambiguous"
    assert proposals[0]["match_count"] == 2


def test_external_insulator_is_mapped_to_dataset_class_zero():
    current = [Box(40, 35, 60, 45, 1)]
    workflow = [Box(10, 25, 90, 55, 7, 0.95, "Insulator")]

    additions = propose_insulator_additions(current, workflow, {"insulator"}, 0.5)

    assert additions[0]["status"] == "add"
    assert additions[0]["box"]["class_id"] == 0
    assert additions[0]["box"]["class_name"] == "insulator"


def test_existing_class_zero_box_is_not_added_twice():
    current = [Box(10, 25, 90, 55, 0), Box(40, 35, 60, 45, 1)]
    workflow = [Box(11, 25, 89, 55, 9, 0.95, "Insulator")]

    additions = propose_insulator_additions(current, workflow, {"insulator"}, 0.5)

    assert additions[0]["status"] == "duplicate"
