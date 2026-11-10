"""Experiment titles say what was evaluated, not the raw dataset name."""

from traccia.cli import experiment_name


def test_task_script_uses_the_dataset_plus_eval():
    assert experiment_name(None, "ci-manual", "task.py", {}) == "ci-manual eval"


def test_meaningful_script_is_the_subject():
    assert experiment_name(None, "support-golden", "experiments/support_agent.py", {}) == (
        "support_agent eval"
    )


def test_branch_and_explicit_name():
    assert experiment_name(None, "ci-manual", "task.py", {"TRACCIA_GIT_HEAD_REF": "feat/tone"}) == (
        "ci-manual eval · feat/tone"
    )
    assert experiment_name("Refund Check", "ci-manual", "task.py", {"TRACCIA_GIT_HEAD_REF": "feat/tone"}) == (
        "Refund Check"
    )
