import pytest

from daignostic.labels import FINDINGS, nih_labels_to_vector


def test_findings_are_seven_and_unique():
    assert len(FINDINGS) == 7
    assert len(set(FINDINGS)) == 7


def test_single_finding():
    assert nih_labels_to_vector("Cardiomegaly") == [1.0, 0, 0, 0, 0, 0, 0]


def test_effusion_name_is_mapped():
    v = nih_labels_to_vector("Effusion")
    assert v[FINDINGS.index("Pleural effusion")] == 1.0


def test_nodule_and_mass_merge():
    assert nih_labels_to_vector("Nodule|Mass") == nih_labels_to_vector("Mass")


def test_multi_label():
    assert sum(nih_labels_to_vector("Effusion|Atelectasis")) == 2.0


def test_no_finding_is_all_zero():
    assert nih_labels_to_vector("No Finding") == [0.0] * 7


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        nih_labels_to_vector("Effusoin")   # typo on purpose