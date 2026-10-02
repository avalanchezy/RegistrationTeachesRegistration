import numpy as np
import pytest

from task2reg.journal.verification import (
    VerificationConfig, point_uncertainty, pose_displacement,
    transform_medoid, verify_registration,
)


def translation(x):
    transform = np.eye(4)
    transform[0, 3] = x
    return transform


def evidence():
    points = np.column_stack([np.arange(100), np.zeros((100, 2))])
    return dict(transforms=np.stack([translation(-.1), translation(0), translation(.1)]),
                points=points, anchors=points[::10], distances=np.full(100, .2),
                heldout_distances=np.full(30, .3), sector_ids=np.repeat(np.arange(10), 10),
                outside_mask=np.zeros(100, dtype=bool),
                candidate_transforms=np.stack([translation(0), translation(4)]),
                candidate_energies=np.array([.2, 1.]))


def test_medoid_is_geometric_and_point_weights_reflect_dispersion():
    data = evidence()
    result = verify_registration(**data)
    assert result['accepted'], result['reasons']
    assert transform_medoid(data['transforms'], data['anchors']) == 1
    assert result['medoid_index'] == 1
    assert pose_displacement(translation(0), translation(3), data['anchors']) == pytest.approx(3)
    np.testing.assert_allclose(point_uncertainty(data['transforms'], translation(0), data['points']), np.sqrt(.02 / 3))
    np.testing.assert_allclose(result['point_weights'], np.exp(-(.02 / 3) / (2 * .75 ** 2)))
    assert result['config']['min_global_coverage'] == .85


@pytest.mark.parametrize('key,value,reason', [
    ('heldout_distances', np.full(30, 2.), 'heldout_median'),
    ('distances', np.full(100, 3.), 'global_coverage'),
    ('outside_mask', np.arange(100) < 5, 'outside_fraction'),
    ('transforms', np.stack([translation(-3), translation(0), translation(3)]), 'pose_uncertainty'),
])
def test_rejects_bad_fit_and_consensus(key, value, reason):
    data = evidence()
    data[key] = value
    result = verify_registration(**data)
    assert not result['accepted']
    assert reason in result['reasons']
    assert np.count_nonzero(result['point_weights']) == 0


def test_sector_failure_cannot_hide_in_global_coverage():
    data = evidence()
    data['distances'][:10] = 3
    result = verify_registration(**data)
    assert result['metrics']['global_coverage'] == pytest.approx(.9)
    assert not result['accepted']
    assert 'sector_coverage' in result['reasons']


def test_duplicate_best_candidate_does_not_hide_distinct_basin():
    data = evidence()
    data['candidate_transforms'] = np.stack([translation(0), translation(0), translation(5)])
    data['candidate_energies'] = np.array([1., 1., 1.05])
    result = verify_registration(**data)
    assert not result['accepted']
    assert 'distinct_basin_ambiguity' in result['reasons']


def test_improper_transforms_are_supported_but_mixed_parity_rejected():
    data = evidence()
    data['transforms'][:, 0, 0] = -1
    data['candidate_transforms'][:, 0, 0] = -1
    assert verify_registration(**data)['accepted']
    data['transforms'][0, 0, 0] = 1
    result = verify_registration(**data)
    assert not result['accepted']
    assert 'parity' in result['reasons']


@pytest.mark.parametrize('key,value', [
    ('transforms', None), ('heldout_distances', []), ('anchors', []),
    ('distances', np.full(100, np.nan)), ('outside_mask', np.full(100, np.nan)),
    ('sector_ids', np.full(100, np.nan)), ('candidate_energies', []),
    ('candidate_energies', np.array([.1, np.inf])),
])
def test_missing_empty_nonfinite_evidence_fails_closed(key, value):
    data = evidence()
    data[key] = value
    result = verify_registration(**data)
    assert not result['accepted']
    assert result['reasons']
    assert np.count_nonzero(result['point_weights']) == 0


def test_invalid_configuration_cannot_disable_verification():
    with pytest.raises(ValueError):
        VerificationConfig(max_outside_fraction=np.nan)
    with pytest.raises(ValueError):
        VerificationConfig(uncertainty_sigma_mm=0)


def test_uncertain_points_zeroed_even_if_pose_thresholds_relaxed():
    data = evidence()
    data['transforms'] = np.stack([translation(-3), translation(0), translation(3)])
    config = VerificationConfig(u50_mm=4, u95_mm=4, max_displacement_mm=4)
    result = verify_registration(**data, config=config)
    assert not result['accepted']
    assert 'no_supervision_mass' in result['reasons']
    assert np.count_nonzero(result['point_weights']) == 0


def test_anchor_uncertainty_preserves_spatial_rotation_tails():
    data = evidence()
    transforms = []
    for theta in (-.1, 0., .1):
        t = np.eye(4)
        t[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        transforms.append(t)
    data['transforms'] = np.stack(transforms)
    data['anchors'] = np.concatenate([np.zeros((90, 3)), np.tile([100., 0, 0], (10, 1))])
    result = verify_registration(**data, config=VerificationConfig(u50_mm=2.))
    # Most anchors remain fixed, but the distant tenth moves ~8.16 mm RMS.
    assert result['metrics']['u50_mm'] == pytest.approx(0.)
    assert result['metrics']['u95_mm'] == pytest.approx(200 * np.sin(.05) * np.sqrt(2 / 3))
    assert not result['accepted']
    assert 'pose_uncertainty' in result['reasons']


def test_medoid_in_worse_basin_cannot_ignore_better_distant_candidate():
    data = evidence()
    data['candidate_energies'] = np.array([1., .2])
    result = verify_registration(**data)
    assert not result['accepted']
    assert 'medoid_worse_basin' in result['reasons']


def test_unscored_medoid_fails_closed():
    data = evidence()
    data['candidate_transforms'] = np.stack([translation(1), translation(4)])
    result = verify_registration(**data)
    assert not result['accepted']
    assert 'missing_medoid_candidate' in result['reasons']
