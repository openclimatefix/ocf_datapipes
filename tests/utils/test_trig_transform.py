import numpy as np

from ocf_datapipes.utils.utils import _trig_transform, trigonometric_datetime_transformation


def test_trig_transform():
    input = np.array([0, 0.5, 1]) * np.pi
    expected_sine = np.array([0.0, 1.0, 0.0])
    expected_cosine = np.array([1.0, 0.0, -1.0])
    actual_sine, actual_cosine = _trig_transform(input, 2 * np.pi)
    assert np.allclose(expected_cosine, actual_cosine) and np.allclose(expected_sine, actual_sine)


def test_trigonometric_datetime_transformations():
    input = np.array(
        [
            "2014-01-01T00:00:00.000000000",
            "2014-01-01T01:00:00.000000000",
            "2014-01-02T00:00:00.000000000",
            "2014-02-01T00:00:00.000000000",
        ],
        dtype="datetime64[ns]",
    )

    expected_output = np.array(
        [
            [
                np.sin(2 * np.pi * 1 / 12),
                np.cos(2 * np.pi * 1 / 12),
                np.sin(2 * np.pi * 1 / 366),
                np.cos(2 * np.pi * 1 / 366),
                np.sin(2 * np.pi * 0 / 24),
                np.cos(2 * np.pi * 0 / 24),
            ],
            [
                np.sin(2 * np.pi * 1 / 12),
                np.cos(2 * np.pi * 1 / 12),
                np.sin(2 * np.pi * 1 / 366),
                np.cos(2 * np.pi * 1 / 366),
                np.sin(2 * np.pi * 1 / 24),
                np.cos(2 * np.pi * 1 / 24),
            ],
            [
                np.sin(2 * np.pi * 1 / 12),
                np.cos(2 * np.pi * 1 / 12),
                np.sin(2 * np.pi * 2 / 366),
                np.cos(2 * np.pi * 2 / 366),
                np.sin(2 * np.pi * 0 / 24),
                np.cos(2 * np.pi * 0 / 24),
            ],
            [
                np.sin(2 * np.pi * 2 / 12),
                np.cos(2 * np.pi * 2 / 12),
                np.sin(2 * np.pi * 1 / 366),
                np.cos(2 * np.pi * 1 / 366),
                np.sin(2 * np.pi * 0 / 24),
                np.cos(2 * np.pi * 0 / 24),
            ],
        ]
    )

    output = trigonometric_datetime_transformation(input)
    assert np.allclose(output, expected_output)
    assert np.allclose(np.sum(output**2, axis=1), 3.0 * np.ones(output.shape[0]))
