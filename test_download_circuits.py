# test_download_circuits.py
#
# What it does: checks the outline maths in download_circuits.py on simple
#   shapes, without touching the internet.
# What it reads: nothing.
# What it produces: pass/fail results when run with pytest.
# Which files use it: none.  Run:  .venv\Scripts\python.exe -m pytest

import numpy as np
import polars as pl

import download_circuits


def test_resampled_points_are_the_same_distance_apart():
    # A 10 x 10 square "track" recorded unevenly: many points on one side.
    square = np.array([[0, 0], [2, 0], [3, 0], [9, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    points = download_circuits.resample_evenly(square, 40)
    assert len(points) == 40
    loop = np.vstack([points, points[:1]])
    gaps = np.hypot(*np.diff(loop, axis=0).T)
    # The lap is 40 long, so 40 points sit 1 apart (corners cut slightly less).
    assert np.allclose(gaps, 1.0, atol=0.01)


def test_outline_fits_the_box_without_being_stretched():
    # A track twice as wide as it is tall.
    points = np.array([[-200, 50], [200, 50], [200, 250], [-200, 250]], dtype=float)
    fitted = download_circuits.fit_to_box(points, 1000)
    assert fitted[:, 0].min() == 0 and fitted[:, 0].max() == 1000
    assert fitted[:, 1].min() == 0 and fitted[:, 1].max() == 500
    # Y is flipped for the screen: the lowest point (y=50) ends up at the bottom.
    assert fitted[0].tolist() == [0, 500]


def test_a_frozen_position_feed_is_rejected():
    angles = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    smooth = np.column_stack([np.cos(angles), np.sin(angles)])
    frozen = np.repeat(smooth[::25], 25, axis=0)  # 300 rows, only 12 positions
    assert download_circuits.has_enough_detail(smooth)
    assert not download_circuits.has_enough_detail(frozen)


def test_rotating_by_a_quarter_turn():
    turned = download_circuits.rotate(np.array([[1.0, 0.0]]), 90)
    assert np.allclose(turned, [[0.0, 1.0]])


def test_races_per_circuit_are_newest_first_and_ignore_old_seasons():
    results = pl.DataFrame({
        "Year": [2010, 2019, 2024, 2024],
        "Round": [1, 5, 3, 3],
        "CircuitId": ["adelaide", "monza", "monza", "monza"],
        "CircuitName": ["Adelaide", "Monza", "Monza", "Monza"],
        "Country": ["Australia", "Italy", "Italy", "Italy"],
        "Laps": [80, 53, 53, 20],
    })
    races = download_circuits.races_per_circuit(results)
    assert list(races) == ["monza"]
    assert [(race["Year"], race["RaceLaps"]) for race in races["monza"]] == [(2024, 53), (2019, 53)]
