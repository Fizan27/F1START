# test_app.py
#
# What it does: starts the website without a browser and checks that the
#   page builds for both seasons without an error, and that changing the
#   strategy controls re-runs the simulation.
# What it reads: everything app.py reads (app_data/, models/, results/).
# What it produces: pass/fail results when run with pytest.
# Which files use it: none.  Run:  .venv\Scripts\python.exe -m pytest

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

needs_final_models = pytest.mark.skipif(
    not Path("models/final/strategist.pt").exists() or not Path("app_data/races.pkl").exists(),
    reason="run final_test.py first: the website needs its models and race files",
)


def open_app() -> AppTest:
    app = AppTest.from_file("app.py", default_timeout=300).run()
    assert not app.exception, app.exception
    return app


@needs_final_models
def test_the_page_builds_for_both_seasons():
    app = open_app()
    assert "what if they had pitted differently" in app.title[0].value
    assert len(app.metric) >= 6  # the real race and the three strategies
    app.sidebar.selectbox[0].set_value(2024).run()
    assert not app.exception, app.exception
    assert len(app.metric) >= 6


@needs_final_models
def test_an_illegal_strategy_shows_a_warning_instead_of_simulating():
    app = open_app()
    # Fit the compound the car started on: only one compound would be used.
    start_caption = [c.value for c in app.caption if "starts on" in c.value][0]
    start = start_caption.split("starts on ")[1].split(" ")[0].upper()
    app.selectbox(key="tyre0").set_value(start).run()
    assert not app.exception, app.exception
    assert any("two different tyre compounds" in warning.value for warning in app.warning)


@needs_final_models
def test_two_stops_can_be_simulated():
    app = open_app()
    app.radio(key="stops").set_value(2).run()
    assert not app.exception, app.exception
    assert len(app.metric) >= 6 or app.warning
