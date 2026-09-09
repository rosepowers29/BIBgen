from pathlib import Path

import pytest
import numpy as np

from BIBgen.preprocessing import *

def test_whitening_matrices():
    data = np.random.rand(50, 5)

    W, W_inv, mu = whitening_matrices(data)
    transformed = (data - mu) @ W.T

    assert np.mean(transformed, axis=0) == pytest.approx(np.zeros(5), abs=1e-4)
    assert np.cov(transformed, rowvar=False) == pytest.approx(np.identity(5), abs=1e-4)

def test_Sphering():
    datas = [np.random.rand(50, 5) for _ in range(10)]
    spherings = [Sphering.from_data(d) for d in datas]

    sphered0 = spherings[0].transform(datas[0])
    assert np.mean(sphered0, axis=0) == pytest.approx(np.zeros(5), abs=1e-4)
    assert np.std(sphered0, axis=0) == pytest.approx(np.ones(5), abs=1e-4)

    avg_sphering = Sphering.from_spherings(spherings)
    data_concat = np.concatenate(datas, axis=0)
    sphered_concat = avg_sphering.transform(data_concat)
    assert np.mean(sphered_concat, axis=0) == pytest.approx(np.zeros(5), abs=0.1)
    assert np.std(sphered_concat, axis=0) == pytest.approx(np.ones(5), abs=0.1)


def test_diffuse():
    data = np.random.rand(500, 5)
    betas = np.ones(1000) * 0.15

    result = diffuse(data, betas)

    assert np.all(result[0] == data)
    assert np.mean(result[-1], axis=0) == pytest.approx(np.zeros(5), abs=0.15)

def test_quadratic_beta_schedule_matches_baseline():
    schedule = quadratic_beta_schedule(100, scale=3e-5)
    baseline = np.loadtxt(Path(__file__).parent.parent / "config" / "noise_schedule.csv")
    assert np.array_equal(schedule, baseline)

def test_cosine_beta_schedule_hits_target_alpha_bar():
    schedule = cosine_beta_schedule(100, target_alpha_bar_T=1e-5)
    alpha_bar_T = np.prod(1 - schedule)
    assert alpha_bar_T == pytest.approx(1e-5, rel=1e-6)
    assert np.all(schedule > 0)
    assert np.all(schedule <= 0.999)