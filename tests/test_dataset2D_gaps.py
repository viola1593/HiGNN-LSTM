"""Checks that TWSA2DDataset only builds windows over consecutive months.

GRACE has missing months (2018-08, 2018-09). A window across such a gap would be
labelled as consecutive months but actually span more, so it must be dropped.
On contiguous data nothing changes.
"""
import numpy as np
import pytest

from synthetic_grid import make_ds

from datamodules.TWSA_dataset2D import TWSA2DDataset

INPUT_SEQ_LEN = 3
PRED_LEN = 2


def build(ds, delay, time_step=None):
    return TWSA2DDataset(
        ds,
        time_period=(ds.time.values[0], ds.time.values[-1]),
        target='twsa',
        delay=delay,
        input_seq_len=INPUT_SEQ_LEN,
        output_seq_len=PRED_LEN,
        time_step=time_step,
        input_vars=list(ds.data_vars.keys()),
    )


def months_of(ds):
    return ds.time.values.astype('datetime64[M]')


def drop_months(ds, *months):
    return ds.drop_sel(time=[np.datetime64(m, 'ns') for m in months])


def expected_samples(ds, delay):
    """Window index t for every input start whose whole window is present in the data."""
    months = months_of(ds)
    present = set(months.tolist())
    window = INPUT_SEQ_LEN + delay + PRED_LEN
    return [s + INPUT_SEQ_LEN for s, m0 in enumerate(months)
            if all((m0 + np.timedelta64(k, 'M')).item() in present for k in range(window))]


@pytest.mark.parametrize('delay', [0, 2])
def test_contiguous_data_is_unchanged(delay):
    ds = make_ds()
    T = len(ds.time)
    assert build(ds, delay).samples == list(range(INPUT_SEQ_LEN, T - delay - PRED_LEN + 1))


@pytest.mark.parametrize('delay', [0, 2])
def test_gap_removes_exactly_the_windows_that_span_it(delay):
    ds = drop_months(make_ds(), '2001-06')
    months = months_of(ds)
    samples = build(ds, delay).samples
    window = INPUT_SEQ_LEN + delay + PRED_LEN

    for t in samples:
        span = months[t + delay + PRED_LEN - 1] - months[t - INPUT_SEQ_LEN]
        assert span == np.timedelta64(window - 1, 'M')
    assert samples == expected_samples(ds, delay)
    # sanity: the gap really removed something
    assert len(samples) < len(range(INPUT_SEQ_LEN, len(ds.time) - delay - PRED_LEN + 1))


def test_two_adjacent_missing_months_near_the_start():
    ds = drop_months(make_ds(), '2000-03', '2000-04')
    months = months_of(ds)
    samples = build(ds, delay=0).samples

    first_input_month = months[samples[0] - INPUT_SEQ_LEN]
    assert first_input_month == np.datetime64('2000-05', 'M')


def test_time_step_is_not_filtered():
    ds = drop_months(make_ds(), '2000-05')
    # input positions 1..3 are 2000-02, 03, 04 and the target 2000-06, 07: spans the gap
    assert build(ds, delay=0, time_step=4).samples == [4]
