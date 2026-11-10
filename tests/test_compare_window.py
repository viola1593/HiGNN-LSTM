"""compare_models picks the Li et al. comparison window by date, not by a hardcoded sample index."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import compare_models


def _grace_first_target_times():
    # first predicted month of each GRACE_era5 sample: the GRACE period and the
    # GRACE-FO period, separated by the gap; mid-month timestamps like the dataset
    months = np.concatenate([np.arange(np.datetime64('2009-03'), np.datetime64('2010-03')),
                             np.arange(np.datetime64('2019-10'), np.datetime64('2023-08'))])
    return months.astype('datetime64[D]') + np.timedelta64(15, 'D')


def test_window_keeps_samples_predicting_nov2019_to_nov2023():
    keep, inits = compare_models.select_comparison_window(_grace_first_target_times())

    assert keep.sum() == 44
    assert np.flatnonzero(keep)[0] == 13  # Nov 2019 after the 12 GRACE samples and Oct 2019
    kept = _grace_first_target_times()[keep].astype('datetime64[M]')
    assert kept[0] == np.datetime64('2019-11')
    assert kept[-1] + 5 == np.datetime64('2023-11')  # last lead lands on Nov 2023


def test_init_is_the_month_before_the_first_prediction():
    keep, inits = compare_models.select_comparison_window(_grace_first_target_times())

    assert len(inits) == keep.sum()
    assert inits[0] == '2019-10'
    assert inits[-1] == '2023-05'
