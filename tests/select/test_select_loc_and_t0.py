import numpy as np
from ocf_datapipes.select import LocationT0Picker


def test_select_loc_and_t0_return_all(gsp_datapipe):
    loctime_datapipe = LocationT0Picker(gsp_datapipe, return_all=True)
    locations = []
    times = []

    for loc, t0 in loctime_datapipe:
        locations.append(loc)
        times.append(t0)

    data = next(iter(gsp_datapipe))

    # Check that the correct number of values have been yielded
    assert len(times) == data.shape[0] * data.shape[1]
    assert len(locations) == data.shape[0] * data.shape[1]

    # All times have been returned
    assert (np.unique(times) == np.unique(data.time_utc)).all()

    # Check all the x-coords are used
    assert (np.unique([loc.x for loc in locations]) == np.unique(data.x_osgb)).all()
    # Check all the y-coords are used
    assert (np.unique([loc.y for loc in locations]) == np.unique(data.y_osgb)).all()
    # Check all the IDs are used
    assert (np.unique([loc.id for loc in locations]) == np.unique(data.gsp_id)).all()


def test_select_loc_and_t0_random(gsp_datapipe):
    loctime_datapipe = LocationT0Picker(gsp_datapipe, return_all=False)
    locations = []
    times = []

    for i, (loc, t0) in enumerate(loctime_datapipe):
        locations.append(loc)
        times.append(t0)
        # This version will keep yielding infinitely. So must break
        if i == 99:
            break

    data = next(iter(gsp_datapipe))

    # Check that the correct number of values have been yielded
    assert len(times) == 100
    assert len(locations) == 100

    def unique(x, label):
        unique_x, counts = np.unique(x, return_counts=True)
        assert len(unique_x) > 1, f"All {label} are the same"
        assert not (counts == 1).all(), f"{label} have not been returned in different amounts"

    unique(times, "times")
    unique([loc.y for loc in locations], "y")
    unique([loc.x for loc in locations], "y")
    unique([loc.id for loc in locations], "id")
