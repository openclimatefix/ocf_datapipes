from ocf_datapipes.select import FilterChannels


def test_filter_channels_nwp(nwp_datapipe):
    nwp_datapipe = FilterChannels(nwp_datapipe, channels=["t"])
    batch = next(iter(nwp_datapipe))
    assert "t" in batch["channel"].values


def test_filter_channels_icon_eu(icon_eu_datapipe):
    nwp_datapipe = FilterChannels(icon_eu_datapipe, channels=["t"])
    batch = next(iter(nwp_datapipe))
    assert "t" in batch["channel"].values
