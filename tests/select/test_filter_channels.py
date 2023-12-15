from ocf_datapipes.select import SelectChannels


def test_select_channels_nwp(nwp_datapipe):
    nwp_datapipe = SelectChannels(nwp_datapipe, channels=["t"])
    batch = next(iter(nwp_datapipe))
    assert "t" in batch["channel"].values


def test_select_channels_icon_eu(icon_eu_datapipe):
    nwp_datapipe = SelectChannels(icon_eu_datapipe, channels=["t"])
    batch = next(iter(nwp_datapipe))
    assert "t" in batch["channel"].values
