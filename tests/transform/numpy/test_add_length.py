import pytest


def test_add_length(sat_datapipe):
    sat_datapipe = sat_datapipe.add_length(length=10)
    assert len(sat_datapipe) == 10


def test_add_length_configuration(sat_datapipe, configuration):
    sat_datapipe = sat_datapipe.add_length(configuration=configuration, train_validation_test='train')
    assert len(sat_datapipe) == 2

    sat_datapipe = sat_datapipe.add_length(configuration=configuration, train_validation_test='validation')
    assert len(sat_datapipe) == 0

    sat_datapipe = sat_datapipe.add_length(configuration=configuration, train_validation_test='test')
    assert len(sat_datapipe) == 0


def test_add_length_configuration_error(sat_datapipe, configuration):
    with pytest.raises(Exception):
        sat_datapipe.add_length(configuration=configuration, train_validation_test='string')








