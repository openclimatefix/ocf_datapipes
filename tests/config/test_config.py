"""Test config."""
import os
import tempfile
from datetime import datetime

import pytest
from pydantic import ValidationError

import ocf_datapipes
from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.config.model import PV, Configuration, set_git_commit
from ocf_datapipes.config.save import save_yaml_configuration


def test_default():
    """
    Test default pydantic class
    """

    _ = Configuration()


def test_yaml_load_on_premises(top_test_directory):
    """Test that yaml loading works for 'on_premises.yaml'"""

    filename = f"{top_test_directory}/data/configs/on_premises.yaml"

    config = load_yaml_configuration(filename)

    assert isinstance(config, Configuration)


def test_yaml_save():
    """
    Check a configuration can be saved to a yaml file
    """

    with tempfile.NamedTemporaryFile(suffix=".yaml") as fp:
        filename = fp.name

        # check that temp file cant be loaded
        with pytest.raises(TypeError):
            _ = load_yaml_configuration(filename)

        # save default config to file
        save_yaml_configuration(Configuration(), filename)

        # check the file can be loaded
        _ = load_yaml_configuration(filename)


def test_yaml_load_env(configuration_filename):
    """
    Check a configuration can be loaded with an env var
    """

    os.environ["PATH"] = "example_path"

    # check the file can be loaded
    config_load = load_yaml_configuration(configuration_filename)

    assert "example_path" in config_load.general.description


def test_extra_field():
    """
    Check a extra parameters in config causes error
    """

    configuration = Configuration()
    configuration_dict = configuration.dict()
    configuration_dict["extra_field"] = "extra_value"
    with pytest.raises(ValidationError):
        _ = Configuration(**configuration_dict)


def test_incorrect_time_resolution():
    """
    Check a a time resolution not divisible by 5 causes an error
    """

    configuration = Configuration()
    configuration.input_data = configuration.input_data.set_all_to_defaults()
    configuration.input_data.satellite.time_resolution_minutes = 27
    with pytest.raises(Exception):
        _ = Configuration(**configuration.dict())


def test_config_git(configuration_filename):
    """Test that git commit is working"""

    config = load_yaml_configuration(configuration_filename)

    config = set_git_commit(configuration=config)

    assert config.git is not None
    assert type(config.git.message) == str
    assert type(config.git.hash) == str
    assert type(config.git.committed_date) == datetime
