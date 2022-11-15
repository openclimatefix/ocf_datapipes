from ocf_datapipes.batch.fake.fake_batch import make_fake_batch
from ocf_datapipes.config.model import Configuration


def test_make_fake_batch():

    configuration = Configuration()
    configuration.input_data = configuration.input_data.set_all_to_defaults()

    _ = make_fake_batch(configuration=configuration)