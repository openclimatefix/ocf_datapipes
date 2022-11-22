from ocf_datapipes.batch.fake.fake_batch import make_fake_batch, fake_data_pipeline
from ocf_datapipes.config.model import Configuration

from torch.utils.data import DataLoader


def test_make_fake_batch():

    configuration = Configuration()
    configuration.input_data = configuration.input_data.set_all_to_defaults()

    _ = make_fake_batch(configuration=configuration)


def test_make_fake_batch_only_pv():

    configuration = Configuration()
    input_data = configuration.input_data.set_all_to_defaults()
    configuration.input_data.pv = input_data.pv

    _ = make_fake_batch(configuration=configuration)


def test_make_fake_batch_only_pv_nwp():

    configuration = Configuration()
    input_data = configuration.input_data.set_all_to_defaults()
    configuration.input_data.pv = input_data.pv
    configuration.input_data.nwp = input_data.nwp

    _ = make_fake_batch(configuration=configuration)


def test_make_fake_batch_only_pv_nwp_gsp():

    configuration = Configuration()
    input_data = configuration.input_data.set_all_to_defaults()
    configuration.input_data.pv = input_data.pv
    configuration.input_data.nwp = input_data.nwp
    configuration.input_data.gsp = input_data.gsp

    _ = make_fake_batch(configuration=configuration)


def test_make_fake_batch_only_nwp_gsp():

    configuration = Configuration()
    input_data = configuration.input_data.set_all_to_defaults()
    configuration.input_data.nwp = input_data.nwp
    configuration.input_data.gsp = input_data.gsp

    _ = make_fake_batch(configuration=configuration)


def test_fake_data_pipeline():

    configuration = Configuration()
    input_data = configuration.input_data.set_all_to_defaults()
    configuration.input_data.nwp = input_data.nwp
    configuration.input_data.gsp = input_data.gsp

    data_pipeline = fake_data_pipeline(configuration=configuration)
    train_dataloader = DataLoader(dataset=data_pipeline, batch_size=None)
    train_dataloader_iter = iter(train_dataloader)
    batch = next(train_dataloader_iter)
