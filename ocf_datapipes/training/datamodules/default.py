""" Data module for pytorch lightning """
from typing import Callable, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import fsspec.asyn


from ocf_datapipes.batch.fake.fake_batch import fake_data_pipeline

def set_fsspec_for_multiprocess() -> None:
    """
    Clear reference to the loop and thread.
    This is a nasty hack that was suggested but NOT recommended by the lead fsspec developer!
    This appears necessary otherwise gcsfs hangs when used after forking multiple worker processes.
    Only required for fsspec >= 0.9.0
    See:
    - https://github.com/fsspec/gcsfs/issues/379#issuecomment-839929801
    - https://github.com/fsspec/filesystem_spec/pull/963#issuecomment-1131709948
    TODO: Try deleting this two lines to make sure this is still relevant.
    """
    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None

def worker_init_fn(worker_id):
    """Configures each dataset worker process.
    1. Get fsspec ready for multi process
    2. To call NowcastingDataset.per_worker_init().
    """
    # fix for fsspec when using multprocess
    set_fsspec_for_multiprocess()
    
    
class DataModule(LightningDataModule):
    """
    Example of LightningDataModule using ocf_datapipes
    """

    def __init__(
        self,
        data_pipeline: Union[str, Callable],
        configuration,
        fake_data,
        n_train_data,
        n_val_data,
        num_workers=0,
        prefetch_factor=2,
    ):
        """
        Set up datamodule

        Args:
            data_pipeline: the pipeline function to be called,
                can also be the name of the data pipeline
            configuration: the configuration file
            fake_data: option to load fake data or not
            n_train_data: TODO
            n_val_data: TODO
            num_workers: number of workers of loading data
            prefetch_factor: number of batches loaded in advance by each worker
        """

        super().__init__()
        self.data_pipeline = data_pipeline
        self.configuration = configuration
        self.fake_data = fake_data
        self.n_train_data = n_train_data
        self.n_val_data = n_val_data
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        if self.fake_data:
            self.data_pipeline = fake_data_pipeline
        elif isinstance(self.data_pipeline, str):
            if self.data_pipeline == "pv_satellite_nwp":
                from ocf_datapipes.training.pv_satellite_nwp import pv_nwp_satellite_data_pipeline

                self.data_pipeline = pv_nwp_satellite_data_pipeline

            if self.data_pipeline == "gsp_pv_satellite_nwp":
                from ocf_datapipes.training.gsp_pv_satellite_nwp import (
                    gsp_pv_nwp_satellite_data_pipeline,
                )

                self.data_pipeline = gsp_pv_nwp_satellite_data_pipeline

        self.dataloader_config = dict(
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            worker_init_fn=worker_init_fn,
            # Persistent_workers option needs num_workers > 0
            persistent_workers=self.num_workers > 0,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )

    def train_dataloader(self):
        """Get the train dataloader"""

        train_datapipe = self.data_pipeline(configuration=self.configuration).set_length(
            self.n_train_data
        )
        train_dataloader = DataLoader(train_datapipe, **self.dataloader_config)
        return train_dataloader

    def val_dataloader(self):
        """Get the validation dataloader"""

        validation_datapipe = self.data_pipeline(configuration=self.configuration).set_length(
            self.n_val_data
        )
        validation_dataloader = DataLoader(validation_datapipe, **self.dataloader_config)

        return validation_dataloader

    def test_dataloader(self):
        """Get the test dataloader"""

        test_datapipe = self.data_pipeline(configuration=self.configuration)
        test_dataloader = DataLoader(test_datapipe, **self.dataloader_config)

        return test_dataloader
