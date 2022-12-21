""" Data module for pytorch lightning """
from typing import Callable, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from ocf_datapipes.batch.fake.fake_batch import fake_data_pipeline


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
        """

        super().__init__()
        self.data_pipeline = data_pipeline
        self.configuration = configuration
        self.fake_data = fake_data
        self.n_train_data = n_train_data
        self.n_val_data = n_val_data

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

    def train_dataloader(self):
        """Get the train dataloader"""

        train_dataset = self.data_pipeline(configuration=self.configuration)
        train_dataloader = DataLoader(train_dataset, batch_size=None)

        return train_dataloader

    def val_dataloader(self):
        """Get the validation dataloader"""

        validation_data_pipeline = self.data_pipeline(configuration=self.configuration)
        validation_dataloader = DataLoader(validation_data_pipeline, batch_size=None)

        return validation_dataloader

    def test_dataloader(self):
        """Get the test dataloader"""

        test_data_pipeline = self.data_pipeline(configuration=self.configuration)
        test_dataloader = DataLoader(test_data_pipeline, batch_size=None)

        return test_dataloader
