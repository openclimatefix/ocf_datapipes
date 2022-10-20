"""Datapipes to add Sun position to NumpyBatch"""

from typing import Optional

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.config.model import Configuration


@functional_datapipe("add_length")
class AddLengthIterDataPipe(IterDataPipe):
    """Adds length to the datapipe"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        length: Optional[int] = None,
        configuration: Optional[Configuration] = None,
        train_validation_test: Optional[str] = "train",
    ):
        """
        Adds length to the data pipe. This is useful when training a model.

        Either 'length' is set or, 'configuration' and 'train_validation_test' is set

        Args:
            source_datapipe: Datapipe of NumpyBatch
            length: Length of datapipe
            configuration: dataset configuration
            train_validation_test: either 'train', 'validation', or 'test'
        """
        self.source_datapipe = source_datapipe
        self.length = length
        self.configuration = configuration
        self.train_validation_test = train_validation_test

        if self.length is None:
            assert self.configuration is not None
            assert self.train_validation_test in ["train", "validation", "test"]

            if self.train_validation_test == "train":
                self.length = self.configuration.process.n_train_batches
            elif self.train_validation_test == "validation":
                self.length = self.configuration.process.n_validation_batches
            else:
                self.length = self.configuration.process.n_test_batches

    def __len__(self):
        return self.length

    def __iter__(self):
        for np_batch in self.source_datapipe:
            yield np_batch
