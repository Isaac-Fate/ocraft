from pydantic import BaseModel


class BatchTrainingLogRecord(BaseModel):

    epoch: int
    batch_number: int
    num_batches: int
    train_loss: float

    def to_message(self) -> str:
        """Convert to a message to log.

        Returns
        -------
        str
            Message of this record.
        """

        return (
            "Training - "
            "Epoch: {epoch} - "
            "Batch: {batch_number}/{num_batches} - "
            "Training Loss: {train_loss}"
        ).format(
            epoch=self.epoch,
            batch_number=self.batch_number,
            num_batches=self.num_batches,
            train_loss=self.train_loss,
        )


class EpochTrainingLogRecord(BaseModel):

    epoch: int
    lr: float
    avg_train_loss: float

    def to_message(self) -> str:
        """Convert to a message to log.

        Returns
        -------
        str
            Message of this record.
        """

        return (
            "Training - "
            "Epoch: {epoch} - "
            "Learning Rate: {lr} - "
            "Average Training Loss: {avg_train_loss}"
        ).format(
            epoch=self.epoch,
            lr=self.lr,
            avg_train_loss=self.avg_train_loss,
        )


class ValidationLogRecord(BaseModel):

    epoch: int
    valid_loss: float

    def to_message(self) -> str:
        """Convert to a message to log.

        Returns
        -------
        str
            Message of this record.
        """

        return ("Validation - Epoch: {epoch} - Validation Loss: {valid_loss}").format(
            epoch=self.epoch,
            valid_loss=self.valid_loss,
        )
