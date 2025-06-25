import os
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """A simple wrapper for torch.utils.tensorboard.SummaryWriter."""

    def __init__(self, log_dir: str):
        """
        Initializes the Logger.

        Args:
            log_dir (str): The directory where TensorBoard logs will be saved.
        """
        # Create the log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"Logger initialized. Logging to: {log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        """
        Logs a scalar value to TensorBoard.

        Args:
            tag (str): The name of the scalar (e.g., 'train/reward').
            value (float): The value to log.
            step (int): The global step count to associate with the value.
        """
        self.writer.add_scalar(tag, value, step)

    def close(self):
        """Closes the SummaryWriter."""
        self.writer.close()