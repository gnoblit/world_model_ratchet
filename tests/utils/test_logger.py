import os
import shutil
import pytest
from utils.logger import Logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Define a temporary directory for test logs
TEST_LOG_DIR = "test_logs"

@pytest.fixture
def cleanup_logs():
    """A pytest fixture to automatically clean up the log directory after a test."""
    # Code that runs before your test
    if os.path.exists(TEST_LOG_DIR):
        shutil.rmtree(TEST_LOG_DIR)
    yield
    # Code that runs after your test
    if os.path.exists(TEST_LOG_DIR):
        shutil.rmtree(TEST_LOG_DIR)

def test_logger_initialization(cleanup_logs):
    """Tests that the Logger creates its directory."""
    assert not os.path.exists(TEST_LOG_DIR)
    logger = Logger(log_dir=TEST_LOG_DIR)
    assert os.path.isdir(TEST_LOG_DIR)
    logger.close()

def test_log_scalar(cleanup_logs):
    """Tests that a scalar value is correctly logged and can be read back."""
    logger = Logger(log_dir=TEST_LOG_DIR)
    
    tag_name = "test/scalar_value"
    scalar_value = 0.123
    step = 100
    
    logger.log_scalar(tag_name, scalar_value, step)
    logger.close()

    # --- Verification ---
    # Use TensorBoard's EventAccumulator to read the log file and verify its contents.
    # This is a robust way to confirm the log was written correctly.
    
    # Get the single event file created by the writer
    event_files = os.listdir(TEST_LOG_DIR)
    assert len(event_files) == 1
    event_file_path = os.path.join(TEST_LOG_DIR, event_files[0])
    
    # Load the events
    accumulator = EventAccumulator(event_file_path)
    accumulator.Reload()
    
    # Check that the tag exists
    assert tag_name in accumulator.Tags()['scalars']
    
    # Check the logged event
    events = accumulator.Scalars(tag_name)
    assert len(events) == 1
    logged_event = events[0]
    
    assert logged_event.step == step
    # Use pytest.approx for floating point comparison
    assert logged_event.value == pytest.approx(scalar_value, rel=1e-5)