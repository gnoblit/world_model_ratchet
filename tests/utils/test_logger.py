import os
import pytest
from utils.logger import Logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# The 'temp_log_dir' fixture is automatically available from tests/conftest.py
# We do not need to define or import it here.

def test_logger_initialization(temp_log_dir): # Use the new fixture name
    """Tests that the Logger creates its directory."""
    # The fixture creates the dir, so we check that it's initially empty
    assert len(os.listdir(temp_log_dir)) == 0
    
    logger = Logger(log_dir=temp_log_dir) # Use the path provided by the fixture
    
    # Assert that the logger created its own event file inside the directory
    assert len(os.listdir(temp_log_dir)) > 0
    assert os.path.isdir(temp_log_dir)
    logger.close()

def test_log_scalar(temp_log_dir): # Use the new fixture name
    """Tests that a scalar value is correctly logged and can be read back."""
    logger = Logger(log_dir=temp_log_dir) # Use the path provided by the fixture
    
    tag_name = "test/scalar_value"
    scalar_value = 0.123
    step = 100
    
    logger.log_scalar(tag_name, scalar_value, step)
    logger.close()

    # --- Verification ---
    event_files = os.listdir(temp_log_dir)
    assert len(event_files) == 1
    event_file_path = os.path.join(temp_log_dir, event_files[0])
    
    accumulator = EventAccumulator(event_file_path)
    accumulator.Reload()
    
    assert tag_name in accumulator.Tags()['scalars']
    
    events = accumulator.Scalars(tag_name)
    assert len(events) == 1
    logged_event = events[0]
    
    assert logged_event.step == step
    assert logged_event.value == pytest.approx(scalar_value, rel=1e-5)