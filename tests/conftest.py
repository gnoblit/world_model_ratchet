import pytest
import os
import shutil

# Define the single, shared temporary directory for all tests at the module level
TEST_LOG_DIR = "temp_test_logs"

@pytest.fixture
def temp_log_dir():
    """
    A pytest fixture that creates a clean temporary log directory before a test,
    provides the path to the test function, and removes the directory after the test.
    
    This is the single source of truth for test-related file I/O.
    """
    # --- Setup Phase ---
    # Always start fresh by removing the directory if it exists from a failed previous run
    if os.path.exists(TEST_LOG_DIR):
        shutil.rmtree(TEST_LOG_DIR)
    
    # Create the directory for the current test to use
    os.makedirs(TEST_LOG_DIR, exist_ok=True)
    
    # --- Provide the resource to the test ---
    # The 'yield' statement passes the value of TEST_LOG_DIR to the test function
    # that requested this fixture. The test runs at this point.
    yield TEST_LOG_DIR
    
    # --- Teardown Phase ---
    # This code runs after the test has completed (whether it passed or failed).
    # It ensures we clean up after ourselves.
    if os.path.exists(TEST_LOG_DIR):
        shutil.rmtree(TEST_LOG_DIR)