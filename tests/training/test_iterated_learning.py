import pytest
from unittest.mock import MagicMock, call

from configs.base_config import get_base_config
from training.iterated_learning import IteratedLearningManager

def test_il_manager_initialization():
    """Tests that the manager and its underlying trainer can be initialized."""
    try:
        config = get_base_config()
        config.training.device = 'cpu'
        manager = IteratedLearningManager(config)
        assert manager is not None
        assert manager.trainer is not None
    except Exception as e:
        pytest.fail(f"IteratedLearningManager initialization failed: {e}")

def test_spawn_new_student_logic():
    """Tests that the spawn_new_student method correctly replaces the student and optimizer."""
    config = get_base_config()
    config.training.device = 'cpu'
    manager = IteratedLearningManager(config)
    
    # Get the original student and optimizer IDs to check they are replaced
    original_student_id = id(manager.trainer.actor_critic)
    original_optimizer_id = id(manager.trainer.action_optimizer)
    
    # Call the method to spawn a new student
    manager.spawn_new_student()
    
    # Assert that the objects have been replaced with new ones
    assert id(manager.trainer.actor_critic) != original_student_id
    assert id(manager.trainer.action_optimizer) != original_optimizer_id

def test_run_il_loop_orchestration():
    """
    Tests that the main IL loop calls the trainer's methods in the correct sequence
    and with the correct arguments, using a mock.
    """
    config = get_base_config()
    config.training.device = 'cpu'
    # Use a small number of generations for a quick test
    config.il.num_generations = 2
    
    manager = IteratedLearningManager(config)
    
    # --- The Mocking Magic ---
    # Replace the real, slow `train_for_steps` with a mock object
    # This mock will record all calls made to it.
    manager.trainer.train_for_steps = MagicMock()
    
    # Replace the spawn method as well, since we test it separately
    manager.spawn_new_student = MagicMock()
    
    # --- Run the Loop ---
    manager.run_il_loop()
    
    # --- Assertions ---
    # Check that the spawn method was called the correct number of times
    assert manager.spawn_new_student.call_count == config.il.num_generations

    # Now, check the sequence and arguments of the calls to train_for_steps
    train_calls = manager.trainer.train_for_steps.call_args_list
    
    # Expected sequence of calls:
    expected_calls = [
        # Generation 0: Warmup
        call(config.il.warmup_steps, teacher_is_frozen=False),
        # Generation 1: Distill -> Interact
        call(config.il.distill_steps, teacher_is_frozen=True),
        call(config.il.interact_steps, teacher_is_frozen=False),
        # Generation 2: Distill -> Interact
        call(config.il.distill_steps, teacher_is_frozen=True),
        call(config.il.interact_steps, teacher_is_frozen=False),
    ]
    
    # Assert that the mock was called with the expected sequence
    assert len(train_calls) == len(expected_calls), \
        f"Expected {len(expected_calls)} calls to train_for_steps, but got {len(train_calls)}"
        
    train_calls[0].assert_called_with(config.il.warmup_steps, teacher_is_frozen=False)
    
    # Check the generational calls
    assert train_calls[1] == expected_calls[1]
    assert train_calls[2] == expected_calls[2]
    assert train_calls[3] == expected_calls[3]
    assert train_calls[4] == expected_calls[4]