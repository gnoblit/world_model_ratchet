import pytest
import torch
from unittest.mock import MagicMock, call, patch

from configs.base_config import get_base_config
from training.iterated_learning import IteratedLearningManager

def test_il_manager_initialization():
    """Tests that the manager and its underlying trainer can be initialized."""
    try:
        config = get_base_config()
        config.training.device = 'cpu'
        config.run_name = None
        manager = IteratedLearningManager(config)
        assert manager is not None
        assert manager.trainer is not None
    except Exception as e:
        pytest.fail(f"IteratedLearningManager initialization failed: {e}")

def test_spawn_new_student_logic():
    """Tests that the spawn_new_student method correctly replaces the student and optimizer."""
    # We use a patch to prevent the real, slow torch.compile from running.
    # We just want to check if it was called.
    with patch('torch.compile') as mock_compile:
        config = get_base_config()
        config.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.run_name = None

        # --- Test Case 1: Compilation is OFF ---
        config.training.use_torch_compile = False
        manager = IteratedLearningManager(config)
        original_student_id = id(manager.trainer.actor_critic)
        original_optimizer_id = id(manager.trainer.action_optimizer)
        manager.spawn_new_student()
        assert id(manager.trainer.actor_critic) != original_student_id
        assert id(manager.trainer.action_optimizer) != original_optimizer_id
        mock_compile.assert_not_called() # Should not be called when disabled

        mock_compile.reset_mock()

        # --- Test Case 2: Compilation is ON ---
        config.training.use_torch_compile = True
        manager = IteratedLearningManager(config)
        original_student_id = id(manager.trainer.actor_critic)
        original_optimizer_id = id(manager.trainer.action_optimizer)
        manager.spawn_new_student()
        assert id(manager.trainer.actor_critic) != original_student_id
        assert id(manager.trainer.action_optimizer) != original_optimizer_id
        if config.training.device == 'cuda':
            mock_compile.assert_called_once() # Should be called when enabled on CUDA
        else:
            mock_compile.assert_not_called() # Should not be called on CPU

def test_run_il_loop_orchestration():
    """
    Tests that the main IL loop calls the trainer's methods in the correct sequence
    and with the correct arguments, using a mock.
    """
    config = get_base_config()
    config.training.device = 'cpu'
    # Use a small number of generations for a quick test
    config.il.num_generations = 2
    config.run_name = None
    
    manager = IteratedLearningManager(config)
    
    # --- The Mocking Magic ---
    # Replace the real, slow `train_for_steps` with a mock object
    manager.trainer.train_for_steps = MagicMock()
    # Mock the teacher refinement method
    manager.trainer.train_from_buffer = MagicMock()
        # Mock the model saving method
    manager.trainer.save_models = MagicMock()
    # Mock the replay buffer's clear method to check it's called
    manager.trainer.replay_buffer.clear = MagicMock()
    
    # Replace the spawn method as well, since we test it separately
    manager.spawn_new_student = MagicMock()
    
    # --- Run the Loop ---
    manager.run_il_loop()

    # --- Assertions ---
    # Check that the spawn method was called the correct number of times
    assert manager.spawn_new_student.call_count == config.il.num_generations

    # 1. Check the initial warmup call
    manager.trainer.train_for_steps.assert_any_call(config.il.warmup_env_steps, teacher_is_frozen=False)

    # 2. Check the student training calls within the loop
    student_training_call = call(config.il.student_env_steps, teacher_is_frozen=True)
    assert manager.trainer.train_for_steps.call_args_list.count(student_training_call) == config.il.num_generations

    # 3. Check the total number of calls to train_for_steps (warmup + generations)
    assert manager.trainer.train_for_steps.call_count == 1 + config.il.num_generations

    # 4. Check that the buffer is cleared. It's called once before the loop
    #    and then once per generation inside the loop.
    expected_clear_calls = 1 + config.il.num_generations
    assert manager.trainer.replay_buffer.clear.call_count == expected_clear_calls

    # 5. Check that the teacher is refined from the buffer in each generation
    manager.trainer.train_from_buffer.assert_called_with(num_updates=config.il.teacher_grad_updates)
    assert manager.trainer.train_from_buffer.call_count == config.il.num_generations

    # 6. Check that models are saved after each generation AND at the very end.
    # The total number of saves should be one per generation, plus one final save from trainer.close().
    assert manager.trainer.save_models.call_count == config.il.num_generations + 1
    # Check the intermediate saves
    for i in range(1, config.il.num_generations + 1):
        manager.trainer.save_models.assert_any_call(suffix=f"_gen_{i}")
    # Check the final save
    manager.trainer.save_models.assert_any_call(suffix="_final")