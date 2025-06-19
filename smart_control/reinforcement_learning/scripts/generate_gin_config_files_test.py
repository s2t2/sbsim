import os
import shutil
import tempfile
import unittest
import re
from smart_control.reinforcement_learning.scripts.generate_gin_config_files import generate_configs, read_config_file, modify_config

class TestGenerateGinConfigFiles(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.base_gin_filename = "base_test_config.gin"
        self.base_gin_filepath = os.path.join(self.test_dir, self.base_gin_filename)
        self.output_dir = os.path.join(self.test_dir, "output_configs")

        # Content for the base gin file
        # Note: The parameters here should match what the script expects to modify,
        # e.g., 'time_step_sec', 'num_days_in_episode', 'start_timestamp'
        self.base_gin_content = """
# Base configuration for testing
time_step_sec = 300
num_days_in_episode = 1
start_timestamp = '2023-07-06 07:00:00+00:00'

# Some other fixed parameter
fixed_parameter = "test_value"

# Example of a parameter that might be imported
# SomeOtherClass.parameter = 10
# For the purpose of this test, we'll focus on direct assignments.
"""
        with open(self.base_gin_filepath, "w", encoding='utf-8') as f:
            f.write(self.base_gin_content)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_read_config_file(self):
        content = read_config_file(self.base_gin_filepath)
        self.assertEqual(content, self.base_gin_content)

    def test_modify_config_direct_assignment(self):
        modified_content = modify_config(self.base_gin_content, "time_step_sec", "600")
        self.assertIn("time_step_sec =600", modified_content) # Corrected: no space after =
        modified_content_str = modify_config(self.base_gin_content, "fixed_parameter", '"new_test_value"')
        self.assertIn('fixed_parameter ="new_test_value"', modified_content_str) # Corrected: no space after =

        # Test with parameter not present - should log warning (cannot check log here) and return original
        original_content = "param_x = 10"
        modified_content_no_change = modify_config(original_content, "non_existent_param", "20")
        self.assertEqual(original_content, modified_content_no_change)


    def test_generate_configs_creates_files_and_substitutes_params(self):
        param_grid_for_test = {
            'num_days_in_episode': ['7', '14'],  # Two values for num_days
            'time_step_sec': ['900']             # One value for time_step_sec
        }

        generate_configs(self.base_gin_filepath, self.output_dir, param_grid_for_test)

        # Expected filenames: config_numdaysinepisode-7_timestepsec-900.gin, config_numdaysinepisode-14_timestepsec-900.gin
        expected_file1_name = "config_numdaysinepisode-7_timestepsec-900.gin"
        expected_file2_name = "config_numdaysinepisode-14_timestepsec-900.gin"

        output_file1_path = os.path.join(self.output_dir, expected_file1_name)
        output_file2_path = os.path.join(self.output_dir, expected_file2_name)

        self.assertTrue(os.path.exists(output_file1_path), f"{expected_file1_name} was not generated.")
        self.assertTrue(os.path.exists(output_file2_path), f"{expected_file2_name} was not generated.")

        # Check content of the first generated file
        with open(output_file1_path, "r", encoding='utf-8') as f:
            content1 = f.read()

        # Perform a more robust check using regex to account for potential whitespace variations
        self.assertTrue(re.search(r"num_days_in_episode\s*=\s*7", content1), "num_days_in_episode was not correctly substituted in file1.")
        self.assertTrue(re.search(r"time_step_sec\s*=\s*900", content1), "time_step_sec was not correctly substituted in file1.") # Script output is param =value
        self.assertTrue(re.search(r"start_timestamp\s*=\s*'2023-07-06 07:00:00\+00:00'", content1), "start_timestamp should remain from base config in file1.")
        self.assertTrue(re.search(r"fixed_parameter\s*=\s*\"test_value\"", content1), "fixed_parameter should remain from base config in file1.")

        # Check content of the second generated file
        with open(output_file2_path, "r", encoding='utf-8') as f:
            content2 = f.read()
        self.assertTrue(re.search(r"num_days_in_episode\s*=\s*14", content2), "num_days_in_episode was not correctly substituted in file2.")
        self.assertTrue(re.search(r"time_step_sec\s*=\s*900", content2), "time_step_sec was not correctly substituted in file2.") # Script output is param =value

    def test_generate_configs_no_variation(self):
        # Test with no parameters to vary, should just copy the base config with a default-like name
        param_grid_empty = {}
        generate_configs(self.base_gin_filepath, self.output_dir, param_grid_empty)

        expected_file_name = "config_.gin" # Based on current naming logic when filename_parts is empty
        output_file_path = os.path.join(self.output_dir, expected_file_name)
        self.assertTrue(os.path.exists(output_file_path), "File for no variation not generated correctly.")

        with open(output_file_path, "r", encoding='utf-8') as f:
            content = f.read()
        self.assertIn("num_days_in_episode = 1", content) # Original formatting preserved
        self.assertIn("time_step_sec = 300", content)   # Original formatting preserved


if __name__ == "__main__":
    unittest.main()
