"""Tests for reinforcement learning time utils.

Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from absl.testing import absltest
import numpy as np

from smart_control.reinforcement_learning.utils.time_utils import time_from_sin_cos
from smart_control.reinforcement_learning.utils.time_utils import to_dow
from smart_control.reinforcement_learning.utils.time_utils import to_hod


class TestTimeFromSinCos(absltest.TestCase):

    def test_time_from_sin_cos_positive_cos(self):
        # Test case 1: sin(0) and cos(1)
        self.assertAlmostEqual(time_from_sin_cos(0, 1), 0)
        
        # Test case 2: sin(1) and cos(0)
        self.assertAlmostEqual(time_from_sin_cos(1, 0), np.pi/2)

        # Test case 3: sin(sqrt(2)/2) and cos(sqrt(2)/2)
        self.assertAlmostEqual(time_from_sin_cos(np.sqrt(2)/2, np.sqrt(2)/2), np.pi/4)

    def test_time_from_sin_cos_negative_cos(self):
        # Test case 4: sin(0) and cos(-1)
        self.assertAlmostEqual(time_from_sin_cos(0, -1), np.pi)

        # Test case 5: sin(1) and cos(0)
        self.assertAlmostEqual(time_from_sin_cos(1, -0.1), np.pi - np.arcsin(1))

    def test_time_from_sin_cos_negative_sin(self):
        # Test case 6: sin(-1) and cos(0)
        self.assertAlmostEqual(time_from_sin_cos(-1, 0), 3 * np.pi/2)

        # Test case 7: sin(-1) and cos(-1)
        self.assertAlmostEqual(time_from_sin_cos(-1, -1), 5*np.pi/4)

    def test_time_from_sin_cos_edge_cases(self):
        # Test case 8: sin(1) and cos(1)
        self.assertAlmostEqual(time_from_sin_cos(1, 1), np.arctan2(1,1))
        # Test case 9: sin(-1) and cos(-1)
        self.assertAlmostEqual(time_from_sin_cos(-1,-1), np.arctan2(-1,-1))
        
    def test_time_from_sin_cos_invalid_input(self):
        # Test case 10: sin(2) and cos(2) - invalid, should be between -1 and 1
         with self.assertRaises(ValueError):
             time_from_sin_cos(2,2)


class TestToDow(absltest.TestCase):
    
    def test_to_dow_valid_inputs(self):
        # Test case 1: Monday
        self.assertEqual(to_dow(0, 1), 0)

        # Test case 2: Tuesday
        self.assertEqual(to_dow(np.sin(2*np.pi/7), np.cos(2*np.pi/7)), 1)
        
        # Test case 3: Sunday
        self.assertEqual(to_dow(np.sin(6*np.pi/7), np.cos(6*np.pi/7)), 6)

    def test_to_dow_edge_cases(self):
        # Test case 4: Boundary case (near 0)
        self.assertEqual(to_dow(0,1), 0)

        # Test case 5: Boundary case (near 2*pi)
        self.assertEqual(to_dow(np.sin(2*np.pi-0.0001), np.cos(2*np.pi-0.0001)), 6)


class TestToHod(absltest.TestCase):
    
    def test_to_hod_valid_inputs(self):
        # Test case 1: 00:00
        self.assertEqual(to_hod(0, 1), 0)

        # Test case 2: 12:00
        self.assertEqual(to_hod(np.sin(np.pi), np.cos(np.pi)), 12)
        
        # Test case 3: 23:00
        self.assertEqual(to_hod(np.sin(23*2*np.pi/24), np.cos(23*2*np.pi/24)), 23)

    def test_to_hod_edge_cases(self):
         # Test case 4: Boundary case (near 0)
        self.assertEqual(to_hod(0,1), 0)

        # Test case 5: Boundary case (near 2*pi)
        self.assertEqual(to_hod(np.sin(2*np.pi-0.0001), np.cos(2*np.pi-0.0001)), 23)
        
if __name__ == '__main__':
    absltest.main()
