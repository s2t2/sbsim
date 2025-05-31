import unittest
from unittest import mock
import requests # Keep for requests.exceptions.RequestException
import shutil
import os
import numpy as np # Needed for creating mock numpy array
import json # Needed for json.load
import pickle # Needed for pickle.load

# Assuming SmartBuildingsDataset is in a way that it can be imported.
try:
    from .dataset import SmartBuildingsDataset
except ImportError:
    from smart_control.dataset.dataset import SmartBuildingsDataset


class TestSmartBuildingsDataset(unittest.TestCase):

    @mock.patch('smart_control.dataset.dataset.SmartBuildingsDataset.download')
    def test_init_calls_download_by_default(self, mock_download):
        """Test that SmartBuildingsDataset.__init__ calls download by default."""
        SmartBuildingsDataset()
        mock_download.assert_called_once()

    @mock.patch('smart_control.dataset.dataset.SmartBuildingsDataset.download')
    def test_init_does_not_call_download_when_false(self, mock_download):
        """Test that SmartBuildingsDataset.__init__ does not call download when download=False."""
        SmartBuildingsDataset(download=False)
        mock_download.assert_not_called()

    @mock.patch('smart_control.dataset.dataset.shutil.unpack_archive')
    @mock.patch('smart_control.dataset.dataset.requests.get')
    @mock.patch('builtins.open', new_callable=mock.mock_open)
    @mock.patch('smart_control.dataset.dataset.print') # Mock print for "Downloading data..."
    def test_download_success(self, mock_print, mock_open_builtin, mock_requests_get, mock_shutil_unpack_archive):
        """Test a successful download operation."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.__enter__ = mock.Mock(return_value=mock_response)
        mock_response.__exit__ = mock.Mock(return_value=None)
        mock_response.iter_content = mock.Mock(return_value=[b"chunk1", b"chunk2"])
        mock_response.raise_for_status = mock.Mock()
        mock_requests_get.return_value = mock_response

        dataset = SmartBuildingsDataset(download=False)
        dataset.download()

        mock_print.assert_any_call("Downloading data...")
        hardcoded_url = "https://storage.googleapis.com/gresearch/smart_buildings_dataset/tabular_data/sb1.zip"
        mock_requests_get.assert_called_once_with(hardcoded_url, stream=True)

        expected_zip_path = "sb1.zip"
        mock_open_builtin.assert_called_once_with(expected_zip_path, 'wb')

        handle = mock_open_builtin()
        handle.write.assert_any_call(b"chunk1")
        handle.write.assert_any_call(b"chunk2")

        mock_shutil_unpack_archive.assert_called_once_with(expected_zip_path, "sb1/")


    @mock.patch('smart_control.dataset.dataset.shutil.unpack_archive')
    @mock.patch('smart_control.dataset.dataset.requests.get')
    @mock.patch('smart_control.dataset.dataset.print')
    def test_download_request_exception_on_get(self, mock_print, mock_requests_get, mock_shutil_unpack_archive):
        """Test download behavior when requests.get raises a RequestException."""
        mock_requests_get.side_effect = requests.exceptions.RequestException("Test connection error")
        dataset = SmartBuildingsDataset(download=False)
        with self.assertRaisesRegex(requests.exceptions.RequestException, "Test connection error"):
            dataset.download()
        mock_print.assert_any_call("Downloading data...")
        hardcoded_url = "https://storage.googleapis.com/gresearch/smart_buildings_dataset/tabular_data/sb1.zip"
        mock_requests_get.assert_called_once_with(hardcoded_url, stream=True)
        mock_shutil_unpack_archive.assert_not_called()

    @mock.patch('smart_control.dataset.dataset.shutil.unpack_archive')
    @mock.patch('smart_control.dataset.dataset.requests.get')
    @mock.patch('smart_control.dataset.dataset.print')
    def test_download_http_error(self, mock_print, mock_requests_get, mock_shutil_unpack_archive):
        """Test download behavior when response.raise_for_status() raises an HTTPError."""
        mock_response = mock.Mock()
        mock_response.status_code = 404
        mock_response.__enter__ = mock.Mock(return_value=mock_response)
        mock_response.__exit__ = mock.Mock(return_value=None)
        mock_response.iter_content = mock.Mock(return_value=[b"chunk1"])
        mock_response.raise_for_status = mock.Mock(side_effect=requests.exceptions.HTTPError("404 Client Error"))
        mock_requests_get.return_value = mock_response
        dataset = SmartBuildingsDataset(download=False)
        with self.assertRaisesRegex(requests.exceptions.HTTPError, "404 Client Error"):
            dataset.download()
        mock_print.assert_any_call("Downloading data...")
        hardcoded_url = "https://storage.googleapis.com/gresearch/smart_buildings_dataset/tabular_data/sb1.zip"
        mock_requests_get.assert_called_once_with(hardcoded_url, stream=True)
        mock_shutil_unpack_archive.assert_not_called()

    @mock.patch('smart_control.dataset.dataset.json.load')
    @mock.patch('builtins.open', new_callable=mock.mock_open)
    @mock.patch('smart_control.dataset.dataset.np.load')
    def test_get_floorplan_success(self, mock_np_load, mock_open_builtin, mock_json_load):
        """Test successful retrieval of floorplan and device layout map."""
        mock_floorplan_array = np.array([[1, 0], [0, 1]])
        mock_device_layout_map = {'device1': [0,0]}
        mock_np_load.return_value = mock_floorplan_array
        mock_json_load.return_value = mock_device_layout_map
        mock_file_handle = mock_open_builtin.return_value
        mock_file_handle.__enter__.return_value = mock_file_handle
        dataset = SmartBuildingsDataset(download=False)
        building_name = 'sb1'
        floorplan, device_layout_map = dataset.get_floorplan(building_name)
        expected_floorplan_path = f"./{building_name}/tabular/floorplan.npy"
        expected_json_path = f"./{building_name}/tabular/device_layout_map.json"
        mock_np_load.assert_called_once_with(expected_floorplan_path)
        mock_open_builtin.assert_called_once_with(expected_json_path)
        mock_json_load.assert_called_once_with(mock_file_handle)
        self.assertTrue(np.array_equal(floorplan, mock_floorplan_array))
        self.assertEqual(device_layout_map, mock_device_layout_map)

    def test_get_floorplan_invalid_building(self):
        """Test get_floorplan with an invalid building name."""
        dataset = SmartBuildingsDataset(download=False)
        with self.assertRaisesRegex(ValueError, "invalid building"):
            dataset.get_floorplan('invalid_building_name')

    @mock.patch('smart_control.dataset.dataset.pickle.load')
    @mock.patch('builtins.open', new_callable=mock.mock_open)
    @mock.patch('smart_control.dataset.dataset.np.load')
    def test_get_building_data_success(self, mock_np_load, mock_open_builtin, mock_pickle_load):
        """Test successful retrieval of building data and metadata."""
        mock_data_array = {'X': np.array([1, 2, 3])}
        mock_metadata_dict = {'key': 'value', 'device_infos': {}, 'zone_infos': {}} # Ensure keys exist

        mock_np_load.return_value = mock_data_array
        mock_pickle_load.return_value = mock_metadata_dict

        mock_file_handle = mock_open_builtin.return_value
        mock_file_handle.__enter__.return_value = mock_file_handle

        dataset = SmartBuildingsDataset(download=False)
        building, partition = 'sb1', '2022_a'
        data, metadata = dataset.get_building_data(building, partition)

        expected_data_path = f"./{building}/tabular/{building}/{partition}/data.npy.npz"
        expected_metadata_path = f"./{building}/tabular/{building}/{partition}/metadata.pickle"

        mock_np_load.assert_called_once_with(expected_data_path)
        # pickle.load is called once for metadata.pickle
        mock_open_builtin.assert_called_once_with(expected_metadata_path, "rb")
        mock_pickle_load.assert_called_once_with(mock_file_handle)

        self.assertEqual(data, mock_data_array)
        self.assertEqual(metadata, mock_metadata_dict)


    @mock.patch('smart_control.dataset.dataset.pickle.load')
    @mock.patch('builtins.open', new_callable=mock.mock_open)
    @mock.patch('smart_control.dataset.dataset.np.load')
    def test_get_building_data_loads_additional_metadata(self, mock_np_load, mock_open_builtin, mock_pickle_load):
        """Test that additional device_infos and zone_infos are loaded if not in main metadata."""
        mock_data_array = {'X': np.array([4, 5, 6])}
        initial_mock_metadata = {'description': 'Initial metadata'} # No device_infos or zone_infos
        mock_device_infos = {'dev1': 'info1'}
        mock_zone_infos = {'zoneA': 'infoA'}

        # Configure pickle.load to return different values based on the file being opened
        # This requires knowing the file paths used by open
        building, partition = 'sb1', '2022_a'
        path_metadata = f"./{building}/tabular/{building}/{partition}/metadata.pickle"
        path_device_info = f"./{building}/tabular/device_info_dicts.pickle"
        path_zone_info = f"./{building}/tabular/zone_info_dicts.pickle"

        # This side_effect function will be called each time mock_pickle_load is invoked
        def pickle_load_side_effect(file_handle):
            # The file_handle here is the mock object returned by mock_open_builtin()
            # We need to see which file mock_open_builtin was called with
            current_open_call_path = mock_open_builtin.call_args.args[0]
            if current_open_call_path == path_metadata:
                return initial_mock_metadata
            elif current_open_call_path == path_device_info:
                return mock_device_infos
            elif current_open_call_path == path_zone_info:
                return mock_zone_infos
            return None # Should not happen in this test

        mock_np_load.return_value = mock_data_array
        mock_pickle_load.side_effect = pickle_load_side_effect

        # This side_effect for open is to return distinct file handles if needed, though not strictly
        # necessary if pickle_load_side_effect checks mock_open_builtin.call_args
        mock_handles = {
            path_metadata: mock.MagicMock(),
            path_device_info: mock.MagicMock(),
            path_zone_info: mock.MagicMock()
        }
        def open_side_effect(path, mode):
            handle = mock_handles[path]
            handle.__enter__.return_value = handle # Make it a context manager
            handle.__exit__.return_value = None
            return handle

        mock_open_builtin.side_effect = open_side_effect

        dataset = SmartBuildingsDataset(download=False)
        data, metadata = dataset.get_building_data(building, partition)

        self.assertEqual(data, mock_data_array)
        self.assertIn('device_infos', metadata)
        self.assertEqual(metadata['device_infos'], mock_device_infos)
        self.assertIn('zone_infos', metadata)
        self.assertEqual(metadata['zone_infos'], mock_zone_infos)
        self.assertEqual(metadata['description'], 'Initial metadata')

        # Check calls to open and pickle.load
        expected_data_path = f"./{building}/tabular/{building}/{partition}/data.npy.npz"
        mock_np_load.assert_called_once_with(expected_data_path)

        # Check open calls
        mock_open_builtin.assert_any_call(path_metadata, "rb")
        mock_open_builtin.assert_any_call(path_device_info, "rb")
        mock_open_builtin.assert_any_call(path_zone_info, "rb")
        self.assertEqual(mock_open_builtin.call_count, 3)

        # Check pickle_load calls
        # The actual file handles passed to pickle.load would be the ones returned by open_side_effect
        mock_pickle_load.assert_any_call(mock_handles[path_metadata])
        mock_pickle_load.assert_any_call(mock_handles[path_device_info])
        mock_pickle_load.assert_any_call(mock_handles[path_zone_info])
        self.assertEqual(mock_pickle_load.call_count, 3)


    def test_get_building_data_invalid_building(self):
        """Test get_building_data with an invalid building name."""
        dataset = SmartBuildingsDataset(download=False)
        with self.assertRaisesRegex(ValueError, "invalid building"):
            dataset.get_building_data('invalid_building', '2022_a')

    def test_get_building_data_invalid_partition(self):
        """Test get_building_data with an invalid partition name."""
        dataset = SmartBuildingsDataset(download=False)
        # Use a valid building from dataset.partitions
        valid_building = list(dataset.partitions.keys())[0]
        with self.assertRaisesRegex(ValueError, "invalid partition"):
            dataset.get_building_data(valid_building, 'invalid_partition')


if __name__ == '__main__':
    unittest.main()
