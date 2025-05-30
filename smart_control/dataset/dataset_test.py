"""Tests for SmartBuildingsDataset"""

from absl.testing import absltest

from smart_control.dataset.dataset import SmartBuildingsDataset


class TestDataset(absltest.TestCase):

  def test_download(self):
    ds = SmartBuildingsDataset()

    # TODO: test download capabilities

    ds.download()


    #breakpoint()
    #self.assertEqual(2+2, 4)


if __name__ == "__main__":
  absltest.main()
