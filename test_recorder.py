import struct
import unittest
from unittest.mock import patch, Mock, MagicMock
import os
from record import OUTPUT_DIRECTORY, Recorder


class TestRecorder(unittest.TestCase):
    @patch("wave.open", autospec=True)
    def setUp(self, wave_open_mock):
        mock_p = MagicMock()
        mock_p.get_device_info_by_host_api_device_index = Mock()
        mock_p.get_device_info_by_host_api_device_index().get.return_value = 1
        mock_p.get_host_api_info_by_index = Mock()
        mock_p.get_host_api_info_by_index().get.return_value = 1
        mock_p.get_sample_size().return_value = 1
        mock_p

        with patch("pyaudio.PyAudio", return_value=mock_p):
            self.recorder = Recorder()
            self.recorder.p = mock_p
            # self.recorder.p = p
            self.stream_mock = Mock()
            self.recorder.record = Mock()
            self.recorder.stream = self.stream_mock
           

    def test_rms(self):
        frame = struct.pack("h", 10) * 2
        output = self.recorder.rms(frame)
        expected_output = 0.2214
        self.assertAlmostEqual(output, expected_output, places=4)
    
    def test_write(self):
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        self.recorder.write(b"")
        self.assertTrue(os.path.exists(OUTPUT_DIRECTORY))
        os.rmdir(OUTPUT_DIRECTORY)

    @patch("time.time")
    def test_record(self, mock_time):
        mock_time.side_effect = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.1, 5.6, 6.0]
        self.stream_mock.read.return_value = struct.pack("h", 10) * 2
        bar = Mock()
        self.recorder.record(bar)
        self.assertEqual(self.stream_mock.read.call_count, 5)


if __name__ == "__main__":
    unittest.main()
