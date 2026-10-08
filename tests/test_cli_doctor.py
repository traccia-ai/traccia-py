"""Tests for CLI doctor command encoding safety."""

import io
import unittest
from unittest.mock import patch

from traccia import cli


class TestCLIDoctorEncoding(unittest.TestCase):
    """Test main command under strict encoding streams."""

    def test_doctor_does_not_crash_on_cp1252(self):
        """Test main(["doctor"]) when stdout is cp1252 strict."""
        strict_stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")

        with patch("sys.stdout", strict_stdout), patch("urllib.request.urlopen"):
            result = cli.main(["doctor"])
            assert isinstance(result, int)


if __name__ == "__main__":
    unittest.main()
