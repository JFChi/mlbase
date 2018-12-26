import jarvis as j
import unittest
import numpy as np

class TestJarvis(unittest.TestCase):
    def test_read(self):
        df = j.read('example.csv')
        assert np.isnan(df['C'][0]), df['C'][0]

if __name__ == '__main__':
    unittest.main()