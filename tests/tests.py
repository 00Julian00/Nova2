"""
Description: Tests for Nova2.
"""

import unittest

import coverage

from Nova2 import *

class Test(unittest.TestCase):
    def setUp(self):
        self.nova = Nova()

    def test_context(self):
        # Add context
        ctx_size = len(self.nova.get_context().data_points)

        self.nova.add_to_context(
            ContextSource_User(),
            "Test message"
        )

        self.assertEquals(
            len(self.nova.get_context().data_points),
            ctx_size + 1
        )

def run_tests():
    cov = coverage.Coverage()
    cov.start()
    
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner().run(suite)
    
    cov.stop()
    cov.save()
    
    print("\nCoverage Report:")
    cov.report()