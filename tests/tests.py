"""
Description: Tests for Nova2.
"""

import unittest

import coverage

from Nova2 import *
from Nova2.app.context_data import ContextSource_User

class Test(unittest.TestCase):
    def setUp(self):
        self.nova = Nova()
        self.nova.set_active_context_file('debug_ctx')

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

    def test_tools(self):
        self.nova.load_tools()
        self.assertGreater(
            len(self.nova._tool_manager._loaded_tools),
            0
        )

    def test_llm(self):
        conditioning = LLMConditioning(
            model="llama-3.3-70b-versatile",
            inference_engine="inference_groq"
        )

        self.nova.configure_llm(conditioning)
        self.nova.apply_config_llm()

        conv = Conversation()

        resp = self.nova.run_llm(conv)

        self.assertTrue(
            type(resp) == LLMResponse
        )

    def test_tts(self):
        conditioning = TTSConditioning(
            model="eleven_flash_v2_5",
            inference_engine="inference_elevenlabs",
            voice="FGY2WhTYpPnrIDTdsKH5",
            expressivness=0.5,
            stability=0.5,
            similarity_boost=0.75,
            use_speaker_boost=False
        )

        self.nova.configure_tts( conditioning)
        self.nova.apply_config_tts()

        resp = self.nova.run_tts("Hello World")

def run_tests():
    cov = coverage.Coverage(
        source=['.'],
        omit=[
            "*/site-packages/*",
            "*/dist-packages/*", 
            "*/venv/*",
            "*/.venv/*",
            "*/tests/*",
            "*/__pycache__/*",
            "debug_runner.py",
            "run_tests.py"
            "*/external/*"
        ]
    )
    cov.start()
    
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner().run(suite)
    
    cov.stop()
    #cov.save()
    
    print("\nCoverage Report:")
    cov.report()