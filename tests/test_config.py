import unittest

from app.config import ConfigurationError, load_model_config


class ConfigTests(unittest.TestCase):
    def test_deepseek_config_reads_key_and_uses_safe_defaults(self):
        config = load_model_config(
            "deepseek",
            env={"DEEPSEEK_API_KEY": "test-key"},
        )

        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.base_url, "https://api.deepseek.com")
        self.assertEqual(config.model, "deepseek-flash")

    def test_deepseek_config_allows_model_and_base_url_override(self):
        config = load_model_config(
            "deepseek",
            env={
                "DEEPSEEK_API_KEY": "test-key",
                "DEEPSEEK_BASE_URL": "https://example.test/v1/",
                "DEEPSEEK_MODEL": "deepseek-v4-flash",
            },
        )

        self.assertEqual(config.base_url, "https://example.test/v1")
        self.assertEqual(config.model, "deepseek-v4-flash")

    def test_deepseek_config_requires_api_key(self):
        with self.assertRaisesRegex(ConfigurationError, "DEEPSEEK_API_KEY"):
            load_model_config("deepseek", env={})

    def test_fake_config_does_not_require_api_key(self):
        config = load_model_config("fake", env={})

        self.assertEqual(config.provider, "fake")
        self.assertIsNone(config.api_key)


if __name__ == "__main__":
    unittest.main()
