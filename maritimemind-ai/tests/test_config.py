from app.configs.config import MaritimeMindSettings

def test_config_loads():
    """Test that configuration can be loaded correctly."""
    # Instantiating the config should not raise errors
    config = MaritimeMindSettings()
    assert config is not None

def test_config_overrides(monkeypatch):
    """Test that environment variables override default settings."""
    monkeypatch.setenv("OLLAMA_MODEL", "overridden-model")
    config = MaritimeMindSettings()
    assert config.OLLAMA_MODEL == "overridden-model"

def test_config_defaults():
    """Test that expected defaults are present."""
    config = MaritimeMindSettings()
    assert isinstance(config.CHUNK_SIZE, int)
    assert config.TOP_K_RESULTS > 0
