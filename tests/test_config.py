"""Configuration precedence and failure behaviour.

Every case here corresponds to a question someone will actually ask months after
deployment: "why is it talking to that endpoint?", "why did it stop working?".
"""

import pytest

from csvai import config as config_module
from csvai.errors import ConfigError


def test_defaults_come_from_provider_preset():
    cfg = config_module.resolve(env={}, config_path='definitely-missing.toml')
    assert cfg['provider'] == 'deepseek'
    assert cfg.base_url == 'https://api.deepseek.com/v1'
    assert cfg.model == 'deepseek-flash'
    assert cfg.source_of('model') == 'provider:deepseek'


def test_env_overrides_defaults():
    cfg = config_module.resolve(env={'CSVAI_MODEL': 'my-model'}, config_path='missing.toml')
    assert cfg.model == 'my-model'
    assert cfg.source_of('model') == 'env:CSVAI_MODEL'


def test_cli_overrides_env():
    cfg = config_module.resolve(
        cli_values={'model': 'from-cli'},
        env={'CSVAI_MODEL': 'from-env'},
        config_path='missing.toml',
    )
    assert cfg.model == 'from-cli'
    assert cfg.source_of('model') == 'cli'


def test_file_overrides_defaults_but_not_env(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text('[csvai]\nmodel = "from-file"\ntimeout = 12\n', encoding='utf-8')

    cfg = config_module.resolve(env={}, config_path=str(path))
    assert cfg.model == 'from-file'
    assert cfg['timeout'] == 12
    assert cfg.source_of('model') == 'file:config.toml'

    cfg2 = config_module.resolve(env={'CSVAI_MODEL': 'from-env'}, config_path=str(path))
    assert cfg2.model == 'from-env'


def test_disable_switch_wins_over_everything():
    cfg = config_module.resolve(
        cli_values={'model': 'x'},
        env={'CSVAI_DISABLE': '1'},
        config_path='missing.toml',
    )
    assert cfg.enabled is False
    assert cfg.source_of('enabled') == 'env:CSVAI_DISABLE'


def test_disable_switch_survives_a_broken_config_file(tmp_path):
    # The kill switch must work even when the config is unusable, otherwise a
    # bad file could prevent someone from turning the feature off.
    path = tmp_path / 'config.toml'
    path.write_text('[csvai]\nprovider = "not-a-provider"\n', encoding='utf-8')
    cfg = config_module.resolve(env={'CSVAI_DISABLE': 'true'}, config_path=str(path))
    assert cfg.enabled is False


def test_unknown_provider_is_an_error():
    with pytest.raises(ConfigError, match='unknown provider'):
        config_module.resolve(cli_values={'provider': 'nope'}, env={},
                              config_path='missing.toml')


def test_custom_provider_requires_base_url():
    with pytest.raises(ConfigError, match='requires base_url'):
        config_module.resolve(
            cli_values={'provider': 'custom', 'model': 'm', 'api_key': 'k'},
            env={}, config_path='missing.toml')


def test_local_endpoint_needs_no_key():
    cfg = config_module.resolve(cli_values={'provider': 'ollama'}, env={},
                                config_path='missing.toml')
    assert config_module.is_configured(cfg) is True


def test_remote_provider_without_key_is_not_configured():
    cfg = config_module.resolve(cli_values={'provider': 'openai'}, env={},
                                config_path='missing.toml')
    assert config_module.is_configured(cfg) is False


def test_provider_key_env_is_picked_up():
    cfg = config_module.resolve(env={'DEEPSEEK_API_KEY': 'sk-abc'},
                                config_path='missing.toml')
    assert cfg.api_key == 'sk-abc'
    assert cfg.source_of('api_key') == 'env:DEEPSEEK_API_KEY'


def test_public_dict_redacts_the_key():
    cfg = config_module.resolve(env={'DEEPSEEK_API_KEY': 'sk-1234567890abcdef'},
                                config_path='missing.toml')
    public = cfg.as_public_dict()
    assert public['api_key'] == 'sk-1...cdef'
    assert 'sk-1234567890abcdef' not in str(public)


def test_invalid_numbers_are_rejected_with_a_clear_message():
    with pytest.raises(ConfigError, match='must be an integer'):
        config_module.resolve(env={'CSVAI_MAX_RETRIES': 'many'}, config_path='missing.toml')
