"""Configuration resolution for csvai.

Precedence, highest first:

1. explicit values passed by the caller (usually parsed CLI flags)
2. environment variables (``CSVAI_*``)
3. a config file: ``$CSVAI_CONFIG``, else ``./.csvai.toml``,
   else ``~/.config/csvai/config.toml``
4. built-in defaults

Every resolved value remembers *where it came from*. That provenance is what
``csvai config`` prints, and it is the only reliable way to answer "why is it
talking to that endpoint?" months later.

The module never raises on a missing config file, and never imports a TOML
writer: reading uses ``tomllib`` (3.11+) with a tiny fallback parser so the
package keeps working on older interpreters.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigError

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on <3.11
    tomllib = None

ENV_PREFIX = 'CSVAI_'

#: Endpoint presets. ``base_url`` is what the user would otherwise have to know
#: by heart; keeping them here means switching provider is one word.
PROVIDERS = {
    'deepseek': {
        'base_url': 'https://api.deepseek.com/v1',
        'model': 'deepseek-flash',
        'key_env': 'DEEPSEEK_API_KEY',
    },
    'openai': {
        'base_url': 'https://api.openai.com/v1',
        'model': 'gpt-4o-mini',
        'key_env': 'OPENAI_API_KEY',
    },
    'moonshot': {
        'base_url': 'https://api.moonshot.cn/v1',
        'model': 'moonshot-v1-8k',
        'key_env': 'MOONSHOT_API_KEY',
    },
    'ollama': {
        # Local models need no key at all; this is the offline-ish path.
        'base_url': 'http://localhost:11434/v1',
        'model': 'qwen2.5:7b',
        'key_env': None,
    },
    'custom': {
        'base_url': '',
        'model': '',
        'key_env': 'CSVAI_API_KEY',
    },
}

DEFAULTS = {
    'provider': 'deepseek',
    'base_url': None,       # resolved from provider preset when None
    'model': None,          # ditto
    'api_key': None,        # resolved from provider's key_env when None
    'timeout': 45.0,
    'max_retries': 2,
    'temperature': 0.0,
    'cache': True,
    'cache_dir': None,      # resolved to a user cache dir when None
    'sample_rows': 40,
    'enabled': True,
}

_TRUTHY = {'1', 'true', 'yes', 'on', 'y', 't'}
_FALSY = {'0', 'false', 'no', 'off', 'n', 'f'}

#: Env var name -> config key.
_ENV_KEYS = {
    'CSVAI_PROVIDER': 'provider',
    'CSVAI_BASE_URL': 'base_url',
    'CSVAI_MODEL': 'model',
    'CSVAI_API_KEY': 'api_key',
    'CSVAI_TIMEOUT': 'timeout',
    'CSVAI_MAX_RETRIES': 'max_retries',
    'CSVAI_TEMPERATURE': 'temperature',
    'CSVAI_CACHE': 'cache',
    'CSVAI_CACHE_DIR': 'cache_dir',
    'CSVAI_SAMPLE_ROWS': 'sample_rows',
}

_BOOL_KEYS = {'cache', 'enabled'}
_INT_KEYS = {'max_retries', 'sample_rows'}
_FLOAT_KEYS = {'timeout', 'temperature'}


def _parse_bool(raw):
    text = str(raw).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise ConfigError(f'not a boolean: {raw!r}')


def _coerce(key, raw):
    if key in _BOOL_KEYS:
        return _parse_bool(raw)
    if key in _INT_KEYS:
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ConfigError(f'{key} must be an integer, got {raw!r}') from None
    if key in _FLOAT_KEYS:
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise ConfigError(f'{key} must be a number, got {raw!r}') from None
    return str(raw)


def _fallback_toml(text):
    """Minimal ``key = value`` reader for interpreters without tomllib.

    Only flat string/number/bool tables under ``[csvai]`` are supported, which
    is exactly what this config format uses. Anything fancier is ignored rather
    than mis-parsed.
    """
    data = {}
    section = None
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('[') and line.endswith(']'):
            section = line[1:-1].strip()
            data.setdefault(section, {})
            continue
        if '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1]
        elif value.lower() in _TRUTHY | _FALSY:
            value = value.lower() in _TRUTHY
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        if section:
            data[section][key] = value
        else:
            data[key] = value
    return data


def read_config_file(path):
    """Read ``[csvai]`` from *path*. Returns ``({}, None)`` when absent."""
    path = Path(path)
    if not path.is_file():
        return {}, None
    try:
        raw = path.read_text(encoding='utf-8')
    except OSError as exc:
        raise ConfigError(f'cannot read config {path}: {exc}') from None
    if tomllib is not None:
        try:
            data = tomllib.loads(raw)
        except Exception as exc:  # tomllib raises several unrelated types
            raise ConfigError(f'cannot parse config {path}: {exc}') from None
    else:  # pragma: no cover
        data = _fallback_toml(raw)
    section = data.get('csvai', data)
    return {k: v for k, v in section.items() if isinstance(k, str)}, path


def _candidate_config_paths(explicit=None):
    if explicit:
        return [Path(explicit)]
    paths = []
    env_path = os.environ.get(ENV_PREFIX + 'CONFIG')
    if env_path:
        paths.append(Path(env_path))
    paths.append(Path.cwd() / '.csvai.toml')
    paths.append(Path.home() / '.config' / 'csvai' / 'config.toml')
    return paths


@dataclass
class Config:
    """A fully resolved configuration, plus the provenance of each value."""

    values: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    config_path: Path = None

    # -- convenience accessors -------------------------------------------
    def __getitem__(self, key):
        return self.values[key]

    def get(self, key, default=None):
        return self.values.get(key, default)

    @property
    def enabled(self):
        return bool(self.values.get('enabled'))

    @property
    def base_url(self):
        return (self.values.get('base_url') or '').rstrip('/')

    @property
    def model(self):
        return self.values.get('model')

    @property
    def api_key(self):
        return self.values.get('api_key')

    def source_of(self, key):
        return self.sources.get(key, 'default')

    def as_public_dict(self):
        """Config for display: the key is redacted, never echoed."""
        out = dict(self.values)
        key = out.get('api_key')
        out['api_key'] = _redact(key)
        out['cache_dir'] = str(self.cache_path()) if self.values.get('cache') else None
        return out

    def cache_path(self):
        raw = self.values.get('cache_dir')
        if raw:
            return Path(raw).expanduser()
        base = os.environ.get('XDG_CACHE_HOME')
        root = Path(base) if base else Path.home() / '.cache'
        return root / 'csvai'


def _redact(key):
    if not key:
        return None
    if len(key) <= 8:
        return '*' * len(key)
    return f'{key[:4]}...{key[-4:]}'


def default_cache_dir():
    base = os.environ.get('XDG_CACHE_HOME')
    root = Path(base) if base else Path.home() / '.cache'
    return root / 'csvai'


def resolve(cli_values=None, env=None, config_path=None):
    """Merge the four layers and return a :class:`Config`.

    *cli_values* maps config keys to already-typed values (``None`` means
    "not given"). *env* defaults to ``os.environ`` and exists so tests can
    inject a hermetic environment.
    """
    env = os.environ if env is None else env
    cli_values = {k: v for k, v in (cli_values or {}).items() if v is not None}

    values, sources = {}, {}

    for key, default in DEFAULTS.items():
        values[key] = default
        sources[key] = 'default'

    # A hard kill switch, checked before anything else so that a broken config
    # file cannot prevent disabling the layer.
    disabled = str(env.get(ENV_PREFIX + 'DISABLE', '')).strip().lower()
    if disabled in _TRUTHY:
        values['enabled'] = False
        sources['enabled'] = 'env:CSVAI_DISABLE'

    file_values, used_path = {}, None
    for candidate in _candidate_config_paths(config_path):
        file_values, used_path = read_config_file(candidate)
        if used_path is not None:
            break

    for key, raw in file_values.items():
        if key in DEFAULTS:
            values[key] = _coerce(key, raw)
            sources[key] = f'file:{used_path.name}'

    for env_name, key in _ENV_KEYS.items():
        if env_name in env and env[env_name] != '':
            values[key] = _coerce(key, env[env_name])
            sources[key] = f'env:{env_name}'

    for key, raw in cli_values.items():
        if key not in DEFAULTS:
            continue
        values[key] = _coerce(key, raw) if not isinstance(raw, (bool, int, float)) else raw
        sources[key] = 'cli'

    _apply_provider_presets(values, sources, env)

    cfg = Config(values=values, sources=sources, config_path=used_path)
    _validate(cfg)
    return cfg


def _apply_provider_presets(values, sources, env):
    provider = values['provider']
    if provider not in PROVIDERS:
        if not values.get('enabled'):
            # A disabled layer must not fail on a bad provider name: the whole
            # point of the kill switch is to work when the config is broken.
            values['base_url'] = values.get('base_url') or ''
            values['model'] = values.get('model') or ''
            return
        raise ConfigError(
            f'unknown provider {provider!r}; choose from {", ".join(sorted(PROVIDERS))}'
        )
    preset = PROVIDERS[provider]

    if not values.get('base_url'):
        values['base_url'] = preset['base_url']
        sources['base_url'] = f'provider:{provider}'

    if not values.get('model'):
        values['model'] = preset['model']
        sources['model'] = f'provider:{provider}'

    if not values.get('api_key'):
        key_env = preset['key_env']
        if key_env and env.get(key_env):
            values['api_key'] = env[key_env]
            sources['api_key'] = f'env:{key_env}'


def _validate(cfg):
    values = cfg.values
    # Skip validation entirely when disabled: a broken config is exactly the
    # situation in which someone reaches for the kill switch.
    if not values.get('enabled'):
        return
    if values['provider'] == 'custom' and not values.get('base_url'):
        raise ConfigError('provider "custom" requires base_url (--ai-base-url or CSVAI_BASE_URL)')
    if values['timeout'] <= 0:
        raise ConfigError('timeout must be positive')
    if values['max_retries'] < 0:
        raise ConfigError('max_retries must be >= 0')
    if values['sample_rows'] < 1:
        raise ConfigError('sample_rows must be >= 1')


def is_configured(cfg):
    """True when a call could plausibly succeed.

    Local endpoints (ollama, or any custom URL with no key) are considered
    configured without a key; remote ones are not.
    """
    if not cfg.enabled:
        return False
    if not cfg.base_url or not cfg.model:
        return False
    if cfg.api_key:
        return True
    return cfg['provider'] in ('ollama', 'custom')
