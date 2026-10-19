"""Test doubles for the provider, so every failure path is reproducible.

The point of these fakes is not to avoid the network for speed - it is that a
real endpoint cannot be made to return 500, hang, or emit "Sure! Here you go:"
on demand. Each scenario below is one that actually broke something during
development, or one that a hostile/cheap model plausibly produces.
"""

import json


class FakeTransport:
    """A ``transport`` compatible with :class:`csvai.provider.ChatProvider`.

    ``script`` is a list of responses consumed in order. An item may be:

    * ``dict`` with ``choices`` - returned verbatim as an API payload
    * ``dict`` without ``choices`` - wrapped into ``choices`` as JSON content
    * ``str`` - returned as the assistant message content
    * ``Exception`` - raised, to exercise retry and fallback paths

    Use :class:`RawPayload` when you need a malformed payload shape; a plain
    dict is assumed to be well-formed content.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({'url': url, 'headers': dict(headers), 'body': body,
                           'timeout': timeout})
        if not self.script:
            raise AssertionError('unexpected extra call')
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, RawPayload):
            return item.value
        if isinstance(item, dict):
            return {
                'choices': [{'message': {'content': json.dumps(item, ensure_ascii=False)}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
            }
        if isinstance(item, str):
            return {
                'choices': [{'message': {'content': item}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
            }
        raise AssertionError(f'unsupported script item: {item!r}')


class RawPayload:
    """Wrap a payload that must reach the provider unmodified.

    Needed to test shape errors: the provider has to *see* ``{"unexpected": ...}``
    rather than a well-formed envelope, otherwise the test passes for the wrong
    reason.
    """

    def __init__(self, value):
        self.value = value


def make_config(**overrides):
    """A hermetic config: no env, no config file, explicit values only."""
    from csvai import config as config_module

    cli = {
        'provider': 'custom',
        'base_url': 'https://example.invalid/v1',
        'model': 'test-model',
        'api_key': 'test-key-0123456789',
        'cache': False,
    }
    cli.update(overrides)
    return config_module.resolve(cli_values=cli, env={})


def make_provider(script, **overrides):
    from csvai.provider import ChatProvider

    cfg = make_config(**overrides)
    transport = FakeTransport(script)
    provider = ChatProvider(cfg, transport=transport, sleep=lambda _s: None)
    return provider, transport


def table_from(header, rows):
    from csvai.profile import profile_rows

    return profile_rows(header, rows, path='test.csv')
