"""Exceptions used across csvai.

Every error type here is *recoverable*: callers are expected to catch these
and fall back to the deterministic path rather than aborting the command.
"""


class CsvaiError(Exception):
    """Base class for all csvai errors."""


class ConfigError(CsvaiError):
    """The AI layer is misconfigured (bad provider name, missing key, ...)."""


class ProviderError(CsvaiError):
    """A remote call failed: transport, HTTP status, or malformed payload."""


class ProviderUnavailable(ProviderError):
    """No usable provider at all - e.g. disabled, or no key configured.

    Raised before any network traffic happens, so it is cheap to catch.
    """


class PlanningError(CsvaiError):
    """The model returned something we refuse to execute."""


class UnsafeCommandError(PlanningError):
    """A proposed argv contained a flag or value outside the allow-list."""


class ModelRefusal(ProviderError):
    """The endpoint answered, but declined to produce usable output."""
