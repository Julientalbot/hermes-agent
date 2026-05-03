from agent.openrouter_headers import (
    OPENROUTER_ATTRIBUTION_HEADERS,
    OPENROUTER_RESPONSE_CACHE_HEADER,
    OPENROUTER_RESPONSE_CACHE_TTL_HEADER,
    openrouter_default_headers,
    openrouter_response_cache_headers,
)


def test_response_cache_disabled_by_default():
    headers = openrouter_default_headers(config={}, environ={})

    assert headers == OPENROUTER_ATTRIBUTION_HEADERS
    assert OPENROUTER_RESPONSE_CACHE_HEADER not in headers
    assert OPENROUTER_RESPONSE_CACHE_TTL_HEADER not in headers


def test_response_cache_config_enabled_with_ttl():
    headers = openrouter_response_cache_headers(
        config={"openrouter": {"response_cache": {"enabled": True, "ttl_seconds": 600}}},
        environ={},
    )

    assert headers == {
        OPENROUTER_RESPONSE_CACHE_HEADER: "true",
        OPENROUTER_RESPONSE_CACHE_TTL_HEADER: "600",
    }


def test_response_cache_env_overrides_config():
    headers = openrouter_response_cache_headers(
        config={"response_cache": {"enabled": False, "ttl_seconds": 300}},
        environ={
            "HERMES_OPENROUTER_RESPONSE_CACHE": "true",
            "HERMES_OPENROUTER_RESPONSE_CACHE_TTL": "900",
        },
    )

    assert headers == {
        OPENROUTER_RESPONSE_CACHE_HEADER: "true",
        OPENROUTER_RESPONSE_CACHE_TTL_HEADER: "900",
    }


def test_response_cache_env_can_disable_config():
    headers = openrouter_response_cache_headers(
        config={"response_cache": {"enabled": True, "ttl_seconds": 600}},
        environ={"HERMES_OPENROUTER_RESPONSE_CACHE": "false"},
    )

    assert headers == {}


def test_response_cache_invalid_ttl_is_ignored():
    headers = openrouter_response_cache_headers(
        config={"response_cache": {"enabled": True, "ttl_seconds": 999999}},
        environ={},
    )

    assert headers == {OPENROUTER_RESPONSE_CACHE_HEADER: "true"}
