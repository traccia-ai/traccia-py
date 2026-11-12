"""TRC-307: outbound HTTP requests under @govern are checked, one check per hop."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from traccia import runtime_config
from traccia.governance import outbound_http
from traccia.governance.policy import AgentBlockedError, ApprovalPending

PROD_URL = "https://api.prod.example.com/admin"
TRACCIA = ("key", "https://app.traccia.ai/v2/traces")


@pytest.fixture(autouse=True)
def outbound_on():
    outbound_http.patch_outbound_http()
    runtime_config.set_config_value(outbound_http.CONFIG_KEY, True)
    yield
    runtime_config.set_config_value(outbound_http.CONFIG_KEY, None)


def _decision(effect="allow"):
    return {"id": "d", "effect": effect, "would_have": False, "reasons": ["boundary"], "obligations": {}}


class _Checks:
    """Answers policy checks without the network, records what was checked."""

    def __init__(self, deny_hosts=(), queue_hosts=()):
        self.seen = []
        self.deny_hosts, self.queue_hosts = deny_hosts, queue_hosts

    def __call__(self, *, action, context=None, resource=None):
        url = context["input"]["url"]
        self.seen.append((action["type"], action["method"], url))
        if any(h in url for h in self.queue_hosts):
            raise ApprovalPending("held", approval_id="a1")
        if any(h in url for h in self.deny_hosts):
            raise AgentBlockedError("denied by Environment Boundary")
        return _decision()


def _governed():
    return runtime_config.run_identity(agent_id="support", pep_enabled=True)


def test_off_by_default_and_outside_govern():
    runtime_config.set_config_value(outbound_http.CONFIG_KEY, None)
    with patch.dict("os.environ", {"TRACCIA_CHECK_OUTBOUND_HTTP": ""}):
        assert outbound_http.enabled() is False
    with patch.dict("os.environ", {"TRACCIA_CHECK_OUTBOUND_HTTP": "1"}):
        assert outbound_http.enabled() is True
    runtime_config.set_config_value(outbound_http.CONFIG_KEY, True)
    # Not inside @govern: nothing is checked.
    assert outbound_http.should_check(PROD_URL) is False


def test_traccia_itself_is_never_checked():
    with _governed(), patch("traccia.client_config.resolve_credentials", return_value=TRACCIA):
        assert outbound_http.should_check("https://app.traccia.ai/api/v1/policy/check") is False
        assert outbound_http.should_check(PROD_URL) is True


def test_requests_checks_before_sending_and_denies():
    requests = pytest.importorskip("requests")

    checks = _Checks(deny_hosts=("prod.example.com",))
    with _governed(), patch("traccia.governance.pep.check_policy", checks), \
            patch("traccia.client_config.resolve_credentials", return_value=TRACCIA):
        with pytest.raises(AgentBlockedError):
            requests.get(PROD_URL, timeout=1)
    assert checks.seen == [("http_request", "GET", PROD_URL)]


def _redirecting_adapter(requests, hops):
    """A transport adapter that answers each URL from *hops* without the network."""
    from requests.adapters import BaseAdapter
    from requests.models import Response

    class Adapter(BaseAdapter):
        sent = []

        def send(self, request, **kwargs):
            self.sent.append(request.url)
            response = Response()
            response.request = request
            response.url = request.url
            target = hops.get(request.url)
            if target:
                response.status_code = 302
                response.headers["Location"] = target
            else:
                response.status_code = 200
            response._content = b"ok"
            return response

        def close(self):
            pass

    return Adapter()


def test_requests_checks_every_redirect_hop():
    requests = pytest.importorskip("requests")
    session = requests.Session()
    adapter = _redirecting_adapter(requests, {"https://safe.example.com/start": PROD_URL})
    session.mount("https://", adapter)
    checks = _Checks(deny_hosts=("prod.example.com",))
    with _governed(), patch("traccia.governance.pep.check_policy", checks), \
            patch("traccia.client_config.resolve_credentials", return_value=TRACCIA):
        with pytest.raises(AgentBlockedError):
            session.get("https://safe.example.com/start")
    assert [url for _, _, url in checks.seen] == ["https://safe.example.com/start", PROD_URL]
    # The production hop was never sent.
    assert adapter.sent == ["https://safe.example.com/start"]


def test_requests_without_a_match_go_out():
    requests = pytest.importorskip("requests")
    session = requests.Session()
    adapter = _redirecting_adapter(requests, {})
    session.mount("https://", adapter)
    checks = _Checks()
    with _governed(), patch("traccia.governance.pep.check_policy", checks), \
            patch("traccia.client_config.resolve_credentials", return_value=TRACCIA):
        assert session.post("https://safe.example.com/x").status_code == 200
    assert checks.seen == [("http_request", "POST", "https://safe.example.com/x")]


def test_policy_check_traffic_does_not_recurse():
    from unittest.mock import MagicMock

    from traccia.governance import pep

    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        return MagicMock(status_code=200, json=lambda: _decision())

    with _governed(), patch("traccia.governance.pep._credentials", return_value=("key", "https://checks.internal.example/v2/traces")), \
            patch("traccia.client_config.resolve_credentials", return_value=("key", "https://other.example/v2/traces")), \
            patch.object(pep._http_session, "post", side_effect=fake_post):
        outbound_http.check_request("GET", "https://safe.example.com/")
    assert calls == ["https://checks.internal.example/api/v1/policy/check"]


def test_approval_pending_stops_the_request():
    requests = pytest.importorskip("requests")
    session = requests.Session()
    adapter = _redirecting_adapter(requests, {})
    session.mount("https://", adapter)
    with _governed(), patch("traccia.governance.pep.check_policy", _Checks(queue_hosts=("prod.example.com",))), \
            patch("traccia.client_config.resolve_credentials", return_value=TRACCIA):
        with pytest.raises(ApprovalPending):
            session.get(PROD_URL)
    assert adapter.sent == []


def _httpx_transport(httpx, hops):
    sent = []

    def handler(request):
        sent.append(str(request.url))
        target = hops.get(str(request.url))
        if target:
            return httpx.Response(302, headers={"Location": target})
        return httpx.Response(200, text="ok")

    return httpx.MockTransport(handler), sent


def test_httpx_checks_every_redirect_hop():
    httpx = pytest.importorskip("httpx")
    transport, sent = _httpx_transport(httpx, {"https://safe.example.com/start": PROD_URL})
    checks = _Checks(deny_hosts=("prod.example.com",))
    with _governed(), patch("traccia.governance.pep.check_policy", checks), \
            patch("traccia.client_config.resolve_credentials", return_value=TRACCIA):
        with httpx.Client(transport=transport, follow_redirects=True) as client:
            with pytest.raises(AgentBlockedError):
                client.get("https://safe.example.com/start")
    assert [url for _, _, url in checks.seen] == ["https://safe.example.com/start", PROD_URL]
    assert sent == ["https://safe.example.com/start"]


def test_httpx_async_checks_every_redirect_hop():
    httpx = pytest.importorskip("httpx")
    transport, sent = _httpx_transport(httpx, {"https://safe.example.com/start": PROD_URL})
    checks = _Checks(deny_hosts=("prod.example.com",))

    async def run():
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            await client.get("https://safe.example.com/start")

    with _governed(), patch("traccia.governance.pep.check_policy", checks), \
            patch("traccia.client_config.resolve_credentials", return_value=TRACCIA):
        with pytest.raises(AgentBlockedError):
            asyncio.run(run())
    assert [url for _, _, url in checks.seen] == ["https://safe.example.com/start", PROD_URL]
    assert sent == ["https://safe.example.com/start"]


def test_patching_twice_is_harmless():
    first = outbound_http.patch_outbound_http()
    second = outbound_http.patch_outbound_http()
    assert first == second
