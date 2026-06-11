"""Tests for pluggable authentication in mpl_fastapi."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from mpl_fastapi import (
    NoAuth,
    PlotConfig,
    SingleUserToken,
    create_mpl_router,
    install_mpl_router,
)
from mpl_fastapi.auth import AuthPolicy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def token() -> str:
    return "test-secret-token-abc123"


@pytest.fixture
def auth_app(
    simple_plot_config: PlotConfig,
    updatable_plot_config: PlotConfig,
    token: str,
) -> FastAPI:
    """FastAPI app with SingleUserToken auth."""
    app = FastAPI()
    mpl = create_mpl_router(
        {
            "simple": simple_plot_config,
            "updatable": updatable_plot_config,
        },
        auth=SingleUserToken(token=token),
    )
    install_mpl_router(app, mpl, prefix="/plots")
    return app


@pytest.fixture
def auth_client(auth_app: FastAPI) -> TestClient:
    return TestClient(auth_app)


@pytest.fixture
def open_app(
    simple_plot_config: PlotConfig,
) -> FastAPI:
    """FastAPI app with NoAuth (explicit)."""
    app = FastAPI()
    mpl = create_mpl_router(
        {"simple": simple_plot_config},
        auth=NoAuth(),
    )
    install_mpl_router(app, mpl, prefix="/plots")
    return app


@pytest.fixture
def open_client(open_app: FastAPI) -> TestClient:
    return TestClient(open_app)


# ---------------------------------------------------------------------------
# AuthPolicy protocol
# ---------------------------------------------------------------------------


class TestAuthPolicyProtocol:
    """Verify the AuthPolicy protocol is satisfied by built-in classes."""

    def test_noauth_satisfies_protocol(self) -> None:
        assert isinstance(NoAuth(), AuthPolicy)

    def test_single_user_token_satisfies_protocol(self) -> None:
        assert isinstance(SingleUserToken(token="x"), AuthPolicy)


# ---------------------------------------------------------------------------
# NoAuth (open access)
# ---------------------------------------------------------------------------


class TestNoAuth:
    """Endpoints are accessible without credentials when using NoAuth."""

    def test_plots_list_open(self, open_client: TestClient) -> None:
        resp = open_client.get("/plots/")
        assert resp.status_code == 200

    def test_plots_json_open(self, open_client: TestClient) -> None:
        resp = open_client.get("/plots/plots")
        assert resp.status_code == 200

    def test_plot_view_open(self, open_client: TestClient) -> None:
        resp = open_client.get("/plots/plot/simple")
        assert resp.status_code == 200

    def test_schema_open(self, open_client: TestClient) -> None:
        resp = open_client.get("/plots/api/plots/simple/schema")
        assert resp.status_code == 200

    def test_health_no_auth_needed(self, open_client: TestClient) -> None:
        resp = open_client.get("/plots/health")
        assert resp.status_code == 200

    def test_health_details_open(self, open_client: TestClient) -> None:
        resp = open_client.get("/plots/health/details")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# SingleUserToken — unauthenticated (should be rejected)
# ---------------------------------------------------------------------------


class TestSingleUserTokenRejection:
    """Protected endpoints reject requests without a valid token."""

    def test_plots_list_requires_token(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/plots/")
        assert resp.status_code == 401

    def test_plots_json_requires_token(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/plots/plots")
        assert resp.status_code == 401

    def test_plot_view_requires_token(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/plots/plot/simple")
        assert resp.status_code == 401

    def test_download_requires_token(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/plots/download/nonexistent-id")
        assert resp.status_code == 401

    def test_schema_requires_token(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/plots/api/plots/simple/schema")
        assert resp.status_code == 401

    def test_www_authenticate_header(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/plots/plots")
        assert resp.headers.get("www-authenticate") == "Bearer"

    def test_wrong_token_rejected(self, auth_client: TestClient) -> None:
        resp = auth_client.get(
            "/plots/plots",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    def test_health_no_auth_needed(self, auth_client: TestClient) -> None:
        """Public health check is always accessible, even with auth enabled."""
        resp = auth_client.get("/plots/health")
        assert resp.status_code == 200

    def test_public_health_is_minimal(self, auth_client: TestClient) -> None:
        """The unauthenticated /health leaks no operational detail."""
        resp = auth_client.get("/plots/health")
        assert resp.json() == {"status": "ok"}

    def test_health_details_requires_token(self, auth_client: TestClient) -> None:
        """Detailed health stats require authentication."""
        resp = auth_client.get("/plots/health/details")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# SingleUserToken — authenticated (should succeed)
# ---------------------------------------------------------------------------


class TestSingleUserTokenAccess:
    """Protected endpoints accept requests with a valid token."""

    def test_bearer_header(self, auth_client: TestClient, token: str) -> None:
        resp = auth_client.get(
            "/plots/plots",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_query_param(self, auth_client: TestClient, token: str) -> None:
        resp = auth_client.get(f"/plots/plots?token={token}")
        assert resp.status_code == 200

    def test_plot_view_with_token(self, auth_client: TestClient, token: str) -> None:
        resp = auth_client.get(
            "/plots/plot/simple",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_schema_with_token(self, auth_client: TestClient, token: str) -> None:
        resp = auth_client.get(
            "/plots/api/plots/simple/schema",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_health_details_with_token(
        self, auth_client: TestClient, token: str
    ) -> None:
        resp = auth_client.get(
            "/plots/health/details",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert "connections" in resp.json()

    def test_plots_list_html_with_token(
        self, auth_client: TestClient, token: str
    ) -> None:
        resp = auth_client.get(
            "/plots/",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# SingleUserToken — WebSocket
# ---------------------------------------------------------------------------


class TestSingleUserTokenWebSocket:
    """WebSocket endpoint respects token auth."""

    def test_ws_rejected_without_token(self, auth_client: TestClient) -> None:
        with (
            pytest.raises(WebSocketDisconnect),
            auth_client.websocket_connect("/plots/ws/v0/simple"),
        ):
            pass  # Should not reach here

    def test_ws_rejected_with_wrong_token(self, auth_client: TestClient) -> None:
        with (
            pytest.raises(WebSocketDisconnect),
            auth_client.websocket_connect("/plots/ws/v0/simple?token=wrong"),
        ):
            pass

    def test_ws_accepted_with_query_token(
        self, auth_client: TestClient, token: str
    ) -> None:
        with auth_client.websocket_connect(f"/plots/ws/v0/simple?token={token}") as ws:
            # Send init message
            ws.send_json(
                {
                    "type": "init",
                    "protocol_version": 0,
                    "device_pixel_ratio": 1.0,
                    "supports_binary": True,
                }
            )
            # Should get config back
            config = ws.receive_json()
            assert config["type"] == "config"


# ---------------------------------------------------------------------------
# SingleUserToken — token sources
# ---------------------------------------------------------------------------


class TestSingleUserTokenSources:
    """Token resolution priority: explicit > env > generated."""

    def test_explicit_token(self) -> None:
        auth = SingleUserToken(token="my-explicit-token")
        assert auth.token == "my-explicit-token"

    def test_env_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MPL_FASTAPI_TOKEN", "env-token-value")
        auth = SingleUserToken()
        assert auth.token == "env-token-value"

    def test_generated_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MPL_FASTAPI_TOKEN", raising=False)
        auth = SingleUserToken()
        # Should be a non-empty string
        assert isinstance(auth.token, str)
        assert len(auth.token) > 16

    def test_explicit_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MPL_FASTAPI_TOKEN", "env-token")
        auth = SingleUserToken(token="explicit-token")
        assert auth.token == "explicit-token"


# ---------------------------------------------------------------------------
# JS bundle endpoints — never behind auth
# ---------------------------------------------------------------------------


class TestJSBundlesNoAuth:
    """JS bundle endpoints should be accessible even with auth enabled."""

    def test_component_js_no_auth(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/plots/component.js")
        # 200 if built, 404 if not — but never 401
        assert resp.status_code != 401

    def test_component_esm_js_no_auth(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/plots/component.esm.js")
        assert resp.status_code != 401

    def test_mpl_js_no_auth(self, auth_client: TestClient) -> None:
        resp = auth_client.get("/plots/js/mpl.js")
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# Default auth (None → NoAuth)
# ---------------------------------------------------------------------------


class TestDefaultAuth:
    """When auth is omitted, the router should behave like NoAuth."""

    def test_no_auth_by_default(self, client: TestClient) -> None:
        """The standard test_app fixture passes no auth — should work."""
        resp = client.get("/plots/plots")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Custom AuthPolicy — user-provided
# ---------------------------------------------------------------------------


class TestCustomAuthPolicy:
    """Users can provide any object satisfying the AuthPolicy protocol."""

    def test_custom_policy(self, simple_plot_config: PlotConfig) -> None:
        """A custom auth policy is accepted and invoked."""
        calls: list[str] = []

        class RecordingAuth:
            def http_dependency(self):
                async def _dep():
                    calls.append("http")

                return _dep

            def ws_dependency(self):
                async def _dep():
                    calls.append("ws")

                return _dep

        app = FastAPI()
        mpl = create_mpl_router(
            {"simple": simple_plot_config},
            auth=RecordingAuth(),
        )
        install_mpl_router(app, mpl, prefix="/plots")

        client = TestClient(app)
        client.get("/plots/plots")
        assert "http" in calls
