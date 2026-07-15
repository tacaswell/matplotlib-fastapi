"""Tests for JSON API endpoints."""

from fastapi.testclient import TestClient


class TestPlotsListAPI:
    """Tests for the /plots endpoint (JSON API)."""

    def test_list_plots_returns_200(self, client: TestClient) -> None:
        """Test that listing plots returns 200 OK."""
        response = client.get("/plots/plots")
        assert response.status_code == 200

    def test_list_plots_returns_json(self, client: TestClient) -> None:
        """Test that the response is valid JSON."""
        response = client.get("/plots/plots")
        assert response.headers["content-type"].startswith("application/json")
        data = response.json()
        assert isinstance(data, dict)

    def test_list_plots_has_expected_structure(self, client: TestClient) -> None:
        """Test that the response has the expected structure."""
        response = client.get("/plots/plots")
        data = response.json()

        # Should have a 'plots' key
        assert "plots" in data
        assert isinstance(data["plots"], dict)

    def test_list_plots_contains_test_plots(self, client: TestClient) -> None:
        """Test that our test plots are listed."""
        response = client.get("/plots/plots")
        data = response.json()

        plots = data["plots"]
        assert "simple" in plots
        assert "updatable" in plots

    def test_plot_info_structure(self, client: TestClient) -> None:
        """Test that each plot has the expected information."""
        response = client.get("/plots/plots")
        data = response.json()

        plot_info = data["plots"]["simple"]

        # Should have description
        assert "description" in plot_info
        assert isinstance(plot_info["description"], str)
        assert plot_info["description"] == "Simple test plot for basic functionality"

        # Should have parameters (JSON schema)
        assert "parameters" in plot_info
        assert isinstance(plot_info["parameters"], dict)
        assert "properties" in plot_info["parameters"]

        # Should have update_schema (None for simple plot)
        assert "update_schema" in plot_info
        assert plot_info["update_schema"] is None

    def test_updatable_plot_has_update_schema(self, client: TestClient) -> None:
        """Test that updatable plots include update schema."""
        response = client.get("/plots/plots")
        data = response.json()

        plot_info = data["plots"]["updatable"]

        # Should have update_schema
        assert plot_info["update_schema"] is not None
        assert isinstance(plot_info["update_schema"], dict)
        assert "properties" in plot_info["update_schema"]

    def test_plot_info_has_ws_url(self, client: TestClient) -> None:
        """Test that each plot includes a ws_url field."""
        response = client.get("/plots/plots")
        data = response.json()

        for name, info in data["plots"].items():
            assert "ws_url" in info, f"{name} missing ws_url"
            assert info["ws_url"] is not None
            assert f"/ws/v0/{name}" in info["ws_url"]

    def test_plot_info_has_view_url(self, client: TestClient) -> None:
        """Test that each plot includes a view_url field."""
        response = client.get("/plots/plots")
        data = response.json()

        for name, info in data["plots"].items():
            assert "view_url" in info, f"{name} missing view_url"
            assert info["view_url"] is not None
            assert f"/plot/{name}" in info["view_url"]

    def test_ws_url_uses_ws_scheme(self, client: TestClient) -> None:
        """Test that ws_url uses ws:// or wss:// scheme."""
        response = client.get("/plots/plots")
        data = response.json()

        for name, info in data["plots"].items():
            assert info["ws_url"].startswith("ws://") or info["ws_url"].startswith(
                "wss://"
            ), f"{name} ws_url has wrong scheme: {info['ws_url']}"

    def test_view_url_uses_http_scheme(self, client: TestClient) -> None:
        """Test that view_url uses http:// or https:// scheme."""
        response = client.get("/plots/plots")
        data = response.json()

        for name, info in data["plots"].items():
            assert info["view_url"].startswith("http://") or info[
                "view_url"
            ].startswith("https://"), (
                f"{name} view_url has wrong scheme: {info['view_url']}"
            )


class TestPlotSchemaAPI:
    """Tests for the /api/plots/{plot_name}/schema endpoint."""

    def test_get_schema_returns_200(self, client: TestClient) -> None:
        """Test that getting schema returns 200 OK."""
        response = client.get("/plots/api/plots/simple/schema")
        assert response.status_code == 200

    def test_get_schema_returns_json(self, client: TestClient) -> None:
        """Test that the response is valid JSON."""
        response = client.get("/plots/api/plots/simple/schema")
        assert response.headers["content-type"].startswith("application/json")

    def test_get_schema_structure(self, client: TestClient) -> None:
        """Test that schema has expected structure."""
        response = client.get("/plots/api/plots/simple/schema")
        data = response.json()

        assert "plot_name" in data
        assert data["plot_name"] == "simple"

        assert "description" in data
        assert isinstance(data["description"], str)

        assert "init_schema" in data
        assert isinstance(data["init_schema"], dict)

        assert "update_schema" in data
        # Simple plot has no update
        assert data["update_schema"] is None

    def test_get_schema_init_schema_content(self, client: TestClient) -> None:
        """Test that init_schema contains expected parameter definitions."""
        response = client.get("/plots/api/plots/simple/schema")
        data = response.json()

        init_schema = data["init_schema"]

        # Should have properties
        assert "properties" in init_schema
        properties = init_schema["properties"]

        # Should have 'value' parameter
        assert "value" in properties
        value_prop = properties["value"]

        # Check constraints
        assert value_prop["type"] == "number"
        assert value_prop["default"] == 1.0
        assert value_prop["minimum"] == 0.1
        assert value_prop["maximum"] == 10.0

    def test_get_schema_update_schema_content(self, client: TestClient) -> None:
        """Test that update_schema is present for updatable plots."""
        response = client.get("/plots/api/plots/updatable/schema")
        data = response.json()

        update_schema = data["update_schema"]
        assert update_schema is not None

        # Should have properties
        assert "properties" in update_schema
        properties = update_schema["properties"]

        # Should have 'phase' parameter
        assert "phase" in properties
        phase_prop = properties["phase"]

        # Check constraints
        assert phase_prop["type"] == "number"
        assert phase_prop["default"] == 0.0
        assert phase_prop["minimum"] == 0.0
        assert phase_prop["maximum"] == 6.28

    def test_get_schema_invalid_plot_returns_404(self, client: TestClient) -> None:
        """Test that requesting schema for non-existent plot returns 404."""
        response = client.get("/plots/api/plots/nonexistent/schema")
        assert response.status_code == 404

    def test_get_schema_invalid_plot_error_message(self, client: TestClient) -> None:
        """Test that 404 response includes helpful error message."""
        response = client.get("/plots/api/plots/nonexistent/schema")
        data = response.json()

        assert "detail" in data
        assert "nonexistent" in data["detail"]
        assert "Available plots:" in data["detail"]


class TestParameterValidation:
    """Tests for parameter schema validation."""

    def test_init_schema_required_fields(self, client: TestClient) -> None:
        """Test that schema correctly identifies required fields."""
        response = client.get("/plots/api/plots/simple/schema")
        data = response.json()

        init_schema = data["init_schema"]

        # 'value' has a default, so it's not required
        # Check the schema structure
        if "required" in init_schema:
            assert "value" not in init_schema["required"]

    def test_schemas_match_between_endpoints(self, client: TestClient) -> None:
        """Test that schemas are consistent across endpoints."""
        # Get from /plots endpoint
        list_response = client.get("/plots/plots")
        list_data = list_response.json()
        list_schema = list_data["plots"]["simple"]["parameters"]

        # Get from /api/plots/{name}/schema endpoint
        schema_response = client.get("/plots/api/plots/simple/schema")
        schema_data = schema_response.json()
        api_schema = schema_data["init_schema"]

        # Schemas should be identical
        assert list_schema == api_schema


class TestHealthCheckAPI:
    """Tests for the /health and /health/details endpoints."""

    def test_health_check_returns_200(self, client: TestClient) -> None:
        """Test that health check returns 200 OK."""
        response = client.get("/plots/health")
        assert response.status_code == 200

    def test_health_check_returns_json(self, client: TestClient) -> None:
        """Test that the response is valid JSON."""
        response = client.get("/plots/health")
        assert response.headers["content-type"].startswith("application/json")

    def test_health_check_is_minimal(self, client: TestClient) -> None:
        """Public /health leaks no operational detail."""
        response = client.get("/plots/health")
        data = response.json()

        # Only a bare status — no connection counts or plot names.
        assert data == {"status": "ok"}

    def test_health_details_structure(self, client: TestClient) -> None:
        """Test that /health/details has the expected structure."""
        response = client.get("/plots/health/details")
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "ok"

        assert "connections" in data
        assert isinstance(data["connections"], int)

        assert "connections_by_plot" in data
        assert isinstance(data["connections_by_plot"], dict)

        assert "cached_files" in data
        assert isinstance(data["cached_files"], int)

    def test_health_details_initial_state(self, client: TestClient) -> None:
        """Test that /health/details shows zero connections initially."""
        response = client.get("/plots/health/details")
        data = response.json()

        # No WebSocket connections in this test
        assert data["connections"] == 0
        assert data["connections_by_plot"] == {}


class TestSplitQueryParams:
    """Tests for the _split_query_params utility."""

    def test_no_update_params(self) -> None:
        from mpl_fastapi.router import _split_query_params

        init, update = _split_query_params({"frequency": "2.0", "amplitude": "1.5"})
        assert init == {"frequency": "2.0", "amplitude": "1.5"}
        assert update == {}

    def test_only_update_params(self) -> None:
        from mpl_fastapi.router import _split_query_params

        init, update = _split_query_params({"_update.phase": "1.57"})
        assert init == {}
        assert update == {"phase": "1.57"}

    def test_mixed_params(self) -> None:
        from mpl_fastapi.router import _split_query_params

        init, update = _split_query_params(
            {"frequency": "2.0", "_update.phase": "1.57", "amplitude": "1.0"}
        )
        assert init == {"frequency": "2.0", "amplitude": "1.0"}
        assert update == {"phase": "1.57"}

    def test_empty(self) -> None:
        from mpl_fastapi.router import _split_query_params

        init, update = _split_query_params({})
        assert init == {}
        assert update == {}


class TestNormalizeOrigin:
    """Tests for the _normalize_origin utility."""

    def test_lowercases_and_strips_trailing_slash(self) -> None:
        from mpl_fastapi.router import _normalize_origin

        assert _normalize_origin("https://Example.com/") == "https://example.com"

    def test_idempotent_on_canonical_form(self) -> None:
        from mpl_fastapi.router import _normalize_origin

        assert (
            _normalize_origin("https://example.com:3000") == "https://example.com:3000"
        )

    def test_strips_whitespace(self) -> None:
        from mpl_fastapi.router import _normalize_origin

        assert _normalize_origin("  http://localhost  ") == "http://localhost"
