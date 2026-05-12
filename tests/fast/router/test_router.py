from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-fast")

import asyncio
from argparse import Namespace

import pytest
import requests
from fastapi import FastAPI
from fastapi.responses import Response

from miles.router.router import MilesRouter, _aggregate_prometheus_metrics
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


def make_router_args(router_port: int, **overrides) -> Namespace:
    defaults = dict(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=router_port,
        rollout_health_check_interval=1.0,
        miles_router_health_check_failure_threshold=3,
        miles_router_max_connections=100,
        miles_router_timeout=None,
        miles_router_middleware_paths=[],
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def create_mock_worker(start_port: int = 30000):
    from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, default_process_fn

    port = find_available_port(start_port)
    return MockSGLangServer(
        model_name="Qwen/Qwen3-0.6B",
        process_fn=default_process_fn,
        host="127.0.0.1",
        port=port,
        latency=0.0,
    )


def create_metrics_worker(metrics_text: str, start_port: int = 31000) -> UvicornThreadServer:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return Response(content=metrics_text, media_type="text/plain")

    server = UvicornThreadServer(app, host="127.0.0.1", port=find_available_port(start_port))
    server.start()
    return server


class RouterEnv:
    def __init__(self, router: MilesRouter, server: UvicornThreadServer):
        self.router = router
        self.server = server

    @property
    def url(self) -> str:
        return self.server.url


@pytest.fixture
def router_env():
    args = make_router_args(find_available_port(20000))
    router = MilesRouter(args, verbose=False)
    server = UvicornThreadServer(router.app, host=args.sglang_router_ip, port=args.sglang_router_port)
    server.start()
    yield RouterEnv(router, server)
    server.stop()


@pytest.fixture
def mock_worker():
    server = create_mock_worker()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def mock_worker_factory():
    servers = []

    def _create():
        start_port = 30000 + len(servers) * 100
        server = create_mock_worker(start_port)
        server.start()
        servers.append(server)
        return server

    yield _create
    for s in servers:
        s.stop()


@pytest.fixture
def router_factory():
    def _create(**overrides) -> MilesRouter:
        args = make_router_args(find_available_port(20000), **overrides)
        return MilesRouter(args, verbose=False)

    return _create


class TestWorkerManagement:
    def test_add_worker_via_query_param(self, router_env: RouterEnv):
        worker_url = "http://127.0.0.1:30001"
        r = requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0)
        r.raise_for_status()

        assert r.json()["status"] == "success"
        assert worker_url in router_env.router.worker_request_counts
        assert router_env.router.worker_request_counts[worker_url] == 0

    def test_add_worker_via_body(self, router_env: RouterEnv):
        worker_url = "http://127.0.0.1:30002"
        r = requests.post(f"{router_env.url}/add_worker", json={"url": worker_url}, timeout=5.0)
        r.raise_for_status()

        assert r.json()["status"] == "success"
        assert worker_url in router_env.router.worker_request_counts

    def test_add_worker_duplicate(self, router_env: RouterEnv):
        worker_url = "http://127.0.0.1:30003"
        requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()
        requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()

        assert len(router_env.router.worker_request_counts) == 1
        assert worker_url in router_env.router.worker_request_counts

    def test_add_worker_missing_url(self, router_env: RouterEnv):
        r = requests.post(f"{router_env.url}/add_worker", json={}, timeout=5.0)
        assert r.status_code == 400
        assert "error" in r.json()

    def test_list_workers(self, router_env: RouterEnv):
        worker_urls = ["http://127.0.0.1:30001", "http://127.0.0.1:30002"]
        for url in worker_urls:
            requests.post(f"{router_env.url}/add_worker", params={"url": url}, timeout=5.0)

        r = requests.get(f"{router_env.url}/list_workers", timeout=5.0)
        r.raise_for_status()
        assert set(r.json()["urls"]) == set(worker_urls)


class TestLoadBalancing:
    def test_use_url_selects_min_load(self, router_factory):
        router = router_factory()
        router.worker_request_counts = {"http://w1:8000": 5, "http://w2:8000": 2, "http://w3:8000": 8}

        selected = router._use_url()
        assert selected == "http://w2:8000"
        assert router.worker_request_counts["http://w2:8000"] == 3

    def test_use_url_excludes_dead_workers(self, router_factory):
        router = router_factory()
        router.worker_request_counts = {"http://w1:8000": 5, "http://w2:8000": 1, "http://w3:8000": 3}
        router.dead_workers = {"http://w2:8000"}

        selected = router._use_url()
        assert selected == "http://w3:8000"
        assert router.worker_request_counts["http://w3:8000"] == 4

    def test_use_url_raises_when_all_dead(self, router_factory):
        router = router_factory()
        router.worker_request_counts = {"http://w1:8000": 0}
        router.dead_workers = {"http://w1:8000"}

        with pytest.raises(RuntimeError, match="No healthy workers"):
            router._use_url()


class TestEngineMetrics:
    def test_aggregate_prometheus_metrics_adds_worker_label_and_normalizes_names(self):
        result = _aggregate_prometheus_metrics(
            [
                (
                    "http://worker:1234",
                    "\n".join(
                        [
                            "# HELP sglang:num_running_reqs The number of running requests",
                            "# TYPE sglang:num_running_reqs gauge",
                            'sglang:num_running_reqs{model_name="m"} 3',
                            "# TYPE sglang:generation_tokens_total counter",
                            "sglang:generation_tokens_total 10",
                        ]
                    ),
                )
            ]
        )

        assert "# HELP sglang_num_running_reqs The number of running requests" in result
        assert "# TYPE sglang_num_running_reqs gauge" in result
        assert 'sglang_num_running_reqs{model_name="m",worker_addr="http://worker:1234"} 3' in result
        assert 'sglang_generation_tokens_total{worker_addr="http://worker:1234"} 10' in result

    def test_aggregate_prometheus_metrics_deduplicates_metadata(self):
        metric_text = "\n".join(
            [
                "# HELP sglang:num_queue_reqs queued",
                "# TYPE sglang:num_queue_reqs gauge",
                'sglang:num_queue_reqs{model_name="m"} 1',
            ]
        )

        result = _aggregate_prometheus_metrics(
            [
                ("http://worker1:30000", metric_text),
                ("http://worker2:30000", metric_text.replace(" 1", " 2")),
            ]
        )

        assert result.count("# HELP sglang_num_queue_reqs queued") == 1
        assert result.count("# TYPE sglang_num_queue_reqs gauge") == 1
        assert 'worker_addr="http://worker1:30000"' in result
        assert 'worker_addr="http://worker2:30000"' in result

    def test_aggregate_prometheus_metrics_handles_escaped_labels_and_invalid_lines(self):
        result = _aggregate_prometheus_metrics(
            [
                (
                    'http://worker\\"1:30000',
                    "\n".join(
                        [
                            'sglang:func_latency_seconds_bucket{name="a}b",le="+Inf"} 12',
                            "not_a_metric no_numeric_value",
                        ]
                    ),
                )
            ]
        )

        assert (
            'sglang_func_latency_seconds_bucket{name="a}b",le="+Inf",'
            'worker_addr="http://worker\\\\\\"1:30000"} 12'
        ) in result
        assert "not_a_metric" not in result

    def test_engine_metrics_scrapes_multiple_workers(self, router_env: RouterEnv):
        worker1 = create_metrics_worker(
            "\n".join(
                [
                    "# HELP sglang:num_running_reqs running",
                    "# TYPE sglang:num_running_reqs gauge",
                    'sglang:num_running_reqs{model_name="m"} 3',
                ]
            ),
            start_port=31000,
        )
        worker2 = create_metrics_worker(
            "\n".join(
                [
                    "# HELP sglang:num_running_reqs running",
                    "# TYPE sglang:num_running_reqs gauge",
                    'sglang:num_running_reqs{model_name="m"} 7',
                ]
            ),
            start_port=31100,
        )
        try:
            requests.post(f"{router_env.url}/add_worker", params={"url": worker1.url}, timeout=5.0).raise_for_status()
            requests.post(f"{router_env.url}/add_worker", params={"url": worker2.url}, timeout=5.0).raise_for_status()

            response = requests.get(f"{router_env.url}/engine_metrics", timeout=5.0)
            response.raise_for_status()
            text = response.text

            assert 'sglang_num_running_reqs{model_name="m",worker_addr="' in text
            assert f'worker_addr="{worker1.url}"' in text
            assert f'worker_addr="{worker2.url}"' in text
            assert text.count("# TYPE sglang_num_running_reqs gauge") == 1
        finally:
            worker1.stop()
            worker2.stop()

    def test_engine_metrics_skips_failed_workers(self, router_env: RouterEnv):
        worker = create_metrics_worker("# TYPE sglang:token_usage gauge\nsglang:token_usage 0.5\n")
        bad_worker_url = "http://127.0.0.1:59999"
        try:
            requests.post(f"{router_env.url}/add_worker", params={"url": worker.url}, timeout=5.0).raise_for_status()
            requests.post(f"{router_env.url}/add_worker", params={"url": bad_worker_url}, timeout=5.0).raise_for_status()

            response = requests.get(f"{router_env.url}/engine_metrics", timeout=5.0)
            response.raise_for_status()

            assert f'worker_addr="{worker.url}"' in response.text
            assert bad_worker_url not in response.text
        finally:
            worker.stop()

    def test_engine_metrics_returns_503_with_no_workers(self, router_env: RouterEnv):
        response = requests.get(f"{router_env.url}/engine_metrics", timeout=5.0)

        assert response.status_code == 503
        assert response.json()["error"] == "No available workers"

    def test_engine_metrics_returns_503_when_all_workers_fail(self, router_env: RouterEnv):
        requests.post(
            f"{router_env.url}/add_worker",
            params={"url": "http://127.0.0.1:59999"},
            timeout=5.0,
        ).raise_for_status()

        response = requests.get(f"{router_env.url}/engine_metrics", timeout=5.0)

        assert response.status_code == 503
        assert response.json()["error"] == "All backend metrics requests failed"


# TODO: extract main body inside `_health_check_loop`, then can test that function
class TestHealthCheck:
    def test_check_worker_health_success(self, router_factory, mock_worker):
        router = router_factory()
        url, healthy = asyncio.run(router._check_worker_health(mock_worker.url))
        assert url == mock_worker.url
        assert healthy is True

    def test_check_worker_health_failure(self, router_factory):
        router = router_factory()
        url, healthy = asyncio.run(router._check_worker_health("http://127.0.0.1:59999"))
        assert url == "http://127.0.0.1:59999"
        assert healthy is False


class TestProxyIntegration:
    def test_proxy_forwards_request(self, router_env: RouterEnv, mock_worker):
        requests.post(f"{router_env.url}/add_worker", params={"url": mock_worker.url}, timeout=5.0).raise_for_status()

        payload = {"input_ids": [1, 2, 3], "return_logprob": True}
        r = requests.post(f"{router_env.url}/generate", json=payload, timeout=10.0)
        r.raise_for_status()

        assert "text" in r.json()
        assert len(mock_worker.request_log) == 1
        assert mock_worker.request_log[0] == payload

    def test_proxy_multi_worker(self, router_env: RouterEnv, mock_worker_factory):
        worker1, worker2 = mock_worker_factory(), mock_worker_factory()
        requests.post(f"{router_env.url}/add_worker", params={"url": worker1.url}, timeout=5.0)
        requests.post(f"{router_env.url}/add_worker", params={"url": worker2.url}, timeout=5.0)

        payload = {"input_ids": [1, 2, 3], "return_logprob": True}
        for _ in range(4):
            requests.post(f"{router_env.url}/generate", json=payload, timeout=10.0).raise_for_status()

        all_requests = worker1.request_log + worker2.request_log
        assert len(all_requests) == 4
        assert all(req == payload for req in all_requests)

    def test_proxy_health_endpoint(self, router_env: RouterEnv, mock_worker):
        requests.post(f"{router_env.url}/add_worker", params={"url": mock_worker.url}, timeout=5.0)

        r = requests.get(f"{router_env.url}/health", timeout=5.0)
        r.raise_for_status()
        assert r.json()["status"] == "ok"
