import argparse
import asyncio
import json
import logging
import re
from collections import OrderedDict
from collections.abc import Mapping

import httpx
import setproctitle
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

logger = logging.getLogger(__name__)

_PROMETHEUS_CONTENT_TYPE = "text/plain; charset=utf-8"
_METRIC_NAME_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)")
_METRICS_REQUEST_TIMEOUT = 5.0
_METRICS_MAX_CONCURRENT = 32


def run_router(args):
    """
    Run the Miles router with the specified configuration.
    """
    # Visible to `pkill -9 miles`; without this the daemon inherits "python".
    setproctitle.setproctitle("miles-router")

    # Initialize the router with tokenizer and lazy worker initialization
    miles_router = MilesRouter(args, verbose=False)

    # Start the server
    uvicorn.run(miles_router.app, host=args.sglang_router_ip, port=args.sglang_router_port, log_level="info")


class MilesRouter:
    def __init__(self, args, verbose=False):
        """Initialize the miles-router with SGLang router address"""
        self.args = args
        self.verbose = verbose

        self.app = FastAPI()
        self.app.router.on_startup.append(self._start_background_health_check)

        # URL -> Active Request Count (load state)
        self.worker_request_counts: dict[str, int] = {}
        # URL -> Consecutive Failures
        self.worker_failure_counts: dict[str, int] = {}
        # Quarantined workers excluded from routing pool
        self.dead_workers: set[str] = set()
        self.router_policy = getattr(args, "miles_router_policy", "least_loaded")
        if self.router_policy not in ("least_loaded", "round_robin"):
            raise ValueError(
                f"Unsupported miles_router_policy={self.router_policy!r}; "
                "expected one of: least_loaded, round_robin"
            )
        self.sticky_routing = getattr(args, "miles_router_sticky_routing", False)
        self.routing_key_header = getattr(args, "miles_router_routing_key_header", "X-Miles-Routing-Key")
        self.sticky_routing_max_keys = getattr(args, "miles_router_sticky_routing_max_keys", 100000)
        if self.sticky_routing and self.sticky_routing_max_keys <= 0:
            raise ValueError("miles_router_sticky_routing_max_keys must be positive when sticky routing is enabled")
        self._round_robin_index = 0
        self._routing_table: OrderedDict[str, str] = OrderedDict()
        self.max_weight_version = None

        max_connections = getattr(args, "miles_router_max_connections", None)
        if max_connections is None:
            max_connections = (
                args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
            )

        timeout = getattr(args, "miles_router_timeout", None)

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(timeout),
        )

        self._setup_routes()

        for middleware_path in args.miles_router_middleware_paths or []:
            from miles.utils.misc import load_function

            if self.verbose:
                print(f"[miles-router] Loading middleware from: {middleware_path}")
            middleware = load_function(middleware_path)
            self.app.add_middleware(middleware, router=self)

    def _setup_routes(self):
        """Setup all the HTTP routes except catch-all proxy"""
        # sglang-router api
        self.app.post("/add_worker")(self.add_worker)
        self.app.get("/list_workers")(self.list_workers)
        self.app.get("/engine_metrics")(self.engine_metrics)
        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    async def _start_background_health_check(self):
        asyncio.create_task(self._health_check_loop())

    async def _check_worker_health(self, url):
        """Encapsulated health check logic for better maintainability."""
        try:
            response = await self.client.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                return url, True
            logger.debug(f"[miles-router] Worker {url} is unhealthy (Status: {response.status_code})")
        except Exception as e:
            logger.debug(f"[miles-router] Worker {url} health check failed: {e}")
        return url, False

    async def _health_check_loop(self):
        """Background loop to monitor worker health and adjust routing pool."""
        interval = self.args.rollout_health_check_interval
        threshold = self.args.miles_router_health_check_failure_threshold

        while True:
            try:
                await asyncio.sleep(interval)

                urls = [u for u in self.worker_request_counts if u not in self.dead_workers]
                if not urls:
                    continue

                results = await asyncio.gather(*(self._check_worker_health(url) for url in urls))

                for url, is_healthy in results:
                    if not is_healthy:
                        failures = self.worker_failure_counts.get(url, 0) + 1
                        self.worker_failure_counts[url] = failures

                        if failures >= threshold:
                            logger.warning(
                                f"[miles-router] Worker {url} failed {threshold} consecutive health checks. Marking as DEAD."
                            )
                            self.dead_workers.add(url)
                            # TODO (chenyang): Connect back 'dead' workers requires a mechanism to sync
                            # model versions to avoid off-policy issues from stale weights, since these
                            # dead workers' parameters may not be refitted.
                    else:
                        self.worker_failure_counts[url] = 0

                logger.debug(
                    f"[miles-router] Health check complete. {len(self.worker_request_counts) - len(self.dead_workers)} workers healthy."
                )

            except asyncio.CancelledError:
                logger.warning("[miles-router] Background health check loop is being cancelled.")
                raise
            except Exception as e:
                logger.error(f"[miles-router] Unexpected error in health check loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router"""
        result = await self.do_proxy(request, path)
        return self.build_proxy_response(result)

    async def do_proxy(
        self,
        request: Request,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
    ) -> dict:
        """Core proxy logic. Returns dict with request_body, response_body, status_code, headers."""
        if headers is None:
            headers = dict(request.headers)
        routing_key = self._get_routing_key(headers)

        worker_url = self._use_url(routing_key=routing_key)
        url = f"{worker_url}/{path}"

        if body is None:
            body = await request.body()
        if body is not None:
            headers = {k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding")}

        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
            content = await response.aread()
            return {
                "request_body": body,
                "response_body": content,
                "status_code": response.status_code,
                "headers": dict(response.headers),
            }
        finally:
            self._finish_url(worker_url)

    def build_proxy_response(self, result: dict) -> Response:
        """Build HTTP response from proxy result."""
        content = result["response_body"]
        status_code = result["status_code"]
        headers = result["headers"]
        headers = {k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding")}
        content_type = headers.get("content-type", "")
        try:
            data = json.loads(content)
            return JSONResponse(content=data, status_code=status_code, headers=headers)
        except Exception:
            return Response(content=content, status_code=status_code, headers=headers, media_type=content_type)

    async def add_worker(self, request: Request):
        """Add a new worker to the router.
        Supports providing the URL via query string or JSON body.
        Examples:
        - POST /add_worker?url=http://127.0.0.1:10090
        - POST /add_worker  with body {"url": "http://127.0.0.1:10090"}
        """
        # 1) Prefer query param
        worker_url = request.query_params.get("url") or request.query_params.get("worker_url")

        # 2) Fallback to JSON body
        if not worker_url:
            body = await request.body()
            payload = json.loads(body) if body else {}
            worker_url = payload.get("url") or payload.get("worker_url")

        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        # Add if new, keep a simple request count per worker
        if worker_url not in self.worker_request_counts:
            self.worker_request_counts[worker_url] = 0
            self.worker_failure_counts[worker_url] = 0
            if self.verbose:
                print(f"[miles-router] Added new worker: {worker_url}")

        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def list_workers(self, request: Request):
        """List all registered workers"""
        return {"urls": list(self.worker_request_counts.keys())}

    async def engine_metrics(self):
        """Aggregate worker /metrics endpoints like SGLang Model Gateway's /engine_metrics."""
        worker_urls = [url for url in self.worker_request_counts if url not in self.dead_workers]
        if not worker_urls:
            return JSONResponse(
                status_code=503,
                content={"error": "No available workers"},
            )

        semaphore = asyncio.Semaphore(_METRICS_MAX_CONCURRENT)
        results = await asyncio.gather(*(self._get_worker_metrics(url, semaphore) for url in worker_urls))
        metric_packs = [(url, text) for url, text in results if text is not None]
        if not metric_packs:
            return JSONResponse(
                status_code=503,
                content={"error": "All backend metrics requests failed"},
            )

        return Response(
            content=_aggregate_prometheus_metrics(metric_packs),
            headers={"content-type": _PROMETHEUS_CONTENT_TYPE},
        )

    async def _get_worker_metrics(self, worker_url: str, semaphore: asyncio.Semaphore) -> tuple[str, str | None]:
        try:
            async with semaphore:
                response = await self.client.get(f"{worker_url}/metrics", timeout=_METRICS_REQUEST_TIMEOUT)
            if response.status_code != 200:
                logger.debug(
                    "[miles-router] Worker %s metrics scrape failed (Status: %s)",
                    worker_url,
                    response.status_code,
                )
                return worker_url, None
            return worker_url, response.text
        except Exception as e:
            logger.debug("[miles-router] Worker %s metrics scrape failed: %s", worker_url, e)
            return worker_url, None

    def _healthy_workers(self) -> list[str]:
        return [w for w in self.worker_request_counts if w not in self.dead_workers]

    @staticmethod
    def _get_header(headers: Mapping[str, str], name: str) -> str | None:
        for key, value in headers.items():
            if key.lower() == name.lower():
                return value
        return None

    def _get_routing_key(self, headers: Mapping[str, str]) -> str | None:
        if not self.sticky_routing:
            return None

        routing_key = self._get_header(headers, self.routing_key_header)
        if routing_key is None and self.routing_key_header.lower() != "x-smg-routing-key":
            routing_key = self._get_header(headers, "X-SMG-Routing-Key")
        return routing_key

    def _select_round_robin(self, workers: list[str]) -> str:
        if not workers:
            raise RuntimeError("No healthy workers available in the pool")
        url = workers[self._round_robin_index % len(workers)]
        self._round_robin_index = (self._round_robin_index + 1) % len(workers)
        return url

    def _select_worker(self, workers: list[str]) -> str:
        if self.router_policy == "round_robin":
            return self._select_round_robin(workers)

        if not workers:
            raise RuntimeError("No healthy workers available in the pool")
        min_load = min(self.worker_request_counts[w] for w in workers)
        least_loaded_workers = [w for w in workers if self.worker_request_counts[w] == min_load]
        return self._select_round_robin(least_loaded_workers)

    def _remember_routing_key(self, routing_key: str, url: str) -> None:
        self._routing_table[routing_key] = url
        self._routing_table.move_to_end(routing_key)
        while len(self._routing_table) > self.sticky_routing_max_keys:
            self._routing_table.popitem(last=False)

    def _use_url(self, routing_key: str | None = None):
        """Select a worker URL according to the configured router policy."""

        workers = self._healthy_workers()
        if not workers:
            raise RuntimeError("No healthy workers available in the pool")

        if self.sticky_routing and routing_key:
            mapped_url = self._routing_table.get(routing_key)
            if mapped_url in workers:
                url = mapped_url
                self._routing_table.move_to_end(routing_key)
            else:
                url = self._select_worker(workers)
                self._remember_routing_key(routing_key, url)
        else:
            url = self._select_worker(workers)

        self.worker_request_counts[url] += 1
        return url

    def _finish_url(self, url):
        """Mark the request to the given URL as finished"""
        assert url in self.worker_request_counts, f"URL {url} not recognized"
        self.worker_request_counts[url] -= 1
        assert self.worker_request_counts[url] >= 0, f"URL {url} count went negative"


def _aggregate_prometheus_metrics(metric_packs: list[tuple[str, str]]) -> str:
    """Aggregate worker Prometheus text, preserving one series per worker."""
    output_lines = []
    seen_metadata = set()
    for worker_url, metrics_text in metric_packs:
        worker_label = _format_prometheus_label("worker_addr", worker_url)
        transformed = _transform_prometheus_metrics(metrics_text, worker_label, seen_metadata)
        if transformed:
            output_lines.extend(transformed)
    return "\n".join(output_lines) + ("\n" if output_lines else "")


def _transform_prometheus_metrics(metrics_text: str, worker_label: str, seen_metadata: set[tuple[str, str]]) -> list[str]:
    output_lines = []
    for line in metrics_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "# EOF":
            continue
        if stripped.startswith("#"):
            metadata_line = _transform_prometheus_metadata_line(line, seen_metadata)
            if metadata_line is not None:
                output_lines.append(metadata_line)
            continue
        sample_line = _add_worker_label_to_metric_sample(line, worker_label)
        if sample_line is not None:
            output_lines.append(sample_line)
    return output_lines


def _transform_prometheus_metadata_line(line: str, seen_metadata: set[tuple[str, str]]) -> str | None:
    parts = line.split(maxsplit=3)
    if len(parts) < 3 or parts[0] != "#" or parts[1] not in ("HELP", "TYPE", "UNIT"):
        return line

    kind = parts[1]
    metric_name = _normalize_prometheus_metric_name(parts[2])
    key = (kind, metric_name)
    if key in seen_metadata:
        return None
    seen_metadata.add(key)

    parts[2] = metric_name
    return " ".join(parts)


def _add_worker_label_to_metric_sample(line: str, worker_label: str) -> str | None:
    match = _METRIC_NAME_RE.match(line)
    if match is None:
        logger.debug("[miles-router] Skipping unparseable metrics sample line: %r", line)
        return None

    metric_name = match.group(1)
    normalized_metric_name = _normalize_prometheus_metric_name(metric_name)
    rest = line[match.end() :]

    if rest.startswith("{"):
        label_end = _find_prometheus_label_block_end(rest)
        if label_end == -1:
            logger.debug("[miles-router] Skipping metrics sample with unterminated label block: %r", line)
            return None
        existing_labels = rest[1:label_end]
        separator = "," if existing_labels else ""
        value_part = rest[label_end + 1 :]
        if not _has_prometheus_sample_value(value_part):
            logger.debug("[miles-router] Skipping metrics sample without a numeric value: %r", line)
            return None
        return f"{normalized_metric_name}{{{existing_labels}{separator}{worker_label}}}{value_part}"

    if not _has_prometheus_sample_value(rest):
        logger.debug("[miles-router] Skipping metrics sample without a numeric value: %r", line)
        return None
    return f"{normalized_metric_name}{{{worker_label}}}{rest}"


def _find_prometheus_label_block_end(text: str) -> int:
    escaped = False
    in_quotes = False
    for i, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_quotes = not in_quotes
            continue
        if char == "}" and not in_quotes:
            return i
    return -1


def _has_prometheus_sample_value(text: str) -> bool:
    parts = text.split(maxsplit=1)
    if not parts:
        return False
    try:
        float(parts[0])
    except ValueError:
        return parts[0] in ("NaN", "+Inf", "-Inf")
    return True


def _normalize_prometheus_metric_name(metric_name: str) -> str:
    return metric_name.replace(":", "_")


def _format_prometheus_label(name: str, value: str) -> str:
    return f'{name}="{_escape_prometheus_label_value(value)}"'


def _escape_prometheus_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--sglang-host", type=str, required=True)
    parser.add_argument("--sglang-port", type=int, required=True)
    parser.add_argument("--tokenizer-name", type=str, help="Name of the tokenizer to use for tokenization")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--miles-router-policy",
        type=str,
        choices=("least_loaded", "round_robin"),
        default="least_loaded",
        help="Routing policy for MilesRouter workers.",
    )
    parser.add_argument(
        "--miles-router-sticky-routing",
        action="store_true",
        default=False,
        help="Route requests with the same routing key header to the same healthy worker.",
    )
    parser.add_argument(
        "--miles-router-routing-key-header",
        type=str,
        default="X-Miles-Routing-Key",
        help="Header used as the sticky routing key.",
    )
    parser.add_argument(
        "--miles-router-sticky-routing-max-keys",
        type=int,
        default=100000,
        help="Maximum sticky routing keys retained by MilesRouter.",
    )

    args = parser.parse_args()

    # Run the router
    run_router(args)
