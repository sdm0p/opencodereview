"""Choose a reachable Cloudflare IPv4 route while retaining TLS verification."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import socket
import ssl
import threading
import time

import httpx


HOST = "api.cloudflare.com"
logger = logging.getLogger(__name__)


class ReachableCloudflareRoutes:
    """Resolve normally and cache only addresses with a verified TLS handshake."""

    def __init__(self):
        self._lock = threading.Lock()
        self._routes = []
        self._expires = 0.0

    @staticmethod
    def _probe(address):
        try:
            with socket.create_connection((address, 443), timeout=3) as conn:
                with ssl.create_default_context().wrap_socket(conn, server_hostname=HOST):
                    return address
        except OSError:
            return None

    def select(self, excluded=()):
        with self._lock:
            now = time.monotonic()
            routes = [item for item in self._routes if item not in excluded] if now < self._expires else []
            if routes:
                return routes[0]
            addresses = list(dict.fromkeys(item[4][0] for item in socket.getaddrinfo(HOST, 443, socket.AF_INET, socket.SOCK_STREAM)))[:12]
            addresses = [item for item in addresses if item not in excluded]
            with ThreadPoolExecutor(max_workers=max(1, len(addresses))) as pool:
                futures = [pool.submit(self._probe, address) for address in addresses]
                routes = [route for future in as_completed(futures) if (route := future.result())]
            self._routes = routes
            self._expires = time.monotonic() + 60
            logger.info("Cloudflare route check: %d candidates, %d verified", len(addresses), len(routes))
            if not routes:
                raise httpx.ConnectError("No verified Cloudflare IPv4 route was reachable")
            return routes[0]

    def invalidate(self, address):
        with self._lock:
            self._routes = [route for route in self._routes if route != address]


def _routed_request(request, address):
    if request.url.scheme != "https" or request.url.host != HOST:
        raise ValueError("Cloudflare transport only permits the official HTTPS API host")
    headers = httpx.Headers(request.headers)
    headers["Host"] = HOST
    extensions = dict(request.extensions)
    extensions["sni_hostname"] = HOST
    return httpx.Request(request.method, request.url.copy_with(host=address), headers=headers,
                         stream=request.stream, extensions=extensions)


class CloudflareTransport(httpx.BaseTransport):
    def __init__(self, routes):
        self.routes = routes
        self.transport = httpx.HTTPTransport(local_address="0.0.0.0", retries=0)

    def handle_request(self, request):
        excluded = []
        for attempt in range(3):
            try:
                address = self.routes.select(excluded)
            except httpx.ConnectError:
                if attempt == 2:
                    raise
                time.sleep(2)
                continue
            try:
                return self.transport.handle_request(_routed_request(request, address))
            except (httpx.ConnectError, httpx.ConnectTimeout):
                self.routes.invalidate(address)
                excluded.append(address)
                if attempt == 2:
                    raise

    def close(self):
        self.transport.close()


class AsyncCloudflareTransport(httpx.AsyncBaseTransport):
    def __init__(self, routes):
        self.routes = routes
        self.transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0", retries=0)

    async def handle_async_request(self, request):
        excluded = []
        for attempt in range(3):
            try:
                address = await asyncio.to_thread(self.routes.select, excluded)
            except httpx.ConnectError:
                if attempt == 2:
                    raise
                await asyncio.sleep(2)
                continue
            try:
                return await self.transport.handle_async_request(_routed_request(request, address))
            except (httpx.ConnectError, httpx.ConnectTimeout):
                await asyncio.to_thread(self.routes.invalidate, address)
                excluded.append(address)
                if attempt == 2:
                    raise

    async def aclose(self):
        await self.transport.aclose()
