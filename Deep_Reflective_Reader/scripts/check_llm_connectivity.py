"""Explicit local-startup network probe for the current OpenAI-backed provider.

No credentials, generation, retries, route fallback or persistence. This is not
an API healthcheck: offline reading services remain available after failure.
"""

import ssl
import sys
from time import monotonic
from urllib.request import getproxies_environment, proxy_bypass_environment

import httpx


API_HOST = "api.openai.com"
PROBE_URL = f"https://{API_HOST}/v1/models"


def main():
    proxies = getproxies_environment()
    use_proxy = bool(proxies.get("https") or proxies.get("all")) and not proxy_bypass_environment(API_HOST, proxies)
    route = "configured proxy" if use_proxy else "direct"
    print(f"Model API network check: {route}; one attempt, 8-second timeout per network phase.", flush=True)
    started = monotonic()
    reason = None
    try:
        with httpx.Client(timeout=8.0, trust_env=True, follow_redirects=False) as client:
            with client.stream("GET", PROBE_URL) as response:
                status = response.status_code
        if status in (200, 401):
            print(f"Model API reachable via {route} (HTTP {status}, {monotonic() - started:.1f}s). "
                  "Connectivity only; API key and model access were not checked.")
            return 0
        reason = ("proxy_authentication_required" if status == 407 else
                  "access_denied" if status == 403 else
                  "upstream_rate_limited" if status == 429 else
                  "upstream_unavailable" if status >= 500 else "unexpected_http_status")
        print(f"Model API network check failed: {reason} (HTTP {status}).", file=sys.stderr)
    except (httpx.RequestError, httpx.InvalidURL, ValueError, ImportError) as error:
        if isinstance(error, httpx.ConnectTimeout):
            reason = "connection_timeout"
        elif isinstance(error, httpx.TimeoutException):
            reason = "response_timeout"
        elif isinstance(error, httpx.ProxyError):
            reason = "proxy_connection_failed"
        elif isinstance(error, (httpx.InvalidURL, ValueError, ImportError)):
            reason = "invalid_proxy_or_client_configuration"
        else:
            reason = "connection_failed"
            cause = error
            while cause is not None:
                if isinstance(cause, ssl.SSLError):
                    reason = "tls_verification_failed"
                    break
                cause = cause.__cause__
        # Exception text may contain proxy credentials or private network details.
        print(f"Model API network check failed via {route}: {reason} "
              f"({monotonic() - started:.1f}s).", file=sys.stderr)
    if use_proxy:
        print("Check that your configured proxy is running and reachable from Docker. "
              "Update DEEP_READER_CONTAINER_PROXY for a different proxy, or clear it only "
              "when direct access is intended and available.", file=sys.stderr)
    else:
        print("Check Docker DNS, routing and TLS access to the model API. If a proxy is required, "
              "set DEEP_READER_CONTAINER_PROXY explicitly in .env.local.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
