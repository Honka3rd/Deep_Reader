"""Deterministic, credential-free startup probe and safe diagnostics checks."""

from contextlib import redirect_stdout, redirect_stderr
import io
import ssl
import unittest
from unittest.mock import patch

import httpx

from scripts import check_llm_connectivity as probe


class ConnectivityTests(unittest.TestCase):
    def check(self, *, status=401, error=None, proxy=False):
        requests = []
        def respond(request):
            requests.append(request)
            if error:
                raise error
            return httpx.Response(status)
        client = httpx.Client(transport=httpx.MockTransport(respond))
        output = io.StringIO()
        with patch.object(probe, "getproxies_environment", return_value={"https": "http://user:secret@proxy:1234"} if proxy else {}), \
             patch.object(probe, "proxy_bypass_environment", return_value=False), \
             patch.object(probe.httpx, "Client", return_value=client) as constructor, \
             redirect_stdout(output), redirect_stderr(output):
            result = probe.main()
        constructor.assert_called_once_with(timeout=8.0, trust_env=True, follow_redirects=False)
        self.assertEqual(len(requests), 1)
        self.assertEqual(str(requests[0].url), probe.PROBE_URL)
        self.assertNotIn("authorization", requests[0].headers)
        self.assertNotIn("proxy-authorization", requests[0].headers)
        self.assertNotIn("secret", output.getvalue())
        return result, output.getvalue()

    def test_direct_and_proxy_without_credentials(self):
        for proxy in (False, True):
            for status in (200, 401):
                with self.subTest(proxy=proxy, status=status):
                    result, output = self.check(status=status, proxy=proxy)
                    self.assertEqual(result, 0)
                    self.assertIn("configured proxy" if proxy else "direct", output)
                    self.assertIn("API key and model access were not checked", output)

    def test_timeout_never_retries_or_falls_back(self):
        for proxy in (False, True):
            result, output = self.check(error=httpx.ConnectTimeout("secret"), proxy=proxy)
            self.assertEqual(result, 1)
            self.assertIn("connection_timeout", output)
            self.assertIn("DEEP_READER_CONTAINER_PROXY", output)

    def test_proxy_error_is_sanitized(self):
        result, output = self.check(error=httpx.ProxyError("http://user:secret@proxy"), proxy=True)
        self.assertEqual(result, 1)
        self.assertIn("proxy_connection_failed", output)

    def test_read_timeout_and_tls_failure(self):
        result, output = self.check(error=httpx.ReadTimeout("secret"))
        self.assertEqual(result, 1)
        self.assertIn("response_timeout", output)
        error = httpx.ConnectError("secret")
        error.__cause__ = ssl.SSLCertVerificationError("secret")
        result, output = self.check(error=error)
        self.assertEqual(result, 1)
        self.assertIn("tls_verification_failed", output)

    def test_unexpected_status_is_not_success(self):
        for status, reason in ((302, "unexpected_http_status"), (403, "access_denied"),
                               (407, "proxy_authentication_required"), (429, "upstream_rate_limited"),
                               (503, "upstream_unavailable")):
            with self.subTest(status=status):
                result, output = self.check(status=status)
                self.assertEqual(result, 1)
                self.assertIn(reason, output)

    def test_invalid_proxy_configuration_is_sanitized(self):
        output = io.StringIO()
        with patch.object(probe, "getproxies_environment", return_value={"https": "invalid"}), \
             patch.object(probe, "proxy_bypass_environment", return_value=False), \
             patch.object(probe.httpx, "Client", side_effect=ValueError("secret")) as constructor, \
             redirect_stdout(output), redirect_stderr(output):
            self.assertEqual(probe.main(), 1)
        constructor.assert_called_once()
        self.assertNotIn("secret", output.getvalue())
        self.assertIn("invalid_proxy_or_client_configuration", output.getvalue())


if __name__ == "__main__":
    unittest.main()
