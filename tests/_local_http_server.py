"""
Minimal local HTTP server for testing resumable downloads, used by
test_download_resume.py. Not a test file itself (no test_ functions).
"""
import http.server
import socketserver
import threading


def make_server(content: bytes, drop_after_bytes=None, support_range=True):
    """Serves `content` at '/file'.

    drop_after_bytes: if set, closes the connection after sending that many
    bytes on the FIRST request only (simulating a dropped connection).

    support_range: if False, ignores Range headers and always returns the
    full content with a 200 (simulating a server/CDN that doesn't support
    resumable downloads).
    """
    state = {"first_request_done": False}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            total = len(content)
            range_header = self.headers.get("Range") if support_range else None

            if range_header:
                start = int(range_header.replace("bytes=", "").split("-")[0])
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{total-1}/{total}")
                self.send_header("Content-Length", str(total - start))
                self.end_headers()
                body = content[start:]
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(total))
                self.end_headers()
                body = content

            if drop_after_bytes is not None and not state["first_request_done"]:
                state["first_request_done"] = True
                self.wfile.write(body[:drop_after_bytes])
                self.connection.close()
                return
            self.wfile.write(body)

    httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port
