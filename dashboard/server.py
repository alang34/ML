import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def start_dashboard_server(dashboard_dir, port=8000):
    handler = partial(_QuietHandler, directory=str(dashboard_dir))
    httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"[dashboard] serving {dashboard_dir} at http://localhost:{port}/index.html")
    return httpd