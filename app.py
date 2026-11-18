"""Start the SignSpeak web app and open it in your browser.

  python app.py                 # http://127.0.0.1:8000
  python app.py --port 8080
  python app.py --no-browser

The server only listens on this computer (127.0.0.1). The browser asks for camera permission the
first time. Press Ctrl+C in this window to stop the server.
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Start the SignSpeak web app.")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = p.parse_args(argv)
    if not 1 <= args.port <= 65535:
        p.error("--port must be between 1 and 65535")
    if not port_free(args.port):
        print(f"Port {args.port} is already in use (is SignSpeak already running?). "
              f"Try: python app.py --port {args.port + 1}")
        return 1

    import uvicorn

    from signspeak.server import create_app

    url = f"http://127.0.0.1:{args.port}/"
    print(f"SignSpeak is running at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, (url,)).start()
    uvicorn.run(create_app(port=args.port), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
