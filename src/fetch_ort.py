"""Download onnxruntime-web into web/vendor/ so the page needs no CDN.

    python src/fetch_ort.py

Loading the runtime from a CDN turned out to be fragile: the loader
fetches its .wasm engine separately, and when that request fails the page
reports "no available backend found ... BufferSource argument is empty" -
which reads like a model problem but is really a missing download.

Vendoring also means the demo runs with no internet at all, which is the
difference between a presentation that works and one that depends on the
room's wifi.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request

from config import ROOT

VERSION = "1.19.2"
VENDOR = ROOT / "web" / "vendor"

# Tried in order; the first host that serves a file wins.
BASES = [
    f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{VERSION}/dist/",
    f"https://unpkg.com/onnxruntime-web@{VERSION}/dist/",
    f"https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/{VERSION}/",
]

# ort.wasm.min.js is the WebAssembly-only build: no WebGPU, so it needs
# only the plain threaded wasm and not the larger "jsep" variant.
FILES = [
    ("ort.wasm.min.js", True, 20_000),
    ("ort-wasm-simd-threaded.wasm", True, 1_000_000),
    ("ort-wasm-simd-threaded.mjs", True, 5_000),
]


def download(name: str, minimum: int) -> tuple[bool, str]:
    for base in BASES:
        url = base + name
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"    {base.split('/')[2]}: {e}")
            continue

        # A CDN that does not have the file often answers 200 with an HTML
        # error page. Size is the cheap way to catch that.
        if len(data) < minimum:
            print(f"    {base.split('/')[2]}: only {len(data)} bytes, "
                  f"expected at least {minimum} - looks like an error page")
            continue

        (VENDOR / name).write_bytes(data)
        return True, f"{len(data) / 1024:.0f} KB from {base.split('/')[2]}"
    return False, "not found on any host"


def main() -> None:
    VENDOR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading onnxruntime-web {VERSION} -> web/vendor/\n")

    failures = []
    for name, required, minimum in FILES:
        existing = VENDOR / name
        if existing.exists() and existing.stat().st_size >= minimum:
            print(f"  {name:<34} already present "
                  f"({existing.stat().st_size / 1024:.0f} KB)")
            continue
        print(f"  {name}")
        ok, detail = download(name, minimum)
        print(f"    -> {detail}")
        if not ok and required:
            failures.append(name)

    if failures:
        print("\nCould not download: " + ", ".join(failures))
        print("Check the internet connection, or download them by hand from")
        print(f"  {BASES[0]}")
        print("and drop them into web/vendor/.")
        sys.exit(1)

    print("\nDone. The page now loads the runtime from web/vendor/ "
          "and works offline.")


if __name__ == "__main__":
    main()
