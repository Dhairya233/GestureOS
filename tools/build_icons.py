"""
tools/build_icons.py
--------------------
Developer tool: turn Lucide SVG icons into the white 96x96 PNGs the UI tints at
runtime (ui/icons.py).  Windows users never need to run this; the PNGs are
committed.

    npm pack lucide-static            # then extract it somewhere
    pip install playwright && playwright install chromium
    python tools/build_icons.py <path-to>/package/icons

Lucide is ISC-licensed (see hand_gesture_system/ui/assets/LICENSE-lucide.txt).
"""
import os
import sys

ICONS = [
    "radio", "hand", "sliders-horizontal", "pen-line", "activity", "power",
    "shield-check", "shield-off", "camera", "camera-off", "triangle-alert",
    "circle-alert", "rotate-ccw", "check", "terminal", "eraser", "circle-check", "x",
]
STROKE = 1.8
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "hand_gesture_system", "ui", "assets", "icons")


def main(src: str) -> None:
    from playwright.sync_api import sync_playwright
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 96, "height": 96})
        for name in ICONS:
            svg = open(os.path.join(src, f"{name}.svg"), encoding="utf-8").read()
            svg = svg.replace('stroke="currentColor"', 'stroke="#ffffff"')
            svg = svg.replace('stroke-width="2"', f'stroke-width="{STROKE}"')
            svg = svg.replace('width="24"', 'width="96"').replace('height="24"', 'height="96"')
            page.set_content(f"<body style='margin:0;background:transparent'>{svg}</body>")
            page.locator("svg").screenshot(path=os.path.join(OUT, f"{name}.png"), omit_background=True)
            print("built", name)
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1])
