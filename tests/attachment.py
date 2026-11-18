"""Exercise a late search hit, saved attachment and actual model context in a browser."""

import argparse, json
import threading
from tempfile import TemporaryDirectory
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def verify(url, output):
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path="/usr/bin/google-chrome")
        ctx = browser.new_context(viewport={"width": 1000, "height": 760})
        turns = [
            dict(
                role="user",
                text="Genel notlar. " * 260 + "IŞIK numarası K99-71. Gün tamamlandı.",
            )
        ] + [
            dict(
                role=("user", "assistant")[i % 2],
                text="Genel notlar, sayfa " + str(i + 1) + ".",
            )
            for i in range(18)
        ]
        legacy = [
            dict(
                id="long-source-test",
                lang="tr",
                turns=turns,
                options=dict(maxNew=8, temperature=0),
            )
        ]
        script = (
            "if(!localStorage.getItem('fixture-init')){localStorage.setItem('fixture-init','1');localStorage.setItem('franok.sessions.v1',"
            + json.dumps(json.dumps(legacy))
            + ");localStorage.setItem('franok.active.v1','long-source-test')}"
        )
        ctx.add_init_script(script)
        page = ctx.new_page()
        page.goto(url)
        expect(
            page.get_by_role("button", name="Pin turn 1", exact=True)
        ).to_be_visible()
        page.locator("#find").click()
        page.locator("#query").fill("genel ışık")
        expect(page.locator("#matches .matchrow").first).to_contain_text(
            "IŞIK numarası K99-71"
        )
        page.get_by_role("button", name="Attach turn 1 to next message").click()
        page.reload()
        expect(page.locator("#quotechip")).to_be_visible()
        page.locator("#draft").fill("Notta hangi numara yazıyordu?")
        page.locator("#send").click()
        expect(page.locator("#send")).to_be_enabled(timeout=180000)
        last = page.locator(".message.assistant").last
        expect(last.locator("details")).to_be_visible(timeout=30000)
        last.locator("details summary").click()
        expect(
            last.get_by_role("button", name="Turn 1 · excerpt", exact=True)
        ).to_be_visible()
        receipt = dict(
            answer=last.locator(".body").inner_text(),
            context=last.locator("details").inner_text(),
        )
        last.get_by_role("button", name="Turn 1 · excerpt", exact=True).click()
        expect(page.locator("#turn-0")).to_have_class(__import__("re").compile("hit"))
        page.locator("#find").click()
        page.locator("#query").fill("ışık")
        page.screenshot(path=str(output / "search.png"))
        page.set_viewport_size({"width": 390, "height": 820})
        expect(page.locator("#matches .matchrow").first).to_contain_text("K99-71")
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        page.screenshot(path=str(output / "search-mobile.png"))
        (output / "result.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
        )
        print(receipt)
        browser.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("output", type=Path)
    p.add_argument("--weights", type=Path)
    a = p.parse_args()
    if a.weights:
        with TemporaryDirectory(prefix="franok-attachment-site-") as directory:
            root = Path(directory)
            for f in Path("web").iterdir():
                if f.name != "release.json":
                    (root / f.name).symlink_to(f.resolve())
            manifest = json.loads(Path("web/release.json").read_text())
            manifest["formats"]["q4"]["url"] = "./fixture.q4"
            (root / "fixture.q4").symlink_to(a.weights.resolve())
            (root / "release.json").write_text(json.dumps(manifest))
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=directory)
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                verify(f"http://127.0.0.1:{server.server_port}/", a.output)
            finally:
                server.shutdown()
                server.server_close()
    else:
        verify(a.url, a.output)
