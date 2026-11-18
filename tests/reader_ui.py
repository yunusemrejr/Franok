"""Exercise source selection, highlighting, persistence and mobile layout in the actual WASM UI."""

import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from playwright.sync_api import sync_playwright, expect


@contextmanager
def fixture(export, head, config):
    spec = json.loads((head / "config.json").read_text())
    with TemporaryDirectory(prefix="franok-reader-ui-") as folder:
        root = Path(folder)
        for p in Path("web").iterdir():
            if p.name not in ("model.json", "release.json", "reader.safetensors"):
                (root / p.name).symlink_to(p.resolve())
        (root / "model.json").symlink_to((export / "model.json").resolve())
        (root / "fixture.q4").symlink_to((export / "model.q4").resolve())
        weights = head / "head.safetensors"
        (root / "reader.safetensors").symlink_to(weights.resolve())
        release = json.loads(Path("web/release.json").read_text())
        release["formats"]["q4"]["url"] = "./fixture.q4"
        # The test fixture exercises both possible decisions independently of task accuracy.
        release["reader"] = dict(
            **spec,
            url="./reader.safetensors",
            baseSha256=release["formats"]["fp16"]["sha256"],
            sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        )
        release["reader"]["threshold"] = config
        (root / "release.json").write_text(json.dumps(release))
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=folder)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/"
        finally:
            server.shutdown()
            server.server_close()


def verify(browser, url, output, expect_answer):
    page = browser.new_page(viewport=dict(width=1100, height=780))
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url)
    page.locator("#lang").select_option("tr")
    page.locator("#readerOpen").click()
    expect(page.get_by_role("dialog", name="Metin hakkında soru sor")).to_be_visible()
    passage = (
        "Salonun kapısındaki küçük kartta IŞIK yazıyor. Toplantı cuma sabahı başlıyor."
    )
    page.locator("#sourceText").fill(passage)
    page.locator("#sourceQuestion").fill("Kartta ne yazıyor?")
    page.screenshot(path=output / "reader-form.png")
    page.locator("#readSource").click()
    expect(page.locator("#status")).to_have_text(
        "Metin bu cihazda okundu.", timeout=250000
    )
    result = page.locator(".message.assistant").last
    answer = result.locator(".body").inner_text()
    accepted = bool(result.locator("blockquote").count())
    if expect_answer is not None:
        assert accepted == expect_answer, (accepted, answer)
    if accepted:
        assert answer and answer in passage
        result.locator("summary").click()
        expect(result.locator("mark")).to_have_text(answer)
        assert result.locator(".sourcepassage p").inner_text() == passage
    else:
        assert "bulamadı" in answer
    page.reload()
    expect(page.locator(".message.assistant .body")).to_have_text(answer)
    page.set_viewport_size(dict(width=360, height=740))
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.locator("#readerOpen").click()
    expect(page.locator("#readSource")).to_be_in_viewport()
    page.screenshot(path=output / "reader-mobile.png")
    page.keyboard.press("Escape")
    expect(page.locator("#readingDialog")).not_to_be_visible()
    assert not errors, errors
    page.close()
    return dict(accepted=accepted, answer=answer)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--export", type=Path)
    p.add_argument("--head", type=Path)
    p.add_argument("--url")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path="/usr/bin/google-chrome")
        receipts = []
        if a.url:
            receipts.append(verify(browser, a.url, a.out, None))
        else:
            for threshold, accepted in ((-1e6, True), (1e6, False)):
                directory = a.out / ("quote" if accepted else "abstain")
                directory.mkdir(exist_ok=True)
                with fixture(a.export, a.head, threshold) as url:
                    receipts.append(verify(browser, url, directory, accepted))
        browser.close()
        (a.out / "results.json").write_text(
            json.dumps(receipts, ensure_ascii=False, indent=2) + "\n"
        )
        print(receipts, flush=True)
