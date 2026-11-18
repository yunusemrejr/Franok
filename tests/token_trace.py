"""Exercise the real tokenizer and routing diagram on English, Turkish and UTF-8 fragments."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from playwright.sync_api import expect, sync_playwright
from tokenizers import Tokenizer


def verify(browser, url, out, prediction=False):
    page = browser.new_page(viewport={"width": 1100, "height": 780})
    page.goto(url)
    page.locator("#architecture").click()
    field = page.locator("#traceText")
    expect(field).to_be_visible(timeout=45000)
    page.locator("#traceLanguage").select_option("tr")
    text = "Kahvenin yanında küçük bir dilim 🍰 ister misin?"
    field.fill(text)
    tokenizer = Tokenizer.from_file("web/tokenizer.json")
    ids = [1, 6, *tokenizer.encode(text).ids, 2]
    items = page.locator("#tracePieces button")
    expect(items).to_have_count(len(ids))
    for i, id_ in enumerate(ids):
        assert f"token ID {id_}, position {i + 1}" in items.nth(i).get_attribute(
            "aria-label"
        )
    i = next(i for i in range(8, len(ids)) if ids[i // 8 * 8 - 1] > 63)
    items.nth(i).click()
    measured = None
    if prediction:
        button = page.locator("#predictPiece")
        button.click()
        items.nth(i + 1).click()
        expect(button).to_be_enabled(timeout=240000)
        expect(page.locator("#pieceOdds li")).to_have_count(0)
        items.nth(i).click()
        button.click()
        expect(page.locator("#pieceOdds li")).to_have_count(5, timeout=240000)
        values = page.locator("#pieceOdds meter").evaluate_all(
            "els => els.map(e => e.value)"
        )
        assert sorted(values, reverse=True) == values and 0 < sum(values) <= 1.00001
        measured = dict(top5=values, note=page.locator("#predictionNote").inner_text())
        assert "−ln(p)" in measured["note"]
        page.locator("#tracePrediction").screenshot(
            path=out / "next-token-probabilities.png"
        )
    page.locator("#layer").fill("3")
    slot = i % 8
    anchor = ids[i // 8 * 8 - 1]
    bank = (slot + (anchor >> 3) + 3) % 8
    expect(page.locator("#route")).to_contain_text(f"slot {slot} → bank {bank}")
    expect(page.locator("#traceInfo")).to_contain_text(f"anchor ID {anchor}")
    items.nth(i).focus()
    page.keyboard.press("ArrowRight")
    expect(items.nth(i + 1)).to_have_attribute("aria-pressed", "true")
    page.locator("#slot").fill("7")
    expect(page.locator("#traceInfo")).to_have_text("Manual routing example")
    items.nth(i).click()
    page.locator(".figure").screenshot(path=out / "routing-turkish.png")
    page.set_viewport_size({"width": 360, "height": 740})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.locator(".figure").screenshot(path=out / "routing-mobile.png")
    field.fill("")
    expect(items).to_have_count(3)
    page.close()
    return dict(
        turkish_tokens=ids,
        selected_position=i + 1,
        anchor=anchor,
        layer=3,
        bank=bank,
        widths=[360, 1100],
        prediction=measured,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url")
    p.add_argument("--prediction", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    server = None
    if not a.url:
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory="web")
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch(executable_path="/usr/bin/google-chrome")
            report = verify(
                b,
                a.url or f"http://127.0.0.1:{server.server_port}/",
                a.out,
                a.prediction,
            )
            b.close()
        (a.out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        print(report)
    finally:
        if server:
            server.shutdown()
            server.server_close()
