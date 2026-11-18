"""Exercise bilingual search, source attachment and offline lexical fallback in a real browser."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from playwright.sync_api import expect, sync_playwright


def check(browser, url, output, docs, gold, query, broken=False, context=False):
    page = browser.new_page(viewport=dict(width=1024, height=740))
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    turns = [
        dict(role="user", text=text, at=1795000001000 + i)
        for i, text in enumerate(docs)
    ]
    session = dict(
        id="meaning-search-check",
        created=1795000000000,
        updated=1795000002000,
        turns=turns,
        lang="tr",
        options={},
        draft="",
    )
    page.add_init_script(
        "if (!localStorage.getItem('franok.idb.v1')) localStorage.setItem('franok.sessions.v1', "
        + json.dumps(json.dumps([session], ensure_ascii=False))
        + ");"
    )
    if broken:
        page.route("**/recall.bin", lambda route: route.fulfill(body=b"bad checksum"))
    page.goto(url, wait_until="domcontentloaded")
    page.locator("#find").click()
    page.locator("#semanticSearch").check()
    field = page.locator("#query")
    started = time.monotonic()
    field.fill(query)
    if broken:
        expect(page.locator("#searchStatus")).to_contain_text(
            "unavailable", timeout=15000
        )
        field.fill("umbrella")
    rows = page.locator("#matches .matchrow")
    expect(rows.first).to_contain_text(docs[gold][:40], timeout=30000)
    if not broken:
        expect(rows.first).to_contain_text("Related by meaning")
        page.screenshot(path=output / "bilingual-session-search.png")
    elapsed = time.monotonic() - started
    rows.first.get_by_role(
        "button", name=f"Attach turn {gold + 1} to next message"
    ).click()
    expect(page.locator("#quotechip")).to_contain_text(f"Attached turn {gold + 1}")
    page.reload()
    expect(page.locator("#quotechip")).to_contain_text(f"Attached turn {gold + 1}")
    source_label = None
    if context:
        page.locator("#quotechip button").click()
        page.locator("#controls").click()
        page.locator("#contextMeaning").check()
        page.locator("#tokens").fill("18")
        page.get_by_role("button", name="Save controls").click()
        page.locator("#draft").fill("umbrella" if broken else query)
        page.locator("#send").click()
        expect(page.locator("#status")).to_contain_text("Generated locally", timeout=240000)
        details = page.locator(".message.assistant .contextinfo").last
        details.locator("summary").click()
        label = "word search" if broken else "meaning search"
        source = details.get_by_role("button", name=f"Turn {gold + 1} · {label}", exact=False)
        expect(source).to_be_visible()
        source_label = source.inner_text()
        if broken:
            expect(details).to_contain_text("Meaning search was unavailable")
        source.click()
        assert page.locator(f"#turn-{gold}").evaluate("turn => turn.classList.contains('hit')")
        page.reload()
        page.locator("#controls").click()
        expect(page.locator("#contextMeaning")).to_be_checked()
    assert not errors, errors
    page.close()
    return dict(
        turns=len(docs), turn=gold + 1, seconds=round(elapsed, 3), word_fallback=broken,
        automatic_source=source_label,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url")
    p.add_argument("--sets", type=Path)
    p.add_argument("--context", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    notes = json.loads(Path("tests/recall_cases.json").read_text())
    target, _, query = notes[4]
    docs = (
        json.loads(args.sets.read_text())["notes-cross"]["docs"]
        if args.sets
        else [r[0] for r in notes]
    )
    gold = docs.index(target)
    server = None
    if not args.url:
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory="web")
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = args.url or f"http://127.0.0.1:{server.server_port}/"
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path="/usr/bin/google-chrome")
            rows = [
                check(browser, url, args.out, docs, gold, query, context=args.context),
                check(
                    browser, url, args.out, [r[0] for r in notes], 4, query, broken=True, context=args.context
                ),
            ]
            browser.close()
        (args.out / "receipt.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(rows)
    finally:
        if server:
            server.shutdown()
            server.server_close()
