"""Ubuntu smoke test through the installed launcher with a temporary private profile."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import shutil
import hashlib
import time
from playwright.sync_api import sync_playwright, expect


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--seed-cache", type=Path)
    p.add_argument("--launcher", type=Path, default=Path.home() / ".local/bin/franok")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    launcher = args.launcher
    with (
        tempfile.TemporaryDirectory(prefix="franok-desktop-test-") as temp,
        sync_playwright() as pw,
    ):
        profile = Path(temp)
        if args.seed_cache:
            with args.seed_cache.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            cache = profile / "weights"
            cache.mkdir()
            os.link(
                args.seed_cache, cache / (digest + ".q8")
            ) if args.seed_cache.stat().st_dev == cache.stat().st_dev else shutil.copyfile(
                args.seed_cache, cache / (digest + ".q8")
            )

        def launch(offline=False):
            portfile = profile / "DevToolsActivePort"
            portfile.unlink(missing_ok=True)
            log = (args.output / "launcher.log").open("ab")
            proc = subprocess.Popen(
                [
                    launcher,
                    f"--user-data-dir={profile}",
                    "--remote-debugging-port=0",
                    *(
                        ["--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost"]
                        if offline
                        else []
                    ),
                ],
                stdout=log,
                stderr=log,
                env={**os.environ, "WAYLAND_DEBUG": "client"},
            )
            log.close()
            try:
                for _ in range(120):
                    if portfile.exists():
                        break
                    if proc.poll() is not None:
                        raise RuntimeError("launcher exited")
                    time.sleep(0.15)
                port = portfile.read_text().splitlines()[0]
                b = pw.chromium.connect_over_cdp("http://127.0.0.1:" + port)
                page = b.contexts[0].pages[0]
                page.wait_for_url("franok://app/index.html*")
                return proc, b, page
            except Exception:
                proc.terminate()
                proc.wait(timeout=10)
                raise

        proc, b, page = launch()
        assert page.locator("#precision").inner_text().endswith("Q8 / WASM")
        page.locator("#draft").fill("Which language is spoken in Ankara?")
        page.locator("#send").click()
        expect(page.locator("#send")).to_be_enabled(timeout=180000)
        status = page.locator("#status").inner_text()
        assert "Generated locally" in status, status
        answer = page.locator(".message.assistant p").inner_text()
        page.screenshot(path=args.output / "desktop.png")
        page.close()
        proc.wait(timeout=15)
        proc, b, page = launch(offline=True)
        assert page.locator(".message.assistant p").inner_text() == answer
        page.locator("#find").click()
        page.locator("#query").fill("Ankara")
        page.get_by_role("button", name="Attach turn 1 to next message").click()
        assert page.locator("#quotechip").is_visible()
        page.locator("#architecture").click()
        page.locator("#slot").fill("7")
        assert "bank 7" in page.locator("#route").inner_text()
        page.get_by_role("button", name="Workspace", exact=True).click()
        page.locator("#draft").fill("Say hello in Turkish.")
        page.locator("#send").click()
        expect(page.locator("#send")).to_be_enabled(timeout=180000)
        offline_status = page.locator("#status").inner_text()
        assert "Generated locally" in offline_status, offline_status
        greetings = {}
        clarifications = {}
        for lang, message, expected in [
            ("en", "hi", "Hello!"),
            ("tr", "selam", "Merhaba!"),
        ]:
            page.locator("#new").click()
            page.locator("#lang").select_option(lang)
            page.locator("#draft").fill(message)
            page.locator("#send").click()
            expect(page.locator("#send")).to_be_enabled(timeout=180000)
            turn = page.locator(".message.assistant").last
            greetings[lang] = turn.locator(".body").inner_text()
            assert greetings[lang].startswith(expected), greetings
            assert "Generated locally" in page.locator("#status").inner_text()
            assert "Franok" in turn.locator("header").inner_text()
            prompt, label = ("wdym", "Conversation helper") if lang == "en" else ("Ne demek istedin?", "Sohbet yardımı")
            page.locator("#draft").fill(prompt)
            page.locator("#send").click()
            followup = page.locator(".message.assistant").last
            expect(followup.locator("header")).to_contain_text(label)
            clarifications[lang] = followup.locator(".body").inner_text()
            assert ("saying hello" if lang == "en" else "selamlaşıyordum") in clarifications[lang]
        page.close()
        proc.wait(timeout=15)
    report = dict(
        status=status,
        offline_status=offline_status,
        restarted=True,
        recovered=answer,
        attached=True,
        architecture=True,
        offline_greetings=greetings,
        offline_clarifications=clarifications,
    )
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    main()
