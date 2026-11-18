"""Exercise legacy session migration, concurrent editing, saved drafts and a local-only tool."""

import argparse, json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def verify(url):
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path="/usr/bin/google-chrome")
        ctx = browser.new_context()
        legacy = [
            dict(
                id="legacy-test",
                lang="tr",
                draft="Yarım kalan taslak",
                turns=[
                    dict(role="user", text="Önceki notum."),
                    dict(role="assistant", text="Notunu gördüm."),
                ],
            )
        ]
        ctx.add_init_script(
            "if(!localStorage.getItem('fixture-init')){localStorage.setItem('franok.sessions.v1',"
            + json.dumps(json.dumps(legacy))
            + ");localStorage.setItem('franok.active.v1','legacy-test');localStorage.setItem('fixture-init','1');}"
        )
        a = ctx.new_page()
        errors = []
        a.on("pageerror", lambda err: errors.append(str(err)))
        a.goto(url)
        expect(a.locator("#draft")).to_have_value("Yarım kalan taslak", timeout=30000)
        a.get_by_role("button", name="Pin turn 1", exact=True).click()
        a.locator("#draft").fill("Kaydedilmiş ikinci taslak")
        a.reload()
        expect(a.locator("#draft")).to_have_value("Kaydedilmiş ikinci taslak")
        expect(
            a.get_by_role("button", name="Unpin turn 1", exact=True)
        ).to_have_attribute("aria-pressed", "true")
        a.locator("#controls").click()
        a.locator("#sysprompt").fill("Tek cümle ile cevap ver.")
        a.get_by_role("button", name="Save controls").click()
        a.locator("#draft").fill("0,2 + 0,1")
        a.locator("#send").click()
        expect(a.locator(".message.assistant .body p").last).to_have_text("0,3")
        assert (
            "Hesap makinesi" in a.locator(".message.assistant header").last.inner_text()
        )
        a.locator("#draft").fill("/hatırla önceki")
        a.locator("#send").click()
        expect(a.locator(".message.assistant .body p").last).to_have_text(
            "Mesaj 1 · Önceki notum."
        )
        a.get_by_role("button", name="Mesaja git 1", exact=True).click()
        assert a.locator("#turn-0").evaluate("e => e.classList.contains('hit')")
        b = ctx.new_page()
        b.goto(url)
        expect(b.locator("#draft")).to_have_value("")

        def rename(page, name):
            page.locator("#actions").click()
            page.locator("#rename").fill(name)
            page.get_by_role("button", name="Rename", exact=True).click()
            page.get_by_role("button", name=name, exact=True).wait_for()

        rename(a, "First edited copy")
        rename(b, "Second edited copy")
        a.reload()
        for name in ("First edited copy", "Second edited copy"):
            expect(a.get_by_role("button", name=name, exact=True)).to_be_visible()
        a.get_by_role("button", name="First edited copy", exact=True).click()
        a.locator("#controls").click()
        expect(a.locator("#sysprompt")).to_have_value("Tek cümle ile cevap ver.")
        a.get_by_role("button", name="Close controls").click()
        a.locator("#actions").click()
        a.get_by_role("button", name="Delete session…").click()
        a.locator("#reallydelete").click()
        expect(
            a.get_by_role("button", name="First edited copy", exact=True)
        ).to_have_count(0)
        assert (
            a.evaluate("JSON.parse(localStorage.getItem('franok.sessions.v1')).length")
            == 0
        )
        a.locator("#new").click()
        a.locator("#lang").select_option("en")
        a.locator("#draft").fill("hi")
        a.locator("#send").click()
        expect(a.locator("#send")).to_be_enabled(timeout=240000)
        first = a.locator(".message.assistant").last
        if a.evaluate(
            "fetch(new URL('./release.json', document.baseURI)).then(r=>r.json()).then(r=>r.systemConditioned)"
        ):
            assert "Generated locally" in a.locator("#status").inner_text()
            assert first.locator(".body").inner_text().startswith("Hello!")
            assert "Franok" in first.locator("header").inner_text()
        else:
            expect(first.locator(".body")).to_have_text(
                "Hi! What would you like to talk about?"
            )
        for prompt, expected in [
            ("wdym?", "I was just saying hello. Is there something on your mind?"),
            (
                "you make no sense.",
                "Let’s start again. What question would you like me to answer?",
            ),
        ]:
            a.locator("#draft").fill(prompt)
            a.locator("#send").click()
            expect(a.locator(".message.assistant .body p").last).to_have_text(expected)
            assert (
                "Conversation helper"
                in a.locator(".message.assistant header").last.inner_text()
            )
        a.locator("#controls").click()
        a.locator("#casual").uncheck()
        a.get_by_role("button", name="Save controls").click()
        expect(a.locator("#prefs")).not_to_be_visible()
        a.reload()
        a.locator("#controls").click()
        expect(a.locator("#casual")).not_to_be_checked()
        assert not errors, errors
        browser.close()
    print(
        "Verified migrated history, persistent pin and draft, exact arithmetic, per-session controls and concurrent edit recovery."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("url")
    args = p.parse_args()
    verify(args.url)
