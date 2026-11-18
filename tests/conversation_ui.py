"""Check the reported greeting and follow-up in fresh and previously saved web sessions."""

import argparse
import json
from pathlib import Path
import re
from playwright.sync_api import sync_playwright, expect


def send(page, text):
    before = page.locator(".message.assistant").count()
    page.locator("#draft").fill(text)
    page.locator("#send").click()
    expect(page.locator("#send")).to_be_enabled(timeout=240000)
    replies = page.locator(".message.assistant")
    expect(replies).to_have_count(before + 1)
    turn = replies.last
    return dict(
        author=turn.locator("header").inner_text(),
        text=turn.locator(".body").inner_text(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    errors, result = [], {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path="/usr/bin/google-chrome")
        page = browser.new_page(viewport=dict(width=1100, height=730))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.url)
        for lang, hello, follow in (
            ("en", "hi", "wdym"),
            ("tr", "selam", "Ne demek istedin?"),
        ):
            page.locator("#new").click()
            page.locator("#lang").select_option(lang)
            opening = send(page, hello)
            assert re.search(
                r"\b(hi|hello|hey)\b" if lang == "en" else r"\b(merhaba|selam)\b",
                opening["text"],
                re.I,
            ), opening
            clarification = send(page, follow)
            assert (
                "Conversation helper" if lang == "en" else "Sohbet yardımı"
            ) in clarification["author"]
            assert (
                "saying hello" if lang == "en" else "selamlaşıyordum"
            ) in clarification["text"]
            result[lang] = dict(opening=opening, clarification=clarification)
            if lang == "en":
                result[lang]["second_repair"] = send(page, "you make no sense.")
                page.screenshot(path=args.output / "greeting-and-followup.png")
            page.reload()
            expect(page.locator(".message.assistant .body").first).to_have_text(
                opening["text"]
            )
        page.close()
        page = browser.new_page(viewport=dict(width=390, height=800))
        page.on("pageerror", lambda e: errors.append(str(e)))
        old = dict(
            id="old-session",
            lang="en",
            created=1795000000000,
            updated=1795000000000,
            turns=[
                dict(role="user", text="hi"),
                dict(
                    role="assistant",
                    text="Humans are the same way to do, but I am a person with it.",
                ),
            ],
        )
        page.add_init_script(
            "localStorage.setItem('franok.sessions.v1', "
            + json.dumps(json.dumps([old]))
            + ")"
        )
        page.goto(args.url)
        expect(page.locator(".message.assistant .body")).to_have_text(
            old["turns"][1]["text"]
        )
        repaired = send(page, "wdym")
        assert (
            "You said hello" in repaired["text"]
            and "Conversation helper" in repaired["author"]
        )
        result["existing_session"] = repaired
        assert not errors, errors
        result["errors"] = errors
        browser.close()
    (args.output / "receipt.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(result)


if __name__ == "__main__":
    main()
