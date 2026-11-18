"""Run full chat, session persistence, search, diagram, and mobile layout checks."""

import argparse, json, re, time
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def verify(page, url, folder):
    issues = []
    page.on("pageerror", lambda error: issues.append(str(error)))
    page.goto(url, wait_until="networkidle")
    page.get_by_role("heading", name="Chat with Franok").wait_for()
    logo = page.locator("#welcome h1 img")
    assert logo.evaluate("img => img.complete && img.naturalWidth === 256")
    assert logo.get_attribute("src") == page.locator(".brand img").get_attribute("src")
    viewport = page.viewport_size
    for width in (320, 390, 391):
        page.set_viewport_size({"width": width, "height": 800})
        assert page.locator(".brand").evaluate("""brand => {
            const word = brand.querySelector('strong');
            const visible = word.getClientRects().length ? word : brand.querySelector('img');
            const tabs = document.querySelector('.tabs').getBoundingClientRect();
            return visible.getBoundingClientRect().right <= tabs.left
                && tabs.right <= document.querySelector('.mastend').getBoundingClientRect().left;
        }""")
    page.set_viewport_size(viewport)
    page.locator("#architecture").click()
    page.locator("#slot").fill("6")
    page.locator("#anchor").fill("17")
    page.locator("#layer").fill("3")
    assert page.locator("#route").inner_text() == "Token slot 6 → bank 3 of 8 · layer 3"
    page.screenshot(path=folder / "architecture.png")
    page.get_by_role("button", name="Workspace", exact=True).click()
    page.screenshot(path=folder / "chat.png")
    page.get_by_role("button", name="Downloads", exact=True).click()
    assert page.get_by_role("heading", name="Franok for Ubuntu").is_visible()
    page.get_by_role("button", name="Workspace", exact=True).click()
    page.locator("#lang").select_option("tr")
    page.locator("#controls").click()
    assert "Türkçe" in page.locator("#sysprompt").input_value()
    page.locator("#temp").fill("0.35")
    page.locator("#tokens").fill("42")
    page.get_by_role("button", name="Save controls").click()
    expect(page.locator("#prefs")).not_to_be_visible()
    page.locator("#draft").fill("Merhaba! İzmir hakkında bir cümle yaz.")
    start = time.monotonic()
    page.locator("#send").click()
    expect(page.locator("#send")).to_be_enabled(timeout=240000)
    expect(page.locator("#status")).to_contain_text("Generated locally", timeout=15000)
    status = page.locator("#status").inner_text()
    assert "Generated locally" in status, (status, issues)
    reply = page.locator(".message.assistant p").last.inner_text()
    assert reply.strip()
    page.reload()
    page.get_by_text(reply, exact=True).wait_for()
    page.locator("#find").click()
    page.locator("#query").fill("İzmir")
    page.locator("#matches").get_by_role("button", name=re.compile(r"^Turn 1 · ")).click()
    assert page.locator("#turn-0").evaluate("e=>e.classList.contains('hit')")
    page.locator("#new").click()
    page.get_by_role("button", name="Merhaba! İzmir hakkında bir cümle yaz.").click()
    assert page.locator(".message.assistant p").inner_text() == reply
    page.set_viewport_size({"width": 390, "height": 820})
    page.locator("#menu").click()
    assert page.locator("#filter").is_visible()
    page.locator("#closemenu").click()
    page.locator("#architecture").click()
    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
    page.screenshot(path=folder / "mobile.png")
    page.get_by_role("button", name="Workspace", exact=True).click()
    page.locator("#draft").fill("Bir cümle daha yaz.")
    page.locator("#send").click()
    page.locator("#stop").click()
    assert page.locator("#status").inner_text() == "Generation stopped."
    page.set_viewport_size({"width": 1140, "height": 790})
    page.locator("#actions").click()
    page.locator("#rename").fill("Bilingual session")
    page.get_by_role("button", name="Rename", exact=True).click()
    page.locator("#filter").fill("Bilingual")
    assert page.locator("#sessions button").count() == 1
    page.locator("#controls").click()
    assert page.locator("#tokens").input_value() == "42"
    page.get_by_role("button", name="Close controls").click()
    assert not issues, issues
    return dict(
        status=status,
        reply=reply,
        elapsed=round(time.monotonic() - start, 2),
        errors=issues,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("output", type=Path)
    p.add_argument(
        "--browser", choices=("chromium", "firefox", "webkit"), default="chromium"
    )
    p.add_argument(
        "--executable", help="Optional local browser executable for this engine"
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        binary = args.executable or (
            "/usr/bin/google-chrome" if args.browser == "chromium" else None
        )
        browser = getattr(pw, args.browser).launch(
            executable_path=binary, headless=True
        )
        page = browser.new_page(viewport={"width": 1140, "height": 790})
        receipt = verify(page, args.url, args.output)
        receipt["browser"] = args.browser
        receipt["version"] = browser.version
        browser.close()
        (args.output / "check.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2)
        )
        print(receipt)
