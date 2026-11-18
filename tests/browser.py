"""Run full chat, session persistence, search, diagram, and mobile layout checks."""
import argparse,json,time
from pathlib import Path
from playwright.sync_api import sync_playwright,expect

def verify(page,url,folder):
    issues=[]
    page.on("pageerror",lambda error:issues.append(str(error)))
    page.goto(url,wait_until="networkidle")
    page.get_by_role("heading",name="Chat with Franok").wait_for()
    page.locator("#architecture").click()
    page.locator("#slot").fill("6");page.locator("#anchor").fill("17");page.locator("#layer").fill("3")
    assert page.locator("#route").inner_text()=="Token slot 6 → bank 3 of 8 · layer 3"
    page.screenshot(path=folder/"architecture.png")
    page.locator("#back").click()
    page.screenshot(path=folder/"chat.png")
    page.locator("#lang").select_option("tr")
    page.locator("#draft").fill("Merhaba! İzmir hakkında bir cümle yaz.")
    start=time.monotonic()
    page.locator("#send").click()
    expect(page.locator("#send")).to_be_enabled(timeout=240000)
    status=page.locator("#status").inner_text()
    assert "Generated locally" in status,(status,issues)
    reply=page.locator(".message.assistant p").last.inner_text()
    assert reply.strip()
    page.reload();page.get_by_text(reply,exact=True).wait_for()
    page.locator("#find").click();page.locator("#query").fill("İzmir")
    page.locator("#matches button").first.click()
    assert page.locator("#turn-0").evaluate("e=>e.classList.contains('hit')")
    page.locator("#new").click()
    page.get_by_role("button",name="Merhaba! İzmir hakkında bir cümle yaz.").click()
    assert page.locator(".message.assistant p").inner_text()==reply
    page.set_viewport_size({"width":390,"height":820})
    page.locator("#menu").click();page.locator("#architecture").click()
    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
    page.screenshot(path=folder/"mobile.png")
    page.locator("#back").click()
    page.locator("#draft").fill("Bir cümle daha yaz.");page.locator("#send").click()
    page.locator("#stop").click()
    assert page.locator("#status").inner_text()=="Stopped"
    assert not issues,issues
    return dict(status=status,reply=reply,elapsed=round(time.monotonic()-start,2),errors=issues)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("url");p.add_argument("output",type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path="/usr/bin/google-chrome",headless=True)
        page=browser.new_page(viewport={"width":1140,"height":790})
        receipt=verify(page,args.url,args.output);browser.close()
        (args.output/"check.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
        print(receipt)
