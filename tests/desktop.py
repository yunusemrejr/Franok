"""Ubuntu smoke test through the installed launcher with a temporary private profile."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from playwright.sync_api import sync_playwright,expect

def main():
    p=argparse.ArgumentParser();p.add_argument("output",type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    launcher=Path.home()/".local/bin/franok"
    with tempfile.TemporaryDirectory(prefix="franok-desktop-test-") as temp,sync_playwright() as pw:
        profile=Path(temp)
        def launch():
            portfile=profile/"DevToolsActivePort";portfile.unlink(missing_ok=True)
            log=(args.output/"launcher.log").open("ab")
            proc=subprocess.Popen([launcher,f"--user-data-dir={profile}","--remote-debugging-port=0"],stdout=log,stderr=log,env={**os.environ,"WAYLAND_DEBUG":"client"})
            log.close()
            try:
                for _ in range(120):
                    if portfile.exists():break
                    if proc.poll() is not None:raise RuntimeError("launcher exited")
                    time.sleep(.15)
                port=portfile.read_text().splitlines()[0]
                b=pw.chromium.connect_over_cdp("http://127.0.0.1:"+port)
                page=b.contexts[0].pages[0];page.wait_for_url("franok://app/index.html")
                return proc,b,page
            except Exception:
                proc.terminate();proc.wait(timeout=10);raise
        proc,b,page=launch()
        page.locator("#draft").fill("Which language is spoken in Ankara?")
        page.locator("#send").click()
        expect(page.locator("#send")).to_be_enabled(timeout=180000)
        status=page.locator("#status").inner_text()
        assert "Generated locally" in status,status
        answer=page.locator(".message.assistant p").inner_text()
        page.screenshot(path=args.output/"desktop.png");page.close();proc.wait(timeout=15)
        proc,b,page=launch()
        assert page.locator(".message.assistant p").inner_text()==answer
        page.locator("#find").click();page.locator("#query").fill("Ankara");page.get_by_role("button",name="Attach turn 1 to next message").click()
        assert page.locator("#quotechip").is_visible()
        page.locator("#architecture").click();page.locator("#slot").fill("7")
        assert "bank 7" in page.locator("#route").inner_text()
        page.close();proc.wait(timeout=15)
    report=dict(status=status,restarted=True,recovered=answer,attached=True,architecture=True)
    (args.output/"receipt.json").write_text(json.dumps(report,indent=2));print(report)
if __name__=="__main__":main()
