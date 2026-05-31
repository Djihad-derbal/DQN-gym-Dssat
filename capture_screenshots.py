"""Drive the running NitroDQN app and save feature screenshots to ./screenshots/."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

URL = "http://localhost:8765"
OUT = Path(__file__).parent / "screenshots"
OUT.mkdir(exist_ok=True)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await ctx.new_page()
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_timeout(1500)

        # 1) Dashboard overview (cold)
        await page.screenshot(path=str(OUT / "01_dashboard_overview.png"), full_page=False)
        print("saved 01_dashboard_overview.png")

        # 2) Start an episode -> State tab populated
        await page.click("button:has-text('New Episode')")
        await page.wait_for_timeout(2500)
        # ensure State tab active
        await page.evaluate("switchTab && switchTab('explain')")
        await page.wait_for_timeout(500)
        await page.screenshot(path=str(OUT / "02_state_explainer.png"))
        print("saved 02_state_explainer.png")

        # 3) Step the DQN once -> Decision tab
        await page.click("button:has-text('Step DQN')")
        await page.wait_for_timeout(2500)
        await page.evaluate("switchTab && switchTab('decision')")
        await page.wait_for_timeout(500)
        await page.screenshot(path=str(OUT / "03_decision_narrator.png"))
        print("saved 03_decision_narrator.png")

        # 4) Auto-run several steps to populate season timeline + N applied
        await page.click("button:has-text('Auto 10 Steps')")
        await page.wait_for_timeout(4000)
        await page.screenshot(path=str(OUT / "04_after_auto10.png"))
        print("saved 04_after_auto10.png")

        # 5) Advisor chat
        await page.evaluate("switchTab && switchTab('chat')")
        await page.wait_for_timeout(500)
        await page.fill("#chat-input", "What does the nstres score mean and when should I worry?")
        await page.click("#chat-send")
        await page.wait_for_timeout(2500)
        await page.screenshot(path=str(OUT / "05_advisor_chat.png"))
        print("saved 05_advisor_chat.png")

        # 6) LLM Settings modal
        await page.click("button:has-text('LLM Settings')")
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(OUT / "06_llm_settings.png"))
        print("saved 06_llm_settings.png")
        # close modal by clicking outside / esc
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(300)

        # 7) Training history (scroll to bottom)
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(OUT / "07_training_history.png"), full_page=False)
        print("saved 07_training_history.png")

        # 8) Full-page composite
        await page.evaluate("window.scrollTo(0,0)")
        await page.wait_for_timeout(300)
        await page.screenshot(path=str(OUT / "08_fullpage.png"), full_page=True)
        print("saved 08_fullpage.png")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
