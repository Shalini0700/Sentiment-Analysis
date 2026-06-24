"""Playwright script — screenshot the running Streamlit app."""
from playwright.sync_api import sync_playwright
import time, pathlib

shots_dir = pathlib.Path(r"C:\Users\karta\OneDrive\Desktop\sentiment_app\screenshots")
shots_dir.mkdir(exist_ok=True)

POSITIVE = (
    "This movie was absolutely brilliant! The acting, direction, and soundtrack "
    "were all top-notch. A true masterpiece of cinema that I will watch again."
)
NEGATIVE = (
    "Terrible film. Boring plot, wooden acting, and a complete waste of two hours. "
    "One of the worst movies I have ever seen. Avoid at all costs."
)

def wait_for_app_ready(page, timeout_s=300):
    """Poll until the textarea is visible and enabled."""
    print("Waiting for models to finish loading...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            page.wait_for_selector("textarea", state="visible", timeout=4000)
            ta = page.locator("textarea").first
            if ta.is_enabled():
                print("  App ready!")
                return
        except Exception:
            pass
        print(f"  Still loading... ({int(time.time() - (deadline - timeout_s))}s elapsed)")
        time.sleep(3)
    raise TimeoutError("App never became ready")


def enter_text_and_analyse(page, text):
    """Fill textarea and click Analyse.

    Streamlit re-runs on textarea *blur*, not on keystrokes, so we must
    Tab away after typing to commit the value and enable the button.
    """
    ta = page.locator("textarea").first
    ta.click()
    page.keyboard.press("Control+A")
    page.keyboard.press("Delete")
    ta.type(text, delay=3)
    # Blur the textarea → Streamlit re-runs → button becomes enabled
    page.keyboard.press("Tab")
    time.sleep(1.5)

    # Find the button — the label contains an emoji so match by prefix
    btn = page.locator("button[data-testid='stBaseButton-primary']").first
    # Poll until enabled (Streamlit may take a render cycle)
    for _ in range(20):
        if btn.is_enabled():
            break
        time.sleep(0.3)

    # Use JS click to bypass Playwright's strict enabled-check if still disabled
    page.evaluate("document.querySelector(\"button[data-testid='stBaseButton-primary']\").click()")


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})

    # Load page and wait for models
    page.goto("http://localhost:8501", wait_until="domcontentloaded", timeout=60000)
    wait_for_app_ready(page, timeout_s=300)

    # ── Shot 1: clean landing page ────────────────────────────────────────────
    page.screenshot(path=str(shots_dir / "01_landing.png"), full_page=True)
    print("Shot 1: landing page saved")

    # ── Shot 2: positive review ───────────────────────────────────────────────
    enter_text_and_analyse(page, POSITIVE)
    # Wait for the results to render (both model headings appear)
    page.wait_for_selector("text=DistilBERT", timeout=120000)
    time.sleep(3)
    page.screenshot(path=str(shots_dir / "02_positive_result.png"), full_page=True)
    print("Shot 2: positive result saved")

    # ── Shot 3: negative review ───────────────────────────────────────────────
    enter_text_and_analyse(page, NEGATIVE)
    page.wait_for_selector("text=DistilBERT", timeout=60000)
    time.sleep(3)
    page.screenshot(path=str(shots_dir / "03_negative_result.png"), full_page=True)
    print("Shot 3: negative result saved")

    browser.close()

print("\nAll screenshots saved to:", shots_dir)
