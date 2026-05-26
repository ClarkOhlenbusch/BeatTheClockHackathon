"""
Tests for the voice shopping agent.
Run: DYLD_LIBRARY_PATH="$(brew --prefix expat)/lib" venv/bin/python3.12 test_agent.py
"""
import asyncio
import os
import sys

from google import genai
from google.genai import types
from playwright.async_api import async_playwright

sys.path.insert(0, os.path.dirname(__file__))
from agent import URLS, API_KEY, MODEL, build_live_config, handle_tool_call

PASSED = 0
FAILED = 0
SKIPPED = 0


def report(name, ok, detail=""):
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        print(f"  ❌ {name}: {detail}")


def skip(name, detail=""):
    global SKIPPED
    SKIPPED += 1
    print(f"  ⏭️ {name}: {detail}")


async def test_pages_load():
    print("\n🧪 All pages load in browser")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        for name, url in URLS.items():
            await page.goto(url)
            title = await page.title()
            report(f"'{name}' loads", len(title) > 0)
        await browser.close()


async def test_add_to_cart_buttons():
    print("\n🧪 Add to Cart buttons exist")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        for name, url in URLS.items():
            if name == "home":
                continue
            await page.goto(url)
            count = await page.locator("button:has-text('Add to Cart')").count()
            report(f"'{name}' has button", count > 0)
        await browser.close()


async def test_handle_tool_call():
    print("\n🧪 handle_tool_call works")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        class FC:
            name = "navigate"
            args = {"page": "large_table"}
            id = "t1"

        result = await handle_tool_call(FC(), page)
        report("navigate returns result", "Navigated" in result)
        title = await page.title()
        report("browser navigated", "Solid Wood" in title)

        class FC2:
            name = "add_to_cart"
            args = {}
            id = "t2"

        result = await handle_tool_call(FC2(), page)
        report("add_to_cart returns result", "Added" in result)
        await browser.close()


async def test_live_session():
    print("\n🧪 Live API connects")
    if not API_KEY:
        skip("Session connected", "set GOOGLE_API_KEY or GEMINI_API_KEY")
        return

    client = genai.Client(api_key=API_KEY)
    try:
        async with client.aio.live.connect(model=MODEL, config=build_live_config()) as session:
            report("Session connected", True)
    except Exception as e:
        report("Session connected", False, str(e))


async def test_multi_turn_with_tools():
    print("\n🧪 Multi-turn conversation with tools")
    if not API_KEY:
        skip("Multi-turn conversation", "set GOOGLE_API_KEY or GEMINI_API_KEY")
        return

    client = genai.Client(api_key=API_KEY)

    async with client.aio.live.connect(model=MODEL, config=build_live_config()) as session:
        # Turn 1: ask for table
        await session.send_client_content(
            turns=[types.Content(role="user", parts=[types.Part.from_text(text="I want a big wooden table for 8 people")])]
        )
        tool_called = None
        async for r in session.receive():
            if r.tool_call:
                fc = r.tool_call.function_calls[0]
                tool_called = (fc.name, dict(fc.args) if fc.args else {})
                await session.send_tool_response(function_responses=[
                    types.FunctionResponse(id=fc.id, name=fc.name, response={"result": "done"})
                ])
            if r.server_content and r.server_content.turn_complete:
                break
        report("Turn 1: navigate(large_table)",
               tool_called and tool_called[0] == "navigate" and tool_called[1].get("page") == "large_table",
               f"got {tool_called}")

        # Turn 2: ask for chairs
        await session.send_client_content(
            turns=[types.Content(role="user", parts=[types.Part.from_text(text="Show me some chairs to go with it")])]
        )
        tool_called = None
        async for r in session.receive():
            if r.tool_call:
                fc = r.tool_call.function_calls[0]
                tool_called = (fc.name, dict(fc.args) if fc.args else {})
                await session.send_tool_response(function_responses=[
                    types.FunctionResponse(id=fc.id, name=fc.name, response={"result": "done"})
                ])
            if r.server_content and r.server_content.turn_complete:
                break
        report("Turn 2: navigate(chairs)",
               tool_called and tool_called[0] == "navigate" and tool_called[1].get("page") == "chairs",
               f"got {tool_called}")

        # Turn 3: add to cart
        await session.send_client_content(
            turns=[types.Content(role="user", parts=[types.Part.from_text(text="Perfect, add these to my cart")])]
        )
        tool_called = None
        async for r in session.receive():
            if r.tool_call:
                fc = r.tool_call.function_calls[0]
                tool_called = (fc.name, dict(fc.args) if fc.args else {})
                await session.send_tool_response(function_responses=[
                    types.FunctionResponse(id=fc.id, name=fc.name, response={"result": "done"})
                ])
            if r.server_content and r.server_content.turn_complete:
                break
        report("Turn 3: add_to_cart",
               tool_called and tool_called[0] == "add_to_cart",
               f"got {tool_called}")


async def run_all():
    print("=" * 50)
    print("🧪 Voice Shopping Agent Tests")
    print("=" * 50)

    await test_pages_load()
    await test_add_to_cart_buttons()
    await test_handle_tool_call()
    await test_live_session()
    await test_multi_turn_with_tools()

    print(f"\n{'=' * 50}")
    print(f"Results: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped")
    print("=" * 50)
    return FAILED == 0


if __name__ == "__main__":
    ok = asyncio.run(run_all())
    sys.exit(0 if ok else 1)
