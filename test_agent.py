"""
Tests for the voice shopping agent.
Run: DYLD_LIBRARY_PATH="$(brew --prefix expat)/lib" venv/bin/python3.12 test_agent.py
"""
import asyncio
import os
import sys
from contextlib import AsyncExitStack

from google import genai
from google.genai import types
from playwright.async_api import async_playwright

sys.path.insert(0, os.path.dirname(__file__))
import agent
from agent import URLS, PRODUCTS, API_KEY, build_live_config, apply_noise_gate, connect_live_session, handle_tool_call, run_tool_call

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
            if name not in PRODUCTS:
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


async def test_cart_checkout_tools():
    print("\n🧪 cart and checkout tools work")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(URLS["chairs"])

        class AddTwo:
            name = "add_to_cart"
            args = {"quantity": 2}
            id = "t3"

        result = await handle_tool_call(AddTwo(), page)
        report("adds requested quantity", "2 sets" in result)
        cart = await page.evaluate("() => JSON.parse(localStorage.getItem('wayfair_voice_cart'))")
        report("cart stores quantity", cart[0]["quantity"] == 2)

        class ViewCart:
            name = "view_cart"
            args = {}
            id = "t4"

        result = await handle_tool_call(ViewCart(), page)
        report("view_cart summarizes cart", "Total is $379.98" in result)
        report("cart page visible", "cart.html" in page.url)

        class StartCheckout:
            name = "start_checkout"
            args = {}
            id = "t5"

        result = await handle_tool_call(StartCheckout(), page)
        report("checkout starts", "Checkout is ready" in result)
        report("checkout page visible", "checkout.html" in page.url)

        class PlaceOrder:
            name = "place_order"
            args = {}
            id = "t6"

        result = await handle_tool_call(PlaceOrder(), page)
        report("order places", "WF-MOCK-1047" in result)
        report("confirmation page visible", "confirmation.html" in page.url)
        cart_after_order = await page.evaluate("() => localStorage.getItem('wayfair_voice_cart')")
        report("cart clears after order", cart_after_order is None)
        await browser.close()


async def test_tool_call_timeout():
    print("\n🧪 tool calls cannot hang the conversation")
    old_timeout = agent.TOOL_TIMEOUT_SECONDS
    agent.TOOL_TIMEOUT_SECONDS = 0.01

    class SlowPage:
        async def goto(self, *args, **kwargs):
            await asyncio.sleep(1)

    class Navigate:
        name = "navigate"
        args = {"page": "square_table"}
        id = "t7"

    try:
        result = await run_tool_call(Navigate(), SlowPage())
        report("slow navigation returns failure", "navigate failed" in result)
    finally:
        agent.TOOL_TIMEOUT_SECONDS = old_timeout


def test_voice_session_config():
    print("\n🧪 Voice session config")
    config = build_live_config()
    vad = config.realtime_input_config.automatic_activity_detection

    report("barge-in disabled for noise", config.realtime_input_config.activity_handling == types.ActivityHandling.NO_INTERRUPTION)
    report("start sensitivity is low", vad.start_of_speech_sensitivity == types.StartSensitivity.START_SENSITIVITY_LOW)
    report("end sensitivity is low", vad.end_of_speech_sensitivity == types.EndSensitivity.END_SENSITIVITY_LOW)
    report("prefix padding configured", vad.prefix_padding_ms >= 500)
    report("silence duration configured", vad.silence_duration_ms >= 900)


def test_noise_gate():
    print("\n🧪 Noise gate")
    quiet = (1).to_bytes(2, byteorder=sys.byteorder, signed=True) * 64
    speech = (2000).to_bytes(2, byteorder=sys.byteorder, signed=True) * 64

    report("quiet audio passes by default", apply_noise_gate(quiet) == quiet)
    report("speech audio passes through", apply_noise_gate(speech) == speech)


async def test_live_session():
    print("\n🧪 Live API connects")
    if not API_KEY:
        skip("Session connected", "set GOOGLE_API_KEY or GEMINI_API_KEY")
        return

    client = genai.Client(api_key=API_KEY)
    try:
        async with AsyncExitStack() as stack:
            await connect_live_session(client, stack)
            report("Session connected", True)
    except Exception as e:
        report("Session connected", False, str(e))


async def test_multi_turn_with_tools():
    print("\n🧪 Multi-turn conversation with tools")
    if not API_KEY:
        skip("Multi-turn conversation", "set GOOGLE_API_KEY or GEMINI_API_KEY")
        return

    client = genai.Client(api_key=API_KEY)

    async with AsyncExitStack() as stack:
        session = await connect_live_session(client, stack)
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
    await test_cart_checkout_tools()
    await test_tool_call_timeout()
    test_voice_session_config()
    test_noise_gate()
    await test_live_session()
    await test_multi_turn_with_tools()

    print(f"\n{'=' * 50}")
    print(f"Results: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped")
    print("=" * 50)
    return FAILED == 0


if __name__ == "__main__":
    ok = asyncio.run(run_all())
    sys.exit(0 if ok else 1)
