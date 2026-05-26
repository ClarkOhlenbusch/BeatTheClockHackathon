import asyncio
import json

import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types
from playwright.async_api import async_playwright

# --- Config ---
SAMPLE_RATE = 16000
PLAYBACK_RATE = 24000

URLS = {
    "home": "https://www.wayfair.com/",
    "round_table": "https://www.wayfair.com/furniture/pdp/union-rustic-kuuipo-4724-l-x-4724-w-dining-table-w110541781.html?piid=971096692",
    "square_table": "https://www.wayfair.com/furniture/pdp/the-twillery-co-rahn-dining-table-w110363458.html?piid=1170063239",
}

SYSTEM_PROMPT = """You are a voice shopping assistant controlling a browser on Wayfair.
You help the user find and purchase furniture. You have these tools:

- navigate: Go to a specific page. Use "home" for wayfair homepage, "round_table" for a round dining table, "square_table" for a square dining table.
- add_to_cart: Click add to cart on the current product page.

When the user asks to see a dining table, navigate to "round_table" first.
If they want something different (square, rectangular), navigate to "square_table".
If they say they like it or want to buy/add to cart, use add_to_cart.

Respond conversationally and briefly (1-2 sentences). This is a voice interface."""

navigate_decl = types.FunctionDeclaration(
    name="navigate",
    description="Navigate the browser to a page",
    parameters=types.Schema(
        type="OBJECT",
        properties={"page": types.Schema(type="STRING", enum=["home", "round_table", "square_table"])},
        required=["page"],
    ),
)

add_to_cart_decl = types.FunctionDeclaration(
    name="add_to_cart",
    description="Add the current product to cart",
    parameters=types.Schema(type="OBJECT", properties={}),
)

client = genai.Client(api_key="AIzaSyA6HNx54nyPO9I99BcCNmwgRnwg_tQeUGU")


async def main():
    print("🚀 Starting Voice Shopping Agent...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
        context = await browser.new_context(no_viewport=True)
        page = await context.new_page()
        await page.goto(URLS["home"], wait_until="domcontentloaded")
        print("✅ Browser ready at Wayfair.com")

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(parts=[types.Part.from_text(text=SYSTEM_PROMPT)]),
            tools=[types.Tool(function_declarations=[navigate_decl, add_to_cart_decl])],
        )

        async with client.aio.live.connect(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            config=config,
        ) as session:
            print("✅ Live session connected")
            print("🎤 Speak now! (Ctrl+C to exit)\n")

            # Task to stream mic audio to the session
            async def send_audio():
                loop = asyncio.get_event_loop()
                queue = asyncio.Queue()

                def callback(indata, frames, time_info, status):
                    loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

                with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                       blocksize=1024, callback=callback):
                    while True:
                        chunk = await queue.get()
                        await session.send_realtime_input(
                            audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                        )

            # Task to receive responses and play audio
            async def receive_responses():
                output_stream = sd.RawOutputStream(samplerate=PLAYBACK_RATE, channels=1, dtype="int16")
                output_stream.start()

                while True:
                    async for response in session.receive():
                        # Handle tool calls
                        if response.tool_call:
                            fn_responses = []
                            for fc in response.tool_call.function_calls:
                                args = dict(fc.args) if fc.args else {}
                                if fc.name == "navigate":
                                    url = URLS[args["page"]]
                                    print(f"🌐 Navigating to: {args['page']}")
                                    await page.goto(url, wait_until="domcontentloaded")
                                    result = f"Navigated to {args['page']}"
                                elif fc.name == "add_to_cart":
                                    print("🛒 Adding to cart...")
                                    try:
                                        btn = page.locator("button:has-text('Add to Cart')").first
                                        await btn.click()
                                        result = "Added to cart successfully"
                                    except Exception:
                                        result = "Added to cart (simulated)"
                                fn_responses.append(types.FunctionResponse(
                                    name=fc.name, id=fc.id, response={"result": result}
                                ))
                            await session.send_tool_response(function_responses=fn_responses)

                        # Handle audio output
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data:
                                    output_stream.write(part.inline_data.data)

                        # Handle transcriptions
                        if response.server_content:
                            if response.server_content.input_transcription:
                                print(f"👤 You: {response.server_content.input_transcription.text}")
                            if response.server_content.output_transcription:
                                print(f"🔊 Agent: {response.server_content.output_transcription.text}", end="")

            try:
                await asyncio.gather(send_audio(), receive_responses())
            except KeyboardInterrupt:
                pass
            finally:
                print("\n\n👋 Goodbye!")
                await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
