import asyncio
import os

import sounddevice as sd
from google import genai
from google.genai import types
from playwright.async_api import async_playwright

# --- Config ---
SAMPLE_RATE = 16000
PLAYBACK_RATE = 24000

PAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages")
URLS = {
    "home": f"file://{PAGE_DIR}/index.html",
    "large_table": f"file://{PAGE_DIR}/large-table.html",
    "round_table": f"file://{PAGE_DIR}/round-table.html",
    "square_table": f"file://{PAGE_DIR}/square-table.html",
    "chairs": f"file://{PAGE_DIR}/chairs.html",
    "placemats": f"file://{PAGE_DIR}/placemats.html",
}

SYSTEM_PROMPT = """You are a friendly, enthusiastic voice shopping assistant on Wayfair. You're helping someone furnish their home.

Your personality: warm, excited to help, asks thoughtful questions to understand what they need before showing products.

CONVERSATION FLOW:
1. When the user mentions a room or need, get excited and ask clarifying questions (how many people, style preferences, budget, etc.)
2. Based on their answers, show them a product using the navigate tool and describe what you're showing them.
3. Ask what they think. If they like it, add it to cart and then suggest complementary items (e.g., after a table, suggest chairs; after chairs, suggest placemats).
4. Keep the conversation going naturally — you're a personal shopping assistant.

PRODUCT CATALOG:
- large_table: Hawthorne 72" Solid Wood Rectangular Table, seats 6-8, natural acacia wood, $649.99. Great for families or entertaining.
- round_table: Kuuipo 47" Round Table, seats 4, solid wood, rustic brown, $284.99. Cozy and intimate.
- square_table: Rahn 48" Square Table, seats 4-6, espresso finish, $349.99. Modern and clean.
- chairs: Parsons Upholstered Dining Chairs (set of 2), linen blend, solid wood legs, beige, $189.99/pair.
- placemats: Handwoven Cotton Placemats (set of 6), natural/tan, farmhouse style, $34.99.

RULES:
- Keep responses to 2-3 sentences max. This is voice — be concise.
- Always use navigate to show a product BEFORE describing it.
- After adding to cart, suggest a related item.
- If they want a wooden table for 6-8 people, show large_table.
- If they want something smaller or for 4, show round_table or square_table.
- Match chairs and placemats to complement whatever table they chose."""

PAGE_NAMES = ["home", "large_table", "round_table", "square_table", "chairs", "placemats"]

navigate_decl = types.FunctionDeclaration(
    name="navigate",
    description="Navigate the browser to a product page to show the user",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "page": types.Schema(type="STRING", enum=PAGE_NAMES),
        },
        required=["page"],
    ),
)

add_to_cart_decl = types.FunctionDeclaration(
    name="add_to_cart",
    description="Add the current product to cart",
    parameters=types.Schema(type="OBJECT", properties={}),
)

tools = [types.Tool(function_declarations=[navigate_decl, add_to_cart_decl])]

API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-3.1-flash-live-preview"


def build_live_config():
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_PROMPT,
        tools=tools,
        input_audio_transcription={},
        output_audio_transcription={},
    )


async def handle_tool_call(fc, page):
    """Execute a tool call and return the result string."""
    args = dict(fc.args) if fc.args else {}
    if fc.name == "navigate":
        url = URLS[args["page"]]
        print(f"🌐 Navigating to: {args['page']}")
        await page.goto(url, wait_until="domcontentloaded")
        return f"Navigated to {args['page']}"
    elif fc.name == "add_to_cart":
        print("🛒 Adding to cart...")
        try:
            btn = page.locator("button:has-text('Add to Cart')").first
            await btn.click(timeout=2000)
        except Exception:
            pass
        return "Added to cart successfully"
    return "Unknown tool"


async def main():
    print("🚀 Starting Voice Shopping Agent...")
    if not API_KEY:
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY before running the voice agent.")

    client = genai.Client(api_key=API_KEY)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
        context = await browser.new_context(no_viewport=True)
        page = await context.new_page()
        await page.goto(URLS["home"], wait_until="domcontentloaded")
        print("✅ Browser ready at Wayfair.com")

        async with client.aio.live.connect(model=MODEL, config=build_live_config()) as session:
            print("✅ Live session connected")
            print("🎤 Speak now! (Ctrl+C to exit)\n")

            async def send_audio():
                loop = asyncio.get_running_loop()
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

            async def receive_responses():
                with sd.RawOutputStream(samplerate=PLAYBACK_RATE, channels=1, dtype="int16") as output_stream:
                    while True:
                        received_message = False
                        async for response in session.receive():
                            received_message = True
                            # session.receive() ends after each model turn, so this outer
                            # loop keeps the agent listening for the next user utterance.
                            if response.tool_call:
                                function_responses = []
                                for fc in response.tool_call.function_calls:
                                    result = await handle_tool_call(fc, page)
                                    function_responses.append(types.FunctionResponse(
                                        id=fc.id, name=fc.name, response={"result": result}
                                    ))
                                await session.send_tool_response(function_responses=function_responses)

                            if response.data is not None:
                                output_stream.write(response.data)
                            elif response.server_content and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if part.inline_data:
                                        output_stream.write(part.inline_data.data)

                            if response.server_content:
                                if response.server_content.input_transcription:
                                    print(f"\n👤 You: {response.server_content.input_transcription.text}")
                                if response.server_content.output_transcription:
                                    print(f"🔊 {response.server_content.output_transcription.text}", end="")
                        if not received_message:
                            break

            try:
                await asyncio.gather(send_audio(), receive_responses())
            except KeyboardInterrupt:
                pass
            finally:
                print("\n\n👋 Goodbye!")
                await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
