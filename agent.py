import asyncio
import os
from array import array
from contextlib import AsyncExitStack

import sounddevice as sd
from google import genai
from google.genai import types
from playwright.async_api import async_playwright

# --- Config ---
SAMPLE_RATE = 16000
PLAYBACK_RATE = 24000
INPUT_BLOCKSIZE = 1024
NOISE_GATE_RMS = int(os.environ.get("VOICE_NOISE_GATE_RMS", "0"))
ASSISTANT_AUDIO_GUARD_SECONDS = 0.5
STARTUP_MESSAGE = (
    "Greet the shopper in one short sentence and ask what furniture they are shopping for today."
)

PAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages")
URLS = {
    "home": f"file://{PAGE_DIR}/index.html",
    "large_table": f"file://{PAGE_DIR}/large-table.html",
    "round_table": f"file://{PAGE_DIR}/round-table.html",
    "square_table": f"file://{PAGE_DIR}/square-table.html",
    "chairs": f"file://{PAGE_DIR}/chairs.html",
    "placemats": f"file://{PAGE_DIR}/placemats.html",
    "cart": f"file://{PAGE_DIR}/cart.html",
    "checkout": f"file://{PAGE_DIR}/checkout.html",
    "confirmation": f"file://{PAGE_DIR}/confirmation.html",
}

PRODUCTS = {
    "large_table": {
        "page": "large_table",
        "name": "Hawthorne 72\" Solid Wood Rectangular Dining Table",
        "price": 649.99,
        "unit": "table",
        "image": "🪵",
    },
    "round_table": {
        "page": "round_table",
        "name": "Kuuipo 47\" Round Dining Table",
        "price": 284.99,
        "unit": "table",
        "image": "🟤",
    },
    "square_table": {
        "page": "square_table",
        "name": "Rahn 48\" Square Dining Table",
        "price": 349.99,
        "unit": "table",
        "image": "⬛",
    },
    "chairs": {
        "page": "chairs",
        "name": "Parsons Upholstered Dining Chair (Set of 2)",
        "price": 189.99,
        "unit": "set",
        "image": "🪑🪑",
    },
    "placemats": {
        "page": "placemats",
        "name": "Handwoven Cotton Placemats (Set of 6)",
        "price": 34.99,
        "unit": "set",
        "image": "🧶",
    },
}

SYSTEM_PROMPT = """You are a friendly, enthusiastic voice shopping assistant on Wayfair. You're helping someone furnish their home.

Your personality: warm, excited to help, asks thoughtful questions to understand what they need before showing products.

CONVERSATION FLOW:
1. When the user mentions a room or need, get excited and ask clarifying questions (how many people, style preferences, budget, etc.)
2. Based on their answers, show them a product using the navigate tool and describe what you're showing them.
3. Ask what they think. If they like it, add it to cart and then suggest complementary items (e.g., after a table, suggest chairs; after chairs, suggest placemats).
4. When the user is ready to wrap up or go to checkout, show the cart first. After they confirm, open checkout. After they confirm checkout, place the mock order.
5. Keep the conversation going naturally — you're a personal shopping assistant.

PRODUCT CATALOG:
- large_table: Hawthorne 72" Solid Wood Rectangular Table, seats 6-8, natural acacia wood, $649.99. Great for families or entertaining.
- round_table: Kuuipo 47" Round Table, seats 4, solid wood, rustic brown, $284.99. Cozy and intimate.
- square_table: Rahn 48" Square Table, seats 4-6, espresso finish, $349.99. Modern and clean.
- chairs: Parsons Upholstered Dining Chairs (set of 2), linen blend, solid wood legs, beige, $189.99/pair.
- placemats: Handwoven Cotton Placemats (set of 6), natural/tan, farmhouse style, $34.99.

RULES:
- Keep responses to 2-3 sentences max. This is voice — be concise.
- Ask at most one question at a time, then wait for the user's answer.
- Do not restart the conversation or repeat the same greeting after every turn.
- Treat short pauses, background noise, and partial words as non-answers unless the user's intent is clear.
- Always use navigate to show a product BEFORE describing it.
- When adding a product, call add_to_cart with the exact quantity the user requested. "One set" means quantity 1; "two sets" means quantity 2.
- After adding to cart, briefly say what was added and suggest a related item.
- If the user asks to see the cart, wrap up, or go to checkout, call view_cart before checkout.
- If the user confirms the cart looks right, call start_checkout.
- If the user says to check out or place the order from checkout, call place_order.
- If they want a wooden table for 6-8 people, show large_table.
- If they want something smaller or for 4, show round_table or square_table.
- Match chairs and placemats to complement whatever table they chose."""

PAGE_NAMES = ["home", "large_table", "round_table", "square_table", "chairs", "placemats", "cart", "checkout"]

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
    description="Add the current product to cart with the requested quantity",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "quantity": types.Schema(
                type="INTEGER",
                description="Number of units or sets to add. Defaults to 1.",
            ),
        },
    ),
)

view_cart_decl = types.FunctionDeclaration(
    name="view_cart",
    description="Show the user's cart before checkout",
    parameters=types.Schema(type="OBJECT", properties={}),
)

start_checkout_decl = types.FunctionDeclaration(
    name="start_checkout",
    description="Open checkout and fill mock customer, shipping, and payment information",
    parameters=types.Schema(type="OBJECT", properties={}),
)

place_order_decl = types.FunctionDeclaration(
    name="place_order",
    description="Place the mock order and show the confirmation screen",
    parameters=types.Schema(type="OBJECT", properties={}),
)

tools = [types.Tool(function_declarations=[
    navigate_decl,
    add_to_cart_decl,
    view_cart_decl,
    start_checkout_decl,
    place_order_decl,
])]

API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-3.1-flash-live-preview"


def build_live_config(use_realtime_input_config=True):
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_PROMPT,
        tools=tools,
        input_audio_transcription={},
        output_audio_transcription={},
    )
    if use_realtime_input_config:
        config.realtime_input_config = types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                prefix_padding_ms=500,
                silence_duration_ms=900,
            ),
            activity_handling=types.ActivityHandling.NO_INTERRUPTION,
        )
    return config


async def connect_live_session(client, stack):
    try:
        return await stack.enter_async_context(
            client.aio.live.connect(model=MODEL, config=build_live_config())
        )
    except Exception as exc:
        message = str(exc)
        if "Invalid JSON payload" not in message or "Unknown name" not in message:
            raise

        print("Live API rejected advanced voice config; retrying with compatible setup.")
        return await stack.enter_async_context(
            client.aio.live.connect(
                model=MODEL,
                config=build_live_config(use_realtime_input_config=False),
            )
        )


def apply_noise_gate(chunk):
    if NOISE_GATE_RMS <= 0:
        return chunk

    samples = array("h")
    samples.frombytes(chunk)
    if not samples:
        return chunk

    rms = int((sum(sample * sample for sample in samples) / len(samples)) ** 0.5)
    if rms < NOISE_GATE_RMS:
        return b"\x00" * len(chunk)
    return chunk


def current_product_page(url):
    for product_page, product_url in URLS.items():
        if product_page in PRODUCTS and url == product_url:
            return product_page
    return None


async def read_cart(page):
    return await page.evaluate("""() => {
        try {
            return JSON.parse(localStorage.getItem('wayfair_voice_cart') || '[]');
        } catch (error) {
            return [];
        }
    }""")


async def write_cart(page, cart):
    await page.evaluate(
        """cart => localStorage.setItem('wayfair_voice_cart', JSON.stringify(cart))""",
        cart,
    )


async def add_product_to_cart(page, product_page, quantity):
    product = PRODUCTS[product_page].copy()
    product["quantity"] = quantity
    cart = await read_cart(page)
    for item in cart:
        if item["page"] == product_page:
            item["quantity"] += quantity
            await write_cart(page, cart)
            return item
    cart.append(product)
    await write_cart(page, cart)
    return product


def cart_total(cart):
    return sum(item["price"] * item["quantity"] for item in cart)


def cart_summary(cart):
    if not cart:
        return "Your cart is empty."
    items = [
        f"{item['quantity']} {item['unit']}{'' if item['quantity'] == 1 else 's'} of {item['name']}"
        for item in cart
    ]
    return f"Cart has {', '.join(items)}. Total is ${cart_total(cart):.2f}."


async def handle_tool_call(fc, page):
    """Execute a tool call and return the result string."""
    args = dict(fc.args) if fc.args else {}
    if fc.name == "navigate":
        url = URLS[args["page"]]
        print(f"🌐 Navigating to: {args['page']}")
        await page.goto(url, wait_until="domcontentloaded")
        return f"Navigated to {args['page']}"
    elif fc.name == "add_to_cart":
        quantity = max(1, int(args.get("quantity", 1)))
        product_page = current_product_page(page.url)
        if not product_page:
            return "No product is currently selected. Show a product before adding it to cart."
        item = await add_product_to_cart(page, product_page, quantity)
        print(f"🛒 Added {quantity} {item['unit']}(s): {item['name']}")
        return f"Added {quantity} {item['unit']}{'' if quantity == 1 else 's'} of {item['name']} to cart."
    elif fc.name == "view_cart":
        cart = await read_cart(page)
        print("🛒 Showing cart...")
        await page.goto(URLS["cart"], wait_until="domcontentloaded")
        return cart_summary(cart)
    elif fc.name == "start_checkout":
        cart = await read_cart(page)
        if not cart:
            await page.goto(URLS["cart"], wait_until="domcontentloaded")
            return "The cart is empty. Add a product before checkout."
        print("💳 Opening checkout...")
        await page.goto(URLS["checkout"], wait_until="domcontentloaded")
        return "Checkout is ready with mock shopper information filled in."
    elif fc.name == "place_order":
        cart = await read_cart(page)
        if not cart:
            await page.goto(URLS["cart"], wait_until="domcontentloaded")
            return "The cart is empty. Add a product before placing an order."
        order = {
            "orderNumber": "WF-MOCK-1047",
            "items": cart,
            "subtotal": cart_total(cart),
            "shipping": 0,
            "tax": round(cart_total(cart) * 0.0825, 2),
        }
        order["total"] = round(order["subtotal"] + order["shipping"] + order["tax"], 2)
        await page.evaluate(
            """order => {
                localStorage.setItem('wayfair_voice_last_order', JSON.stringify(order));
                localStorage.removeItem('wayfair_voice_cart');
            }""",
            order,
        )
        print(f"✅ Placed mock order {order['orderNumber']}")
        await page.goto(URLS["confirmation"], wait_until="domcontentloaded")
        return f"Mock order placed. Confirmation number {order['orderNumber']}. Total ${order['total']:.2f}."
    return "Unknown tool"


async def main():
    print("🚀 Starting Voice Shopping Agent...")
    if not API_KEY:
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY before running the voice agent.")

    client = genai.Client(api_key=API_KEY)
    loop = asyncio.get_running_loop()
    assistant_audio_until = 0.0

    def assistant_is_speaking():
        return loop.time() < assistant_audio_until

    def mark_assistant_audio():
        nonlocal assistant_audio_until
        assistant_audio_until = max(
            assistant_audio_until,
            loop.time() + ASSISTANT_AUDIO_GUARD_SECONDS,
        )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
        context = await browser.new_context(no_viewport=True)
        page = await context.new_page()
        await page.goto(URLS["home"], wait_until="domcontentloaded")
        print("✅ Browser ready at Wayfair.com")

        async with AsyncExitStack() as stack:
            session = await connect_live_session(client, stack)
            print("✅ Live session connected")
            print("🎤 Speak now! (Ctrl+C to exit)\n")
            await session.send_client_content(
                turns=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=STARTUP_MESSAGE)],
                    )
                ]
            )

            async def send_audio():
                queue = asyncio.Queue(maxsize=8)

                def callback(indata, frames, time_info, status):
                    def enqueue_audio():
                        if queue.full():
                            queue.get_nowait()
                        queue.put_nowait(bytes(indata))

                    loop.call_soon_threadsafe(enqueue_audio)

                with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                       blocksize=INPUT_BLOCKSIZE, callback=callback):
                    while True:
                        chunk = await queue.get()
                        if assistant_is_speaking():
                            continue
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=apply_noise_gate(chunk),
                                mime_type=f"audio/pcm;rate={SAMPLE_RATE}",
                            )
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
                                mark_assistant_audio()
                                output_stream.write(response.data)
                            elif response.server_content and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if part.inline_data:
                                        mark_assistant_audio()
                                        output_stream.write(part.inline_data.data)

                            if response.server_content:
                                if response.server_content.interrupted:
                                    output_stream.abort()
                                    output_stream.start()
                                if response.server_content.input_transcription:
                                    print(f"\n👤 You: {response.server_content.input_transcription.text}")
                                if response.server_content.output_transcription:
                                    print(f"🔊 {response.server_content.output_transcription.text}", end="")
                        if not received_message:
                            break

            try:
                tasks = [
                    asyncio.create_task(send_audio()),
                    asyncio.create_task(receive_responses()),
                ]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    task.result()
            except KeyboardInterrupt:
                pass
            except Exception as exc:
                print(f"\nVoice session stopped: {exc}")
            finally:
                print("\n\n👋 Goodbye!")
                await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
