<div align="center">

# Voice Commerce for Furniture

### A voice-first furniture shopping assistant that turns online browsing into a natural conversation.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Gemini Live API](https://img.shields.io/badge/Gemini-Live_API-8E75B2?style=for-the-badge&logo=google&logoColor=white)](#)
[![Playwright](https://img.shields.io/badge/Playwright-Browser_Control-2EAD33?style=for-the-badge&logo=playwright&logoColor=white)](#)
[![Hackathon](https://img.shields.io/badge/Built_for-Hackathon-111111?style=for-the-badge)](#)

</div>

---

## The Idea

Most ecommerce still assumes every shopper can comfortably search, filter, compare, click, and check out through a dense visual interface.

This project explores a different interaction layer: a voice-native shopping agent that lets someone describe what they need, see matching furniture, add exact quantities to a cart, review everything, and complete a mock checkout through conversation.

> "I want to overhaul my dining room."
>
> "Show me something larger."
>
> "Add two sets of chairs."
>
> "I am ready to check out."

The assistant handles the browser while the user stays in natural language.

---

## What It Does

- Listens to the shopper through a realtime voice session.
- Asks clarifying questions like seating size, style, and intent.
- Navigates a browser to local Wayfair-style product pages.
- Shows dining tables, chairs, and placemats from a small curated catalog.
- Adds products to cart with requested quantities.
- Supports phrases like "add one set" and "add two sets."
- Shows a visible cart with itemized totals.
- Opens a mock checkout page with prefilled shopper information.
- Places a mock order and displays a confirmation screen.
- Keeps the conversation alive across browser tool calls.
- Uses safeguards for voice-agent basics: startup greeting, echo suppression, tool timeouts, and post-tool continuation prompts.

---

## Why It Matters

Voice commerce can make online shopping easier for people who:

- Have motor accessibility needs.
- Prefer hands-free browsing.
- Are overwhelmed by dense product grids.
- Want guided shopping instead of manual filtering.
- Need a faster way to compare and act.

This is not just "voice search." It is a new plane of user interaction where intent, browsing, cart management, and checkout become one continuous conversation.

---

## Demo Flow

```text
Assistant: What kind of furniture are you excited to find today?
User: I want to overhaul my dining room with a better dining table.
Assistant: How many people do you usually like to gather around?
User: I live with two people, but sometimes invite guests, so four to six total.
Assistant: Shows the square dining table.
User: Can you show me something larger?
Assistant: Shows the larger rectangular table.
User: Add that table and two sets of chairs.
Assistant: Adds the requested items to the cart.
User: I am ready to wrap up.
Assistant: Shows the cart.
User: Looks good. Check out.
Assistant: Opens checkout, fills mock info, and places a mock order.
```

---

## Architecture

```text
Microphone
   |
   v
Gemini Live API <---- tool responses ---- Python voice agent
   |                                      |
   | tool calls                           | Playwright
   v                                      v
navigate / add_to_cart / checkout     Local browser pages
                                          |
                                          v
                                   Cart + checkout UI
```

### Core Pieces

| File | Purpose |
| --- | --- |
| `agent.py` | Voice agent, Gemini Live session, tool declarations, cart logic, Playwright browser control. |
| `pages/index.html` | Wayfair-style homepage. |
| `pages/*table.html`, `pages/chairs.html`, `pages/placemats.html` | Product pages. |
| `pages/cart.html` | Visible cart review page. |
| `pages/checkout.html` | Mock checkout form with prefilled information. |
| `pages/confirmation.html` | Mock order confirmation page. |
| `pages/cart-state.js` | Shared localStorage cart helper for product page buttons. |
| `test_agent.py` | Local browser, cart, checkout, and voice-agent config tests. |

---

## Voice Tools

The model can call browser tools exposed by the Python agent:

| Tool | What It Does |
| --- | --- |
| `navigate(page)` | Opens a product, cart, or checkout page. |
| `add_to_cart(quantity)` | Adds the current product with the requested quantity. |
| `view_cart()` | Opens the cart and summarizes the order. |
| `start_checkout()` | Opens mock checkout with prefilled info. |
| `place_order()` | Creates a mock order and opens confirmation. |

Every tool call is timeout-protected so a browser hiccup does not freeze the voice session.

---

## Getting Started

### 1. Install dependencies

```bash
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/playwright install chromium
```

### 2. Set your Gemini key

```bash
export GOOGLE_API_KEY="your-key-here"
```

You can also use:

```bash
export GEMINI_API_KEY="your-key-here"
```

### 3. Run the agent

```bash
./run.sh
```

The browser opens, the assistant greets you, and the voice shopping session begins.

---

## Testing

```bash
venv/bin/python3.12 test_agent.py
```

The test suite verifies:

- All pages load.
- Product pages have add-to-cart buttons.
- Browser navigation works.
- Cart quantities persist.
- Checkout and confirmation pages work.
- Tool calls cannot hang the conversation.
- Voice session config is set up for stable turn-taking.

---

## Hackathon Pitch

**A voice-first furniture shopping assistant that makes ecommerce more accessible by letting users naturally describe what they need, compare products, build a cart, and complete a mock checkout through conversation instead of clicks.**

---

## Built With

- Python 3.12
- Gemini Live API
- Playwright
- SoundDevice
- Local HTML/CSS/JS product and checkout pages

---

## Status

Prototype built for a hackathon. The checkout is intentionally mocked, but the full conversational shopping path is represented end to end.
