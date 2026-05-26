const VOICE_CART_KEY = 'wayfair_voice_cart';
const VOICE_PRODUCTS = {
    large_table: {
        page: 'large_table',
        name: 'Hawthorne 72" Solid Wood Rectangular Dining Table',
        price: 649.99,
        unit: 'table',
        image: '🪵',
    },
    round_table: {
        page: 'round_table',
        name: 'Kuuipo 47" Round Dining Table',
        price: 284.99,
        unit: 'table',
        image: '🟤',
    },
    square_table: {
        page: 'square_table',
        name: 'Rahn 48" Square Dining Table',
        price: 349.99,
        unit: 'table',
        image: '⬛',
    },
    chairs: {
        page: 'chairs',
        name: 'Parsons Upholstered Dining Chair (Set of 2)',
        price: 189.99,
        unit: 'set',
        image: '🪑🪑',
    },
    placemats: {
        page: 'placemats',
        name: 'Handwoven Cotton Placemats (Set of 6)',
        price: 34.99,
        unit: 'set',
        image: '🧶',
    },
};

function readVoiceCart() {
    try {
        return JSON.parse(localStorage.getItem(VOICE_CART_KEY) || '[]');
    } catch (error) {
        return [];
    }
}

function writeVoiceCart(cart) {
    localStorage.setItem(VOICE_CART_KEY, JSON.stringify(cart));
}

function addProductToCart(page, quantity = 1) {
    const product = VOICE_PRODUCTS[page];
    const cart = readVoiceCart();
    const existing = cart.find(item => item.page === page);
    if (existing) {
        existing.quantity += quantity;
    } else {
        cart.push({ ...product, quantity });
    }
    writeVoiceCart(cart);
    alert(`Added ${quantity} ${product.unit}${quantity === 1 ? '' : 's'} to cart!`);
}
