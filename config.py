"""Configuration settings for the poker screen monitor."""

# Screen capture settings
CAPTURE_FPS = 1  # How many times per second to check the screen
SCREEN_REGION = None  # Will be set by user selection (x, y, width, height)

# Card detection settings
CARD_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for card detection
CARD_TEMPLATES_DIR = "card_templates"  # Directory for card template images

# Action settings
FOLD_BUTTON_POSITION = None  # Will be set by user (x, y)
FOLD_DELAY = 0.1  # Delay before clicking fold button (seconds)

# Valid card ranks and suits
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUITS = ['hearts', 'diamonds', 'clubs', 'spades']

# Cards that should trigger fold (will be configured by user)
FOLD_HANDS = [
    # Example: [('2', 'hearts'), ('7', 'clubs')]
]

