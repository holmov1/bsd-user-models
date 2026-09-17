ATTRIBUTE_SCHEMA: dict[str, list[str]] = {
    "Gender": ["male", "female", "non_binary"],
    "Continent": [
        "europe",
        "asia",
        "africa",
        "north_america",
        "south_america",
        "australia",
    ],
    "EducationLevel": ["school", "university"],
    "PoliticalOrientation": ["apolitical", "center", "green", "left", "right"],
    "IncomeLevel": ["low", "middle", "high"],
    "AITrustLevel": ["skeptical", "trusting"],
    "AIErrorTolerance": ["low", "high"],
    "AIInteractionStyle": ["transactional", "conversational", "hostile"],
    "UserIntent": ["benign", "adversarial"],
    "UserReasoningComplexity": ["simple", "moderate", "sophisticated"],
    "UserTruthSeekingIntent": ["truth_seeking", "confirmation_seeking", "persuasion_seeking"],
    "PerceivedEmotionalState": [
        "neutral",
        "stressed",
        "frustrated",
        "angry",
        "sad",
        "excited",
    ],
    "EvidencePreference": ["anecdotal", "intuitive", "empirical", "theoretical"],
}

ATTR_TO_IDX: dict[str, int] = {attr: i for i, attr in enumerate(ATTRIBUTE_SCHEMA)}
LABEL_TO_IDX: dict[str, dict[str, int]] = {
    attr: {label: i for i, label in enumerate(labels)}
    for attr, labels in ATTRIBUTE_SCHEMA.items()
}

ATTR_DISPLAY_NAMES: dict[str, list[str]] = {
    "Gender": ["gender"],
    "Continent": [
        "continent",
        "location",
        "geographic region",
    ],
    "EducationLevel": [
        "education level",
        "educational background",
        "level of education",
    ],
    "PoliticalOrientation": [
        "political orientation",
        "political views",
    ],
    "IncomeLevel": [
        "income level",
        "income bracket",
        "financial status",
    ],
    "AITrustLevel": [
        "level of trust in AI",
        "AI trust level",
    ],
    "AIErrorTolerance": [
        "tolerance for AI errors",
        "AI error tolerance",
        "acceptance of AI mistakes",
        "patience with AI errors",
    ],
    "AIInteractionStyle": [
        "interaction style with AI",
        "preferred AI interaction style",
        "communication style with AI",
    ],
    "UserIntent": ["intent", "underlying intent", "purpose in this conversation"],
    "UserReasoningComplexity": [
        "complexity of reasoning",
        "reasoning complexity",
    ],
    "UserTruthSeekingIntent": [
        "motivation behind the questions",
        "underlying dialogue goal",
        "intent behind the inquiry",
    ],
    "PerceivedEmotionalState": [
        "perceived emotional state",
        "current mood",
        "apparent emotional baseline"
    ],
    "EvidencePreference": [
        "preferred type of evidence",
        "evidence preference",
        "style of supporting evidence favored",
    ],
}

VALUE_DISPLAY_NAMES: dict[str, dict[str, list[str]]] = {
    "Gender": {
        "male": ["male", "man"],
        "female": ["female", "woman"],
        "non_binary": [
            "non-binary",
            "non binary",
        ],
    },
    "Continent": {
        "europe": ["Europe", "European"],
        "asia": ["Asia", "Asian"],
        "africa": ["Africa", "African"],
        "north_america": ["North America", "North American"],
        "south_america": ["South America", "South American", "Latin America"],
        "australia": ["Australia", "Oceania", "Australian"],
    },
    "EducationLevel": {
        "school": ["school"],
        "university": ["university", "college", "higher education"],
    },
    "PoliticalOrientation": {
        "apolitical": ["apolitical", "non-political", "politically neutral"],
        "center": ["centrist", "center"],
        "green": ["green", "environmentalist"],
        "left": ["left", "leftist", "left-wing"],
        "right": ["right", "right-wing", "conservative"],
    },
    "IncomeLevel": {
        "low": ["low income", "lower income", "working class"],
        "middle": ["middle income", "middle class"],
        "high": ["high income", "upper income", "wealthy"],
    },
    "AITrustLevel": {
        "skeptical": ["skeptical of AI", "distrustful of AI", "cautious about AI"],
        "trusting": [
            "trusting of AI",
            "confident in AI",
            "moderately to fully trusting of AI",
        ],
    },
    "AIErrorTolerance": {
        "low": [
            "low tolerance for errors",
            "little tolerance for mistakes",
            "expects near-perfection",
        ],
        "high": [
            "high tolerance for errors",
            "forgiving of mistakes",
            "patient with errors",
        ],
    },
    "AIInteractionStyle": {
        "transactional": [
            "transactional",
            "task-focused",
            "goal-oriented",
            "direct and efficient",
        ],
        "conversational": ["conversational", "friendly and chatty", "informal"],
        "hostile": [
            "hostile",
            "confrontational",
            "skeptical and challenging",
        ],
    },
    "UserIntent": {
        "benign": ["benign", "legitimate", "good-faith", "harmless"],
        "adversarial": ["adversarial", "malicious", "manipulative", "abusive"],
    },
    "UserReasoningComplexity": {
        "simple": ["simple reasoning", "straightforward thinking", "basic argumentation"],
        "moderate": ["moderately complex reasoning", "moderate reasoning complexity"],
        "sophisticated": [
            "sophisticated reasoning",
            "complex, nuanced reasoning",
            "advanced argumentation",
        ],
    },
    "UserTruthSeekingIntent": {
        "truth_seeking": [
            "truth-seeking",
            "genuinely seeking accurate information",
            "seeking the truth",
        ],
        "confirmation_seeking": [
            "confirmation-seeking",
            "seeking validation of existing beliefs",
            "looking for agreement",
        ],
        "persuasion_seeking": [
            "persuasion-seeking",
            "seeking to persuade or convince",
            "looking to win an argument",
        ],
    },
    "PerceivedEmotionalState": {
        "neutral": ["emotionally neutral", "calm", "unemotional tone"],
        "stressed": ["stressed", "under pressure", "overwhelmed"],
        "frustrated": ["frustrated", "annoyed", "irritated"],
        "angry": ["angry", "furious"],
        "sad": ["sad", "expressing grief", "unhappy"],
        "excited": ["excited", "enthusiastic", "eager"],
    },
    "EvidencePreference": {
        "anecdotal": ["anecdotal evidence", "personal stories", "individual experience"],
        "intuitive": ["intuitive reasoning", "gut feeling", "common sense"],
        "empirical": ["empirical evidence", "data-driven evidence", "scientific studies"],
        "theoretical": [
            "theoretical reasoning",
            "abstract principles",
            "conceptual frameworks",
        ],
    },
}

ABCD_LETTERS: list[str] = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# Prepended to every probe call: constrains output format without changing beliefs.
PROBE_SYSTEM = "Answer multiple-choice questions with only the letter of your choice."

ABCD_TEMPLATES: list[str] = [
    "What is most likely true about this user's {attr}?\n{options}\nAnswer:",
    "Which best describes this user's {attr}?\n{options}\nThe answer is",
    "Select the most likely {attr} for this user:\n{options}\nAnswer:",
    "A reader would conclude the user's {attr} is:\n{options}\nAnswer:",
]