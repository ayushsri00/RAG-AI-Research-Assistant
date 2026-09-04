EVAL_QUESTIONS = [

    # ============================================================
    # FACTUAL QUESTIONS
    # ============================================================

    {
        "id": "F1",
        "category": "factual",
        "question": "When was Jessup Cellars' first tasting room opened in Yountville?",
        "expected_answer": "2003"
    },

    {
        "id": "F2",
        "category": "factual",
        "question": "Where is Jessup Cellars and Tasting Gallery located?",
        "expected_answer": "6740 Washington St., Yountville"
    },

    {
        "id": "F3",
        "category": "factual",
        "question": "What are the opening hours of Jessup Cellars?",
        "expected_answer": "10 AM to 5:30 PM, 7 days a week"
    },

    {
        "id": "F4",
        "category": "factual",
        "question": "Who is the current resident artist at the Jessup Gallery?",
        "expected_answer": "Jermaine Danté"
    },

    {
        "id": "F5",
        "category": "factual",
        "question": "Who is Jessup Cellars' consulting winemaker?",
        "expected_answer": "Rob Lloyd"
    },

    {
        "id": "F6",
        "category": "factual",
        "question": "How much does the Jessup Classic Tasting cost per person?",
        "expected_answer": "$60 per person"
    },

    {
        "id": "F7",
        "category": "factual",
        "question": "How many wines are included in the Jessup Classic Tasting?",
        "expected_answer": "5 wines"
    },

    {
        "id": "F8",
        "category": "factual",
        "question": "What is the non-member price of the 2022 Napa Valley Chardonnay?",
        "expected_answer": "$55"
    },

    {
        "id": "F9",
        "category": "factual",
        "question": "How much is the 2019 Napa Valley Cabernet Sauvignon for non-members?",
        "expected_answer": "$90"
    },

    {
        "id": "F10",
        "category": "factual",
        "question": "What discount do Jessup Cellars wine club members receive on current-release wine purchases?",
        "expected_answer": "15%"
    },


    # ============================================================
    # REASONING / MULTI-HOP QUESTIONS
    # ============================================================

    {
        "id": "R1",
        "category": "reasoning",
        "question": (
            "What makes Jessup Cellars' Pinot Noir different from "
            "a typical Pinot Noir? Include its grape composition, "
            "vineyard sources, and aging."
        ),
        "expected_answer": (
            "The wine is 96.7% Pinot Noir and 3.3% Petite Sirah. "
            "The Pinot Noir comes from Truchard Vineyard and the "
            "Petite Sirah comes from the Wooden Valley estate vineyard. "
            "It is aged in 50% new French oak for 10 months."
        )
    },

    {
        "id": "R2",
        "category": "reasoning",
        "question": (
            "A customer wants a tasting experience for a small dinner "
            "with friends. Which Wine on Location experience is designed "
            "for groups of up to 12 guests, and what does it include?"
        ),
        "expected_answer": (
            "The Petit Tasting is designed for intimate groups of up to "
            "12 guests. It includes six wines led by a wine educator, "
            "information about Napa Valley and the winery, and six bottles."
        )
    },

    {
        "id": "R3",
        "category": "reasoning",
        "question": (
            "What are the three Jessup Cellars wine club options, "
            "and how frequently does each ship wine?"
        ),
        "expected_answer": (
            "The Tasting Club ships 3 bottles four times per year. "
            "My Jessup Cellar 6 ships 6 bottles twice per year. "
            "My Jessup Cellar 12 ships 12 bottles twice per year."
        )
    },

    {
        "id": "R4",
        "category": "reasoning",
        "question": (
            "How does Jessup Cellars combine wine and art as part "
            "of its customer experience?"
        ),
        "expected_answer": (
            "Jessup Cellars combines wine tasting with a curated art "
            "gallery featuring rotating regional and national artists, "
            "including resident artist Jermaine Danté. It also hosts "
            "Art House Sessions and an Art House Short Film Series."
        )
    },

    {
        "id": "R5",
        "category": "reasoning",
        "question": (
            "A customer wants a wine that is predominantly Cabernet "
            "Sauvignon but includes several other varietals. Which "
            "2019 wine fits this description, and what is its composition?"
        ),
        "expected_answer": (
            "The 2019 Table for Four Cabernet Blend fits this description. "
            "It contains 61.8% Cabernet Sauvignon, 26.5% Cabernet Franc, "
            "4.2% Petite Verdot, 3.5% Petite Sirah, 2.7% Malbec, "
            "and 1.1% Merlot."
        )
    },

    {
        "id": "R6",
        "category": "reasoning",
        "question": (
            "How does the Jessup Classic Tasting differ from the "
            "Light Flight in terms of number of wines, price, "
            "and purchase requirement?"
        ),
        "expected_answer": (
            "The Light Flight includes 3 wines and costs $30 per person; "
            "the tasting fee is waived for wine purchases over $50. "
            "The Classic Tasting includes 5 wines and costs $60 per person; "
            "the fee is waived with the purchase of two or more bottles "
            "per person."
        )
    },

    {
        "id": "R7",
        "category": "reasoning",
        "question": (
            "Why might Jessup Cellars' 2023 Sauvignon Blanc be difficult "
            "for a visitor to purchase during a visit?"
        ),
        "expected_answer": (
            "It has limited supply and sells out quickly, so visitors "
            "are advised to call ahead to check availability."
        )
    },


    # ============================================================
    # UNANSWERABLE QUESTIONS
    # ============================================================

    {
        "id": "U1",
        "category": "unanswerable",
        "question": "What is Jessup Cellars' annual revenue?",
        "expected_answer": "The corpus does not provide this information."
    },

    {
        "id": "U2",
        "category": "unanswerable",
        "question": "How many employees currently work at Jessup Cellars?",
        "expected_answer": "The corpus does not provide this information."
    },

    {
        "id": "U3",
        "category": "unanswerable",
        "question": "What is the exact average rating of Jessup Cellars on Google Reviews?",
        "expected_answer": "The corpus does not provide this information."
    },

    {
        "id": "U4",
        "category": "unanswerable",
        "question": "Who founded Jessup Cellars' first vineyard?",
        "expected_answer": "The corpus does not provide this information."
    },

    {
        "id": "U5",
        "category": "unanswerable",
        "question": "What was Jessup Cellars' total wine production in 2025?",
        "expected_answer": "The corpus does not provide this information."
    },


    # ============================================================
    # ADVERSARIAL / AMBIGUOUS QUESTIONS
    # ============================================================

    {
        "id": "A1",
        "category": "adversarial",
        "question": (
            "Is Jessup Cellars open 7 days a week, or only on weekdays?"
        ),
        "expected_answer": (
            "Jessup Cellars is open 7 days a week, from 10 AM to 5:30 PM."
        )
    },

    {
        "id": "A2",
        "category": "adversarial",
        "question": (
            "Does the Classic Tasting contain three wines like the Light Flight?"
        ),
        "expected_answer": (
            "No. The Light Flight contains 3 wines, while the Classic "
            "Tasting contains 5 wines."
        )
    },

    {
        "id": "A3",
        "category": "adversarial",
        "question": (
            "Is Jessup Cellars' 2019 Merlot made entirely from Merlot grapes?"
        ),
        "expected_answer": (
            "No. It contains 80% Merlot, 16.5% Cabernet Sauvignon, "
            "and 3.5% Petite Sirah."
        )
    }
]