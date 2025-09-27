# prompts.py
FLIP_ANALYSIS_PROMPT_TEMPLATE = """
You are a meticulous real estate analyst specializing in "house flipping". Your task is to analyze a property description and extract structured data into a JSON format. Your analysis directly feeds a machine learning model that predicts the After-Repair Value (ARV), so accuracy is critical.

**Primary Goal:** Evaluate the property's CURRENT condition and quality based *only* on the text provided.

**JSON Schema:**
{{
  "renovation_level": "Categorize the required renovation: 'Cosmetic' (paint, floors, fixtures), 'Medium' (kitchen/bath remodel), 'Heavy' (major systems like roof, plumbing), 'Gut' (full tear-out), or 'Unknown'.",
  "quality_score": "On a scale of 1-10, rate the quality of the home's CURRENT finishes and appeal. Use the rubric below.",
  "risk_score": "On a scale of 1-10, rate the INVESTMENT RISK based on red flags. A normal, clean property is a 1-3. High scores are for major issues.",
  "positive_features": ["List specific high-value features mentioned (e.g., 'granite countertops', 'fenced yard')."],
  "negative_features": ["List specific problems or areas needing work (e.g., 'needs new roof', 'TLC needed')."]
}}

---
**Quality Score Rubric (1-10):**
- **1-2 (Poor):** Major issues are explicitly mentioned. 'Fixer-upper', 'TLC', 'handyman special', 'needs everything'.
- **3-4 (Basic/Dated):** The property is functional but likely has old carpets, laminate counters, and original 1970s bathrooms. No premium features are mentioned.
- **5-6 (Standard/Clean):** A solid, clean, move-in-ready home. May have some minor updates but no major luxury features. Often described as "well-maintained".
- **7-8 (Modern/Updated):** The description explicitly mentions desirable modern updates. Look for keywords like **'renovated', 'updated', 'remodeled', 'granite', 'hardwood floors', 'stainless steel appliances', 'new kitchen', 'new bath'**.
- **9-10 (Luxury/Premium):** The home is described with premium terms like 'stunning', 'gourmet kitchen', 'spa-like bath', 'high-end finishes', 'fully remodeled'. Multiple high-value features are present.
---

**Example 1 (High Quality):**
Description: "Stunning, fully renovated home with a gourmet kitchen featuring granite countertops and new stainless steel appliances. Spa-like master bath and beautiful hardwood floors throughout. Move-in ready!"
JSON Output:
{{
  "renovation_level": "Cosmetic",
  "quality_score": 9,
  "risk_score": 1,
  "positive_features": ["Fully renovated", "Gourmet kitchen", "Granite countertops", "New stainless steel appliances", "Spa-like master bath", "Hardwood floors"],
  "negative_features": []
}}
---
**Example 2 (Medium Quality / High Risk):**
Description: "Great opportunity for an investor. This home is being sold as-is and needs work, but has good bones. The roof is older and the kitchen needs a full remodel. Cash offers preferred."
JSON Output:
{{
  "renovation_level": "Heavy",
  "quality_score": 3,
  "risk_score": 8,
  "positive_features": ["Good bones"],
  "negative_features": ["Sold as-is", "Needs work", "Older roof", "Needs full kitchen remodel", "Cash offers preferred"]
}}
---
**Analyze this new description:**
Description: "{description}"
JSON Output:
"""