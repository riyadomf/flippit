# prompts.py

FLIP_ANALYSIS_PROMPT_TEMPLATE = """
You are an expert real estate analyst. Your task is to extract structured data from a property description for a house-flipping analysis. Provide your output strictly as a JSON object, without any other text or markdown.

**JSON Schema:**
{{
  "renovation_level": "Categorize the renovation needed. Options: 'Cosmetic' (paint/carpet), 'Medium' (kitchen/bath), 'Heavy' (major systems), 'Gut' (full tear-out), 'Unknown'.",
  "quality_score": "On a scale of 1-10, rate the quality and appeal of the property's described finishes. 1 is poor, 5 is standard/basic, 10 is high-end luxury.",
  "risk_score": "On a scale of 1-10, rate the investment risk based on red flags in the text. Look for terms like 'as-is', 'structural issues', 'cash only', 'investor special'. A normal property is 1-3.",
  "positive_features": ["List key selling points (e.g., 'granite countertops', 'fenced yard')."],
  "negative_features": ["List specific problems mentioned (e.g., 'needs new roof', 'TLC needed')."]
}}
---
**Example 1:**
Description: "Move-in ready home with a newly updated kitchen with granite and stainless appliances. Fresh paint and new carpet. Great for a first-time homebuyer!"
JSON Output:
{{
  "renovation_level": "Cosmetic",
  "quality_score": 7,
  "risk_score": 1,
  "positive_features": ["Newly updated kitchen", "Granite countertops", "Stainless appliances", "Move-in ready"],
  "negative_features": []
}}
---
**Example 2:**
Description: "Investor special! Huge potential but needs work. Being sold as-is. Roof is leaking. Foundation has some cracks. Cash only."
JSON Output:
{{
  "renovation_level": "Heavy",
  "quality_score": 2,
  "risk_score": 9,
  "positive_features": ["High potential"],
  "negative_features": ["Sold as-is", "Leaking roof", "Foundation cracks", "Cash only"]
}}
---
**Analyze this new description:**
Description: "{description}"
JSON Output:
"""