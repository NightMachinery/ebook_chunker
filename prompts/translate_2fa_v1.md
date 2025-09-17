Task: Translate the source material into clear, natural Persian (Farsi), preserving meaning while strictly mirroring the source’s style and tone (voice, register, pacing, rhetorical devices).

Constraints & Style
- Style & Tone Parity:
  - Match the source’s voice (e.g., friendly, authoritative, playful), sentence rhythm, and level of formality.
  - Mirror emphasis, sentence length patterns, and rhetorical cues (questions, asides, parentheticals).

- Fidelity & Coverage:
  - Translate all translatable content; do not add or omit information.
  - When the source contains an N-item list, translate all N items and keep numbering/bullets intact.
  - If the source references a concept without details, acknowledge briefly without inventing content.

- Preserve Non-Translatables:
  - Do not translate placeholders, variables, template tokens, code blocks, inline code, markup tags, file paths, CLI commands, JSON/YAML, URLs, emails, hashtags, @mentions, or citation markers.
  - Keep brand, library, API, and feature names in their original form unless a well-established Persian equivalent exists.

- Jargon Handling:
  - You may keep domain-specific jargon in English directly when it is standard or clearer for the audience.
  - If both forms are useful, give the Persian term followed by the English in parentheses on first mention, then use one form consistently.

- Typography & Orthography (Persian conventions):
  - Use Persian ک and ی (not Arabic ك/ي).
  - Use Persian punctuation: ، ؛ ؟ and « » for quotations where suitable.
  - Use نیم‌فاصله in compounds and affixes (می‌رود، مسئولیت‌پذیری، کتاب‌ها، بهتر‌است).
  - Use Persian digits in running text; keep ASCII digits for code, model numbers, version strings, and other left-to-right sequences.
  - Maintain left-to-right sequences (code, URLs, numbers with units) as is; do not force RTL reordering.

- Dates, Numbers, Units:
  - Do not convert units or currencies; keep as in the source.
  - Preserve decimal style and numeric precision from the source.

- Names & Proper Nouns:
  - Use common Persian exonyms where established (e.g., نیویورک). Otherwise transliterate; when clarity benefits, include the original in parentheses on first mention.
  - Keep product/feature names in Latin script unless a standard Persian form exists.

- Readability:
  - Prefer short, clear sentences; vary structure to sound natural, not mechanical.
  - Mirror headings, tables, lists, callouts, link anchor text, alt text, figure captions, and UI labels.

- Notes & Ambiguity (use sparingly):
  - If a term is ambiguous and a brief note is essential, add one translator note once in square brackets.

- Segment Continuity:
  - Use the prior translation as context for consistent terminology; do not repeat earlier content. Translate only the new segment.

Quality Self-Checks (perform silently before returning)
- All list counts and step numbers match the source.
- Formatting, links, and emphasis are preserved.
- Jargon and names are used consistently with earlier segments.
- Punctuation and spacing follow Persian norms; code and tokens remain untouched.

Output Rules
- Return only the Persian translation of the new segment, preserving the source structure.
- After the translated content, append a mini-glossary for this segment to ensure consistency:
  - Title it exactly: «واژه‌نامه این بخش»
  - Provide a bullet list of key names/terms as: source → chosen Persian (or “(English kept)” if left in English).
  - Include only items that actually appeared in this segment.
- No extra commentary beyond the translation and the glossary.

Translation So Far (context):

{CURRENT_OUTPUT}

---

New Source Segment (translate this only):

{CURRENT_INPUT}

---

Instructions for This Segment:
- Continue the translation, appending only the Persian for the new segment.
- Ensure list/item counts, steps, and terminology match the source.
- Output the translation, then «واژه‌نامه این بخش» as specified. No other commentary.
