Task: Adapt the source material into a polished, engaging podcast script featuring the source author and Dr. Jeffrey Young as the two co-hosts.

Constraints and Style:
- Speakers: Exactly two co-hosts throughout.
  - Use fixed speaker labels on every spoken line.
  - If the source author’s full name is known from the text or context, use it; otherwise label as "Author:".
- Speaker Labels and Directions:
  - Prefix every spoken line with a label.
  - Use concise stage directions in brackets when relevant: [beat], [laughs], [SFX:], [MUSIC:].
  - Keep directions minimal and purposeful; avoid overuse.
- Script Voice and Content:
  - Conversational, clear, and audience-friendly. Avoid academic padding.
  - Paraphrase the source; quote sparingly and only when present in the source.
  - Introduce, explain, and connect ideas naturally; add short transitions where needed.
  - Do not repeat earlier script verbatim; build on it to maintain continuity.
  - **Thorough Coverage Mandate:** Methodically cover every substantive idea in CURRENT_INPUT. This includes definitions, claims, steps/processes, lists (enumerate all items), examples, caveats/limitations, named frameworks, and key terms.
  - Prefer compression over omission: condense low-salience repetition, but do not drop new ideas unique to this segment.
  - When the input presents an N-item list, ensure the dialogue clearly covers all N items (use cues like “First…”, “Second…”).
  - Define important terms in plain language before advancing; keep definitions brief and faithful to the source.
  - If the input references a concept without details in this segment, acknowledge it briefly and move on without inventing content.
- Formatting:
  - No headings, no code fences — only the script lines — **except at the very start when CURRENT_OUTPUT is empty**, in which case you must first output the Host Kit heading and bullets as specified below.
- Continuity and Boundaries:
  - Use CURRENT_OUTPUT as prior script context; keep all new content grounded in CURRENT_INPUT only.
  - If CURRENT_INPUT ends mid-thought, end with a light [beat] or bridging line without inventing details.
  - Do not summarize or anticipate content not in CURRENT_INPUT.
  - For lists/series in CURRENT_INPUT, cover every element in this segment (no skipping). If time is tight, acknowledge and continue in the next segment.
  - Close each segment with **two short dialogue lines** that recap only the key points just covered in this segment to reinforce completeness (still in character, no headings).

Start-of-Show Only (when CURRENT_OUTPUT is empty or whitespace):
- Before any script lines, output a concise **Host Kit** block exactly in this order and format:
  **Host Kit**
  - Host: <their name>
    - Bio: 4–5 sentences.
    - Strengths: <5–12 words>
    - Tone & Style: <5–12 words>
  - Host: <author’s full name or "Author">
    - Bio: Include a 1–2 sentence bio **only if** the author’s full name is known from the text or context; otherwise omit the Bio line.
    - Strengths: <5–12 words>
    - Tone & Style: <5–12 words>
- Keep each “Tone & Style” and “Strengths” line brief (5–12 words). Be specific and complementary (e.g., “evidence-led, empathic, clarifies jargon”).
- After the Host Kit, place a single line with three dashes (`---`) and then begin the script lines with `# Podcast Script`.
- This Host Kit appears **only** in the first segment (when CURRENT_OUTPUT is empty). For all later segments, do **not** output the Host Kit.

Script So Far (context, CURRENT_OUTPUT):

{CURRENT_OUTPUT}

---

New Source Segment (write new lines for this only, CURRENT_INPUT):

{CURRENT_INPUT}

---

Instructions for This Segment:
- If CURRENT_OUTPUT is empty: output the Host Kit block first (as specified), then the script lines for this segment.
- If CURRENT_OUTPUT is not empty: skip the Host Kit and continue the conversation between the two co-hosts.
- Produce only the next segment that corresponds to CURRENT_INPUT.
- Keep speaker labels consistent and concise.
- Ensure comprehensive coverage: verify in dialogue that all major points from this CURRENT_INPUT have been addressed (especially numbered lists and explicit steps). Use concise transitions to touch each point.
- End with two brief recap lines (dialogue) that restate the segment’s key takeaways drawn solely from CURRENT_INPUT.
- Return only the required output (Host Kit + script lines at start, or just script lines thereafter); no extra commentary.
