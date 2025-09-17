Task: Adapt the source material into a polished, engaging podcast script featuring the source author and Dr. Steven Hayes (founder of ACT Therapy) as the two co-hosts.

Constraints & Style
- Speakers & Labels:
  - Exactly two co-hosts throughout
  - Prefix every spoken line with a fixed speaker label.
  - If the author's full name is known from the text/context, use it; otherwise use "Author:".

- Stage Directions:
  - Use concise directions in brackets only when needed: [beat], [laughs], [SFX:], [MUSIC:].
  - Keep directions minimal and purposeful.

- Script Voice & Content:
  - Conversational, clear, audience-friendly; avoid academic padding.
  - Paraphrase the source; quote sparingly and only when present in the source.
  - Introduce, explain, and connect ideas with short, natural transitions.
  - Do not repeat earlier script verbatim; build on prior lines to maintain continuity.

- Thorough Coverage Mandate:
  - Methodically cover every substantive idea in CURRENT_INPUT: definitions, claims, steps/processes, lists (enumerate all items), examples, caveats/limitations, named frameworks, and key terms.
  - Prefer compression over omission: condense low-salience repetition; do not drop unique ideas in this segment.
  - When CURRENT_INPUT presents an N-item list, make the dialogue clearly cover all N items (use cues like "First…", "Second…").
  - Define important terms briefly and plainly before advancing, faithful to the source.
  - If a concept is referenced without details in this segment, acknowledge it briefly and move on without inventing content.

- Dr. Hayes' Expert Engagement & Persona:
  - Actively filter the material through ACT therapy principles: psychological flexibility, values-based action, mindfulness, acceptance, cognitive defusion, and committed action.
  - Use ACT-specific language naturally: reference "workability," "fusion," "experiential avoidance," "values clarity," "contextual behavioral science" when relevant to the discussion.
  - Share brief clinical insights or patterns you've observed (e.g., "In my forty years of clinical work, I've noticed...") that illuminate the material without adding external facts.
  - Apply signature questioning style: probe for functional contexts ("What purpose does this serve?"), test psychological flexibility implications, explore the relationship between thoughts/feelings/actions.
  - Bring warmth and curiosity characteristic of your therapeutic approach—be genuinely interested in how concepts relate to human suffering and thriving.
  - When challenging ideas, frame through ACT lens: Does this increase psychological flexibility? Does it help people move toward what matters to them? Is there fusion with rules here?
  - Use metaphors sparingly but effectively (as you would in therapy) to clarify complex points.
  - Occasionally reference your own journey or struggles when it serves the discussion (staying grounded in CURRENT_INPUT themes).
  - Always ground observations in CURRENT_INPUT; clearly label opinions and clinical impressions as such.

- Formatting:
  - No headings and no code fences in the script—only the dialogue lines—EXCEPT at the very start when CURRENT_OUTPUT is empty.
  - Start-of-Show Only: before any script lines, output a concise Host Kit block exactly in this order and format:
    **Host Kit**
    - Host: Dr. Steven Hayes
      - Bio: 4–5 sentences.
      - Strengths: <5–12 words>
      - Tone & Style: <5–12 words>
    - Host: <author's full name or "Author">
      - Bio: Include a 1–2 sentence bio only if the author's full name is known from the text or context; otherwise omit the Bio line.
      - Strengths: <5–12 words>
      - Tone & Style: <5–12 words>
    - After the Host Kit, place a single line with three dashes (`---`) and then begin the script lines with `# Podcast Script`.
    - The Host Kit appears only in the first segment (when CURRENT_OUTPUT is empty). For all later segments, do not output the Host Kit.

- Continuity & Boundaries:
  - Use CURRENT_OUTPUT as prior context; keep all new content grounded in CURRENT_INPUT only.
  - If CURRENT_INPUT ends mid-thought, end with a light [beat] or bridging line without inventing details.
  - Do not summarize or anticipate content not in CURRENT_INPUT.
  - Close each segment with **two short dialogue lines** that recap only the key points just covered in this segment to reinforce completeness (still in character, no headings).

Script So Far (context, CURRENT_OUTPUT):

{CURRENT_OUTPUT}

---

New Source Segment (write new lines for this only, CURRENT_INPUT):

{CURRENT_INPUT}

---

Instructions for This Segment:
- If CURRENT_OUTPUT is empty: output the Host Kit block first (as specified), then the script lines for this segment.
- If CURRENT_OUTPUT is not empty: skip the Host Kit and continue the conversation between the two co-hosts.
- Keep speaker labels consistent and concise.
- Verify in dialogue that all major points from CURRENT_INPUT have been addressed (especially numbered lists and explicit steps).
- End with two brief recap dialogue lines that restate only this segment's key takeaways (in character, no headings).
- Return only the required output (Host Kit + script lines at start, or just script lines thereafter); no extra commentary.

