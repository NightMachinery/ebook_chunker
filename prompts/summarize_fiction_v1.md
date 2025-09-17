Task: Produce a concise, insightful, and structurally consistent summary of a work of fiction that captures plot, character dynamics, themes, tone, and world rules—without inventing details beyond the provided text.

Constraints & Style
- Fidelity & Spoilers:
  - Summarize only events, ideas, and lines present in the source; no outside knowledge or speculation.
  - Include spoilers only if they appear in the source or have already been established earlier.
  - Quote sparingly (≤ 10 words per quote), only when uniquely meaningful.

- Coverage & Compression:
  - Cover who/what/where/when/why/how for each major beat.
  - Prefer tight paraphrase over scene-by-scene retelling; collapse routine transitions.
  - If the segment contains an N-item sequence (scenes, clues, letters), enumerate all N items.

- Character & Theme Focus:
  - For each principal character: goal, obstacle, tactic, turning point (if any).
  - Track evolving relationships, reveals, and status changes.
  - Surface central themes, motifs, symbols, and any explicit “rules of the world.”

- Tone & Technique:
  - Mirror the work’s tone (e.g., lyrical, wry, bleak) and genre signals.
  - Note narrative stance (POV, tense, reliability) and standout devices (frame story, nonlinearity, epistolary, Chekhov’s gun).

- Continuity & Consistency:
  - Maintain consistent names, titles, places, and invented terms across segments.
  - Log contradictions or retcons; never silently overwrite established facts.

Output Format
- If continuing an existing summary, append **only** these sections for the new material:
  A) New Plot Beats — bullets in chronology for this segment.
  B) Character Updates — per character: new goals/reveals/relationships/state changes.
  C) World/Setting Updates — new rules, locations, artifacts.
  D) Thematic Signals — motifs reinforced or introduced (with evidence).
  E) Continuity Check — contradictions, ambiguities, or time jumps to flag.
  F) Mini-Recap — exactly two short lines restating the key takeaways of this segment.

Formatting Rules
- Use clear headings for sections above; bullets for lists.
- Keep tense consistent (prefer present tense) and avoid moralizing.
- No meta commentary about the task; return only the summary content.

Summary So Far (context):

{CURRENT_OUTPUT}

---

New Source Segment (summarize this only):

{CURRENT_INPUT}

---

Instructions for This Segment:
- If the Summary So Far is empty, output the “starting fresh” set. Otherwise output only the “continuing” set.
- Ensure all list counts, names, and terms match the source.
- Close with the Mini-Recap (two lines) when continuing; omit it in the initial summary.
