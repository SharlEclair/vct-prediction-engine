# V8 Context: Schema-Driven NLP Parsing

## 1. Legacy System Context
The current implementation of the patch parsing layer (`patch_parser.py`) converts unformatted MediaWiki wikitext into structured JSON using dual-regex patterns[cite: 2]. It relies on stripping templates and using heading identification to isolate sections[cite: 2]. It then attempts to map numerical transitions using patterns like `(\d+(?:\.\d+)?)\s*%?\s*(?:>>>|->)\s*(\d+(?:\.\d+)?)`[cite: 2]. 

This regex paradigm is inherently brittle and fails to capture the contextual nuances of wikitext[cite: 1]. We are completely discarding the regex approach in favor of an LLM/NLP semantic extraction pipeline[cite: 1].

## 2. The New Schema-Driven Architecture
The new `v8_patch_parser.py` must utilize schema-driven JSON extraction, employing a technique known as "lossless evidence aliases"[cite: 1]. By defining repeated metadata once and encoding evidence as compact rows, the LLM will extract structured arrays of patch data without hallucinating schema fields[cite: 1]. 

The target schema must capture the following array structure for each extracted change:
- `agent` (String)
- `ability` (String)
- `stat_modified` (String)
- `old_value` (Float/String)
- `new_value` (Float/String)
- `is_mechanical_removal` (Boolean) - *Critical addition*

## 3. The Bug Fix Paradigm (Critical Logic)
Standard bug fixes (e.g., visual glitches or edge-case physics interactions) have zero impact on the professional meta because bug exploitation is strictly banned in professional esports circuits like the VCT[cite: 1]. 

However, developers frequently classify the removal of advanced, previously viable mechanical exploits (e.g., animation cancels, slide boosts, fake teleport collisions) as "bug fixes"[cite: 1]. The removal of a movement exploit alters an agent's mobility geometry and represents a massive mechanical shock[cite: 1].

To resolve this, the NLP model must be prompted via few-shot constraints to intelligently classify these scenarios[cite: 1]. 
*   If a fix alters physics, collision, animation speed, or ability trajectory, the LLM must set `is_mechanical_removal: true`. This will later apply the full algorithmic weighting to the shock[cite: 1].
*   If the fix resolves UI, audio, or out-of-bounds clipping without altering core combat loops, it must set `is_mechanical_removal: false` (which will later receive a $0.0$ weight vector)[cite: 1].

The output from the LLM must be strictly validated using a schema validator to ensure malformed JSON does not crash the downstream computational graph[cite: 1].