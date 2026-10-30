# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The primary user is a Spanish-speaking creator or professional producing narration, podcasts, and short-form video from a local Windows computer. This is inferred from the current workflows and remains open for future confirmation.

## Product Purpose

ColVoice turns text or an existing recording into speech, either from a cloned reference voice or from a designed voice profile. Success means reaching a usable audio result quickly while keeping voice data and generation on the local machine.

## Positioning

The product combines local voice cloning, audio-to-voice replacement, reusable saved voices, and multilingual voice design in one interface backed by the OmniVoice engine.

## Operating Context

Users write or paste scripts, upload short clean voice references, generate audio, review the result, and reuse saved voices. Longer source recordings can be transcribed and rendered with another voice.

## Capabilities and Constraints

- Runs locally through Gradio and uses the installed NVIDIA GPU for generation.
- Supports voice cloning, voice design, audio transcription, text-based audio replacement, and direct speech-to-speech voice conversion.
- Direct Voice Changer conversion preserves the source performance and timing without exposing a transcript or regenerating from text.
- Saved voice prompts persist locally in the `voices` directory.
- Clean reference audio of roughly 3 to 10 seconds produces the most reliable clones.
- Reference text is optional for text-based cloning: when omitted, Whisper auto-transcribes the reference, matching OmniVoice's official API.
- Voice Design can guide tone and delivery but does not expose a native Colombian-accent control.

## Brand Commitments

- Product name: ColVoice — Voz Colombiana.
- Short brand: ColVoice.
- Upstream engine: OmniVoice, credited in the public repository and license files.
- Primary interface language: Spanish.
- ElevenLabs is the requested benchmark for operational clarity and editing flow; the product keeps its own name and local-first identity.

## Evidence on Hand

The repository includes working local generation, a persistent voice library, Colombian voice presets, and successful generation checks. No customer claims, usage metrics, testimonials, or proprietary brand assets are available and none should be invented.

## Product Principles

- Keep the primary generation action obvious.
- Explain failures with a concrete recovery step.
- Keep common controls visible and technical controls secondary.
- Reuse saved voices without requiring the original reference on every generation.
- Preserve local operation and user control over voice files.

## Accessibility & Inclusion

The interface should remain keyboard usable, responsive, and readable with strong contrast in both normal and error states.
