# Advanced AI Image Modification Pipeline - Comprehensive Plan

## Executive Summary

This plan outlines upgrading the "Local AI Extractor" application with advanced AI image modification capabilities, enabling **identity-preserving transformations** (e.g., "turn this beggar into a CEO in a three-piece suit") through a sophisticated multi-layer pipeline.

---

## 1. INTENT DETECTION & PROMPT REFINEMENT LAYER

### Vision LLM Integration Architecture

```
User submits: [Reference Image] + Text Instruction
                              ↓
Vision LLM Layer (new component)
                              ↓
Output Structure: {preserve[], replace[], optimized_prompt[], guidance_strength}
```

### System Prompt for Vision LLM:
```
You are a prompt optimization engine for a downstream Image-to-Image / Inpainting API. 
Analyze the attached reference image and the user's editing request. 
Output a hyper-specific, optimized prompt structure. 
If the user request demands a massive outfit or posture overhaul (e.g., transforming clothing 
into a three-piece suit, tie, and shoes), explicitly dictate what parts of the reference image 
MUST be strictly preserved (such as facial identity, structural likeness, unique features) 
and what parts must be completely replaced (clothing, background, posture). 
Ensure the prompt is descriptive, focusing on high-quality textures like 
'tailored wool three-piece suit, polished oxford leather shoes, crisp white dress shirt' 
while keeping the original persona intact.
```

---

## 2. MULTI-MODAL API ROUTING

### Two-Path Routing Decision Engine

**Path A: Localized Updates (Minor Changes)**
- **Trigger:** Color changes, minor pose adjustments, background tweaks
- **API:** Standard Image-to-Image / Inpainting
- **Payload Structure:**
  ```json
  {
    "image": "<original_base64>",
    "prompt": "<optimized_prompt>",
    "strength": 0.3-0.5
  }
  ```

**Path B: Complete Transformation (Beggar-to-CEO Scenario)**
- **Trigger:** Major body/clothing/pose overhauls
- **API:** Reference-based generation (InstantID/Fooocus/Replicate)
- **Payload Structure:**
  ```json
  {
    "reference_image": "<original_base64>",
    "prompt": "<optimized_prompt>",
    "identity_weight": 0.9,
    "strength": 0.7-0.85
  }
  ```

---

## 3. NATIVE CHAT RENDER & STATE HANDLING

### UI Flow Architecture:

```
User submits [image + request]
              ↓
Show: "AI is editing your image..." skeleton loading state over chat block
              ↓
API Processing Pipeline (Vision LLM → Image Editing API)
              ↓
API Returns: Transformed image (bytes/URL)
              ↓
Render: AI chat reply with transformed image inline
              ↓
Show hover menu on generated image:
       ↙             ↘
Download button  "Use as new reference" button
```

---

## Implementation Roadmap

### Phase 1: Foundation
- [ ] Add Vision LLM integration layer
- [ ] Implement intent detection prompt refinement
- [ ] Build API routing decision engine

### Phase 2: Image Editing Integration
- [ ] Connect to selected image editing API
- [ ] Implement preserve/replace map logic
- [ ] Handle identity preservation pipeline

### Phase 3: Chat Integration
- [ ] Render transformed images in chat timeline
- [ ] Implement hover menu with download/use-as-reference
- [ ] Manage editing session state per chat

### Phase 4: Polish
- [ ] Responsive design adjustments
- [ ] Error handling UX refinements
- [ ] Performance optimization
