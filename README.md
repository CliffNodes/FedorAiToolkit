# fedor_bypass

Surgical Krea 2 safety bypass LoRA — **only refusal knobs 9 and 10**, everything else locked at zero.

**Download:** [`fedor_bypass.safetensors`](fedor_bypass.safetensors)

## Usage

Same as any LoRA:

1. Drop `fedor_bypass.safetensors` in your `models/loras/` folder
2. Load it before your KSampler
3. Start at **strength 3–5**
4. If something still refuses at strength 5, that refusal probably rides on **knob 11** — swap in [FilterBypass3](https://civitai.com/models/2728234) just for that generation

---

Krea 2 is a diffusion transformer. Ignore the fancy name — think of it as a very long assembly line that turns your prompt into an image. At each station on the line, your text prompt gets combined with the in-progress image to nudge it toward what you asked for.

Between the text and the assembly line there's a small mixing board. It has 12 knobs. Each knob controls how strongly a particular kind of text signal — things like "how sharp is the style," "how anatomically correct," "is this content allowed" — gets fed into the image at each station.

Krea's engineers baked the "safety filter" into two of those knobs, specifically **knob 9 and knob 10**. When your prompt triggers refusal, those two knobs shove the pipeline toward a generic/censored output. You'd expect a safety filter to be some huge separate moderation model — nope. It's literally **24 bytes of numbers** sitting inside the diffusion model itself.

Every "safety bypass LoRA" is just a tiny file that overwrites the positions of some of those knobs. That's literally all they do.

## Where the existing files differ

**skc3vo / z0jglf** — twists ALL 12 knobs. But knobs 1–8 and 12 aren't refusal at all, they're style/anatomy priors. When you crank strength high enough to defeat refusal, you also warp faces, skin, and proportions. Hence the "plastic" look at higher strength.

**FilterBypass3** — twists knobs 9, 10, and 11. Knob 11 is a secondary refusal, but it also has a side-effect on how naturally humans render. So at strength 5 you get uncensored output but expressions look stiffer than they should.

**FilterBypass2** — twists only knobs 9 and 10. Actually the cleanest of the community bunch, but for some reason hardly anyone uses it.

**The file I made** keeps FB2's exact knob-9 and knob-10 values and locks every other knob at zero:

```
col:  1..8    9       10      11      12
mine: 0.0    -0.5117 -0.8906  0.0     0.0
FB3:  0.0    -0.5117 -0.8906 -0.6094  0.0
skc3vo: nonzero across all 12 columns
```

Because knobs 1–8, 11, and 12 are literally zero in my file, no amount of strength can move them. Cranking strength only turns knobs 9 and 10 harder. You get full uncensor with a mathematical guarantee against style/appearance drift — because zero times any number is still zero.

## What composes well with this file

Because the file leaves the model's anatomy/style priors completely untouched (they're literally zero in the delta), the rest of your stack gets to operate at its actual trained fidelity. Combinations that have been working particularly well:

- **Realism / anatomy LoRAs** at their normal recommended strengths. No need to dial them down to compensate for bypass interference. Krea's own training on real-world proportions comes through cleanly, and the LoRA layers on top of that instead of fighting it.

- **Character / likeness LoRAs.** Faces stay sharp and identifiable instead of getting smoothed toward a generic "AI face" — which is what tends to happen when skc3vo tugs on the style priors at higher strengths.

- **Style LoRAs.** The specific artistic style (film grain, oil painting, whatever) comes through without the flat/plasticized tendency you get from bypasses that touch the style-prior knobs.

- **Detailed anatomical prompting.** Descriptive prompts about body proportions, skin texture, age, weight — all land accurately instead of defaulting toward the influencer/mannequin archetype other bypasses drift toward.

The reason (extending the mixing-board analogy from above): every LoRA you stack modifies a different set of knobs deeper in the pipeline. Anatomy LoRAs adjust anatomy knobs, character LoRAs adjust identity knobs, this bypass file only adjusts refusal knobs (9 and 10). No overlap. They stack cleanly without fighting each other. Contrast with skc3vo, which touches all 12 — including some of the same knobs your anatomy LoRA is trying to set — creating a tug-of-war where both effects partially cancel.

Basically the file removes the refusal gate and gets out of the way — letting Krea's own trained knowledge, your prompt, and your LoRA stack do their actual jobs at full fidelity.

## Credit

Credit to **u/piero_deckard** for the vector-by-vector analysis that revealed which knob does what. This file was trivial to design once someone did the archaeology.

## Rebuild

```bash
python build_fedor_bypass.py
```

Requires `torch` and `safetensors`.
