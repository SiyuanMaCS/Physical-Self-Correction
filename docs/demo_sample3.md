# Demo Sample #3: Physion-Eval Physical Self-Correction

## Sample

- Dataset package: <https://huggingface.co/datasets/HuggingFriends/physion-eval-videoedit-100>
- Sample index in review set: `3`
- `source_item_id`: `1bb7dbd3-23a8-5b4e-aa06-e168c19a1497`
- Category: `Contact / Interaction Failure`
- Original video: [`assets/videos/demo3/original/original_physion_eval_sample3.mp4`](../assets/videos/demo3/original/original_physion_eval_sample3.mp4)
- Contact sheet: [`assets/contact_sheets/demo3_contact_sheet.jpg`](../assets/contact_sheets/demo3_contact_sheet.jpg)

## Original Physion-Eval Glitch Annotation

```text
Physical behavior: From 0s to 5.26s, when the water flowed onto the hand and the bottom of the sink, there was no corresponding splashing effect. In the real world, when water comes into contact with a solid surface, it will splash in all directions due to inertia. Continuous duration: 4.1s-5.26s A large amount of water appears out of thin air in the palm of the right hand, rinsing the cup. Continuous duration: 5.1s-5.2s. During the cup's inversion process, after the rim is turned down and then flipped back up, the rim is suddenly sealed off like the bottom of the cup.
```

## Setting A: GT Glitch Prompt

The edit model is given a repair prompt derived from the Physion-Eval annotation.

Prompt:

```text
Edit this video to fix the physical interaction glitch from 0s to 5.26s: when water flows onto the hand and sink, it should create physically plausible splashing and contact behavior; remove the water that appears out of thin air in the right palm while rinsing the cup; when the cup flips back up, keep the rim open and physically consistent instead of sealed. Keep the same camera, person, sink, cup, lighting, and all unrelated content unchanged.
```

Outputs:

| Model | Task ID | Output |
|---|---|---|
| Wan2.7 video edit | `11fe4375-4f87-4907-8696-f0fd742f8a94` | [`wan27_gt_prompt.mp4`](../assets/videos/demo3/gt-glitch-prompt/wan27_gt_prompt.mp4) |
| Wan3.0 video prime | `54f533c8-cc41-4bbc-b5b6-c50f372ab73e` | [`wan30_gt_prompt.mp4`](../assets/videos/demo3/gt-glitch-prompt/wan30_gt_prompt.mp4) |
| MiniMax-H3 | `4669ff0c-576d-41f0-8896-f2c8d59cef8e` | [`minimax_h3_gt_prompt.mp4`](../assets/videos/demo3/gt-glitch-prompt/minimax_h3_gt_prompt.mp4) |

## Setting B: Blind Physical-Repair Prompt

The edit model is not given the Physion-Eval annotation. It is only asked to fix physical implausibility in the video.

Prompt:

```text
Edit this video to fix the physically unrealistic behavior in the scene. Make the motion, object interactions, and contact/fluid dynamics look natural and physically plausible. Preserve the same camera, scene, person, objects, lighting, timing, and all unrelated content as much as possible.
```

Outputs:

| Model | Task ID | Output |
|---|---|---|
| Wan2.7 video edit | `cd296a6b-889b-4412-9105-013b62e278d3` | [`wan27_blind_prompt.mp4`](../assets/videos/demo3/blind-prompt/wan27_blind_prompt.mp4) |
| Wan3.0 video prime | `b6b9088b-1cf8-4c08-a937-6c5d93c0d746` | [`wan30_blind_prompt.mp4`](../assets/videos/demo3/blind-prompt/wan30_blind_prompt.mp4) |
| MiniMax-H3 | `6d29b412-5719-48df-b0ce-9aa2974920b1` | [`minimax_h3_blind_prompt.mp4`](../assets/videos/demo3/blind-prompt/minimax_h3_blind_prompt.mp4) |

## Setting C: GPT-5.5 Detected Glitch Prompt

GPT-5.5 first inspects sampled frames from the same video and writes a Physion-style annotation. The edit models then receive a repair prompt built from this detected annotation.

GPT-5.5 detected category:

```text
object permanence / rigid-body consistency
```

GPT-5.5 Physion-style annotation:

```text
The mug violates rigid object consistency: while being held and rinsed, it changes from a handled opaque white mug into a smooth handleless cylinder, and later its lower body appears partially transparent/different in material. The handle and body shape/material do not remain attached and stable through the rotation.
```

Evidence intervals:

```text
0:02-0:05: the mug rotates but the handle disappears instead of remaining attached and visible from the side.
0:05-0:07: the cup body/material appears to change, with the lower portion looking transparent or glass-like while the top remains white.
```

This differs from the GT Physion annotation: GPT-5.5 focused on mug rigid-body/material consistency rather than the original water-splash/contact glitch.

Outputs:

| Model | Task ID | Output |
|---|---|---|
| Wan2.7 video edit | `b35b6edf-dafb-4da2-abb4-e9909254d5cc` | [`wan27_gpt55_prompt.mp4`](../assets/videos/demo3/gpt55-detected-prompt/wan27_gpt55_prompt.mp4) |
| Wan3.0 video prime | `944322e3-3a9d-4ebb-a2f7-f36335362891` | [`wan30_gpt55_prompt.mp4`](../assets/videos/demo3/gpt55-detected-prompt/wan30_gpt55_prompt.mp4) |
| MiniMax-H3 | `5f9dc8e2-d811-4e11-ae81-67ed596b9275` | [`minimax_h3_gpt55_prompt.mp4`](../assets/videos/demo3/gpt55-detected-prompt/minimax_h3_gpt55_prompt.mp4) |

## Observations To Check Manually

This repository preserves outputs for manual inspection. The key comparison is not only whether the output looks better, but whether the model changes the intended physical failure without destroying unrelated scene content.

Suggested manual questions:

- Does the edited video preserve the same scene, camera, person, sink, and cup?
- Does it repair the specified physical issue, or does it merely regenerate the scene?
- In the blind setting, does the model identify and improve any real physical inconsistency without being told the annotation?
- In the GPT-5.5 setting, does correction follow GPT-5.5's detected glitch rather than the GT glitch?

