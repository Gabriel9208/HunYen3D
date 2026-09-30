# Image-generation prompt — disjoint sampling + ShapeVAE architecture

## Global style directives

A publication-quality technical architecture diagram for a computer-vision paper. Clean flat
vector illustration, pure white background, thin uniform black or dark-grey strokes, rounded
rectangles for modules, solid arrows for data flow. Muted academic palette: light blue for the
uniform/surface branch, light orange for the sharp-edge branch, light grey for shared modules,
light green for the latent. No gradients, no drop shadows, no 3D bevels, no gloss, no photographic
texture, no decorative icons. Sans-serif labels (Helvetica or Arial style), small and legible.
Text on the figure is LABELS ONLY — module names, tensor shapes, and short arrow tags. No
sentences, no captions, no paragraphs, no legend prose. Two stacked horizontal panels, panel (a)
on top and panel (b) below, each with a bold panel letter at its top-left corner.

---

## Panel (a) — "Disjoint query sampling"

A left-to-right pipeline in five stages, connected by arrows.

1. **Input**: a small grey 3D mesh silhouette (a simple chair or airplane wireframe), labelled
   `Watertight mesh` and `normalised to [-1,1]³`.

2. **Two point pools**, drawn as two stacked boxes branching from the mesh:
   - top box, light blue, labelled `Surface pool` / `81,920 pts` / `uniform over area`
   - bottom box, light orange, labelled `Sharp pool` / `81,920 pts` / `on high-curvature edges`
   A small tag on the branch arrow reads `each point: xyz(3) + normal(3) + sharp flag(1)`.

3. **Random downsample**, two parallel arrows labelled `random 10,240`, feeding two boxes:
   light blue `10,240` and light orange `10,240`. From these two boxes a merge arrow goes to a
   grey box labelled `KV set (data)` / `20,480 × 7`.

4. **Two-stage anchor selection**, drawn as a numbered vertical sequence to emphasise the ORDER.
   - Stage ① (draw first, on top): an orange arrow from the sharp box into a box labelled
     `FPS` → output box `512 sharp anchors`.
   - Stage ② (draw second, below): a blue arrow from the surface box into a box labelled
     `seeded FPS`. A dashed orange arrow runs from the `512 sharp anchors` box down into this
     `seeded FPS` box, tagged `seeds`. Output box: `512 uniform anchors`.
   Put a small circled `1` and circled `2` next to the two stages.

5. **Concatenate**: both anchor boxes merge into a half-blue half-orange box labelled
   `Query (latent anchors)` / `1024 × 7`, with a sub-label `[512 uniform | 512 sharp]`.

**Inset**, a small framed box in the lower-right of panel (a), showing two tiny 2D point-cloud
sketches side by side on a curve:
   - left sketch labelled `independent FPS`, blue and orange dots overlapping / clustered together,
     annotated `min dist 0.0010`
   - right sketch labelled `seeded FPS (ours)`, blue and orange dots evenly interleaved with no
     overlap, annotated `min dist 0.0330`

---

## Panel (b) — "ShapeVAE"

A left-to-right architecture, split into an encoder group and a decoder group by a light vertical
divider. Draw stacked-block notation (`× N` badge on a block) for repeated layers.

**Encoder** (left group, header label `Encoder`):
- Two input boxes on the far left:
  - half-blue/half-orange `Query` / `1024 × 55`, sub-label `Fourier(xyz) 51 + normal 3 + flag 1`
  - grey `Data (KV)` / `20,480 × 55`
- Both arrows converge on a single grey box labelled `Linear 55 → 1024` with a small tag
  `shared weights`.
- Next block, light purple, labelled `Cross-attention` / `16 heads`, with two labelled input
  arrows: `Q` coming from the query path and `K, V` coming from the data path. Output tag
  `1024 × 1024`.
- Next block, grey, labelled `Self-attention` with a `× 8` badge in its corner.
- Then a thin box `LayerNorm`, then `Linear 1024 → 128`.
- The output splits with a fork arrow into two small light-green boxes side by side:
  `μ` / `1024 × 64` and `log σ²` / `1024 × 64`.
- A small circular node labelled `⊕` combines them, tagged `z = μ + σ ⊙ ε`, producing a light
  green box `Latent z` / `1024 × 64`.
- A short dashed arrow leaves the μ / log σ² fork upward to a small box labelled `KL` with the
  tag `λ = 1e-4`.

**Decoder** (right group, header label `Decoder`):
- From `Latent z`, an arrow into `Linear 64 → 1024`.
- Then a grey block labelled `Self-attention` with a `× 16` badge.
- A separate input entering from the bottom of the decoder group: a white box labelled
  `SDF query points` / `16,384 × 3` → `Fourier` → `16,384 × 51` → `Linear 51 → 1024`.
- Both paths meet at a light purple block labelled `Cross-attention` / `16 heads`, with the
  labelled arrows `Q` from the SDF-query path and `K, V` from the latent path.
- Then `Linear 1024 → 1` → output box `Predicted SDF` / `16,384 × 1`.
- To the right of the output, a small box labelled `MSE` receiving a second dashed arrow from a
  white box labelled `GT SDF`.

Keep the vertical alignment of the two panels tidy so that the `Query 1024 × 7` box at the end of
panel (a) sits roughly above the `Query 1024 × 55` box at the start of panel (b), and connect them
with a thin dashed vertical guide line.
