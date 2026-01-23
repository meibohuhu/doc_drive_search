# SimLingo Training: Architecture Analysis

## Overview

End-to-end autonomous driving model combining vision-language understanding with trajectory prediction. Processes multi-modal inputs (images, language, vehicle state) to predict waypoints and generate language responses.

---

## 1. Input

### 1.1 Input Components

**Camera Images** (`camera_images`)
- Shape: `[B, T, N, C, H, W]` (B=batch, T=1, N=3 cameras, 448×448 patches)
- Preprocessing: Resize, patch extraction, augmentation

**Vehicle Speed** (`vehicle_speed`)
- Shape: `[B, 1]` (m/s)
- **Integration**: Included as **text** in prompt (e.g., "Current speed: 5.0 m/s")

**Target Point** (`target_point`)
- Shape: `[B, 2]` (x, y coordinates)
- **Integration**: Special tokens `<TARGET_POINT><TARGET_POINT>` in prompt, coordinates stored in `placeholder_values`

**Language Prompt** (`prompt`)
- Format: `"Current speed: 5.0 m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Command: turn left."`
- Special tokens:
  - `<IMG_CONTEXT>`: Placeholder for image features (256 tokens)
  - `<TARGET_POINT>`: Placeholder for target point coordinates (2 tokens)
- `placeholder_values`: Dict mapping `<TARGET_POINT>` token ID → coordinates (only contains target_point, not images)

### 1.2 Input Processing Pipeline

```
1. Language Tokenization
   Prompt text → Token IDs → Embeddings
   (Speed as text, <TARGET_POINT> as special tokens)

2. Placeholder Token Replacement
   a) <TARGET_POINT> → wp_encoder(coordinates) → Embeddings
   b) <IMG_CONTEXT> → vision_encoder(images) → Features
   
3. Adaptor Concatenation
   [Language embeddings] + [Driving query embeddings]
```

**Final Input to LLM** (`input_embeds`):
```
[Text tokens (speed as text)] + [Target Point Embeddings] + [Image Features] + [Learnable Queries]
     ↓                              ↓                          ↓                      ↓
"Current speed: 5.0 m/s"      <TARGET_POINT> embeddings   <IMG_CONTEXT> features   Driving queries
```

**Key Points**:
- `input_embeds` = `adaptor_embeds` (same object)
- Speed: Text tokenization (no special encoding)
- Target Point: Encoded via `wp_encoder` MLP, replaces `<TARGET_POINT>` token embeddings
- Images: Vision features replace `<IMG_CONTEXT>` tokens (found by token ID matching, not `placeholder_values`)

---

## 2. Output

### 2.1 Output Components

**Speed Waypoints** (`speed_wps`)
- Shape: `[B, 10, 2]` (2D coordinates)
- Method: Predict **deltas** → `cumsum()` → absolute waypoints

**Route** (`route`)
- Shape: `[B, 20, 2]` (when `predict_route_as_wps=True`)
- Purpose: High-level navigation path

**Language** (`language`)
- Type: `List[str]`
- Generation: Autoregressive, greedy sampling, max 100 tokens

---

## 3. Model Architecture

### 3.1 Architecture Flow

```
Input (Images + Language + State)
    ↓
[Vision Encoder] → Image Features
[Language Adaptor] → Text Embeddings (with placeholders)
[Driving Adaptor] → Query Embeddings
    ↓
[Replace Placeholders]
  - <TARGET_POINT> → wp_encoder(coords)
  - <IMG_CONTEXT> → vision features
    ↓
[Concatenate] → Unified Sequence
    ↓
[Language Model] (InternVL2-1B LLM with LoRA)
    ↓
[Hidden States] → Split by Adaptor
    ↓
[Driving Head] → Waypoint Deltas → Cumsum → Waypoints
[Language Head] → Next Token Logits → Text
```

### 3.2 Key Components

**Vision Model** (`LingoInternVLModel`)
- Base: `OpenGVLab/InternVL2-1B`
- Functions: `extract_feature()`, `replace_placeholder_tokens()`

**Language Model** (`LLM`)
- Base: `OpenGVLab/InternVL2-1B` (LoRA fine-tuning)
- Processes unified embedding sequence

**Waypoint Encoder** (`WaypointInputAdaptor`)
- MLP: `[2] → [256] → [512] → [hidden_size]`
- Encodes target point coordinates for placeholder replacement

**Driving Adaptor** (`DrivingAdaptor`)
- Query Embeddings: Learnable `[1, 10, hidden_size]` (speed_wps) + `[1, 20, hidden_size]` (route)
- Prediction Heads: MLPs → waypoint deltas

**Language Adaptor** (`LanguageAdaptor`)
- Input: `embed_tokens` (text embeddings)
- Output: `lm_head` (next token logits)

**AdaptorList**
- Concatenates embeddings from all adaptors
- Applies random permutation during training
- Splits outputs back to respective adaptors

---

## 4. Loss Functions

### 4.1 Loss Components

**Waypoint Loss** (`speed_wps_loss`, `route_loss`)
- Type: Smooth L1 Loss
- Shape: `[B]` (per-sample loss)
- Formula: `F.smooth_l1_loss(pred, label).sum(-1)`

**Language Loss** (`language_loss`)
- Type: Cross-Entropy Loss
- Shape: `[B, T]` (per-token loss)
- Masking: Only tokens with `loss_masking=True` contribute
- **Note**: Always present due to conversational structure, even when answer is minimal (e.g., "Waypoints:")

### 4.2 Loss Aggregation

```python
# Average each loss over valid samples, then sum
total_loss = sum(averaged_losses)
```

---

## 5. Key Design Choices

1. **Incremental Prediction**: Predict deltas, use `cumsum()` for absolute waypoints
2. **Unified Sequence**: Concatenate all modalities into single sequence
3. **LoRA Fine-tuning**: Efficient adaptation of LLM
4. **Query-Based**: Fixed learnable queries for consistent output shape

---

## 6. Data Flow

```
Raw Data (CARLA)
    ↓
[Dataset] → Images, measurements, waypoints
    ↓
[DataModule] → Batch collation
    ↓
[DrivingInput] → Images, speed, target_point, prompt
    ↓
[Adaptors] → Language embeddings + Driving queries
    ↓
[Replace Placeholders] → Target point + Image features
    ↓
[LLM] → Hidden states
    ↓
[Split] → Driving features + Language features
    ↓
[Heads] → Waypoints (deltas) + Language (logits)
    ↓
[Loss] → Smooth L1 (waypoints) + CE (language)
```

---

## 7. Configuration Highlights

**Model**:
- Vision/Language: `OpenGVLab/InternVL2-1B`
- LoRA: `r=32, alpha=64`
- Learning rate: `3e-5`

**Data**:
- Batch size: `4`
- `pred_len`: `11` (waypoint prediction length)
- `route_as`: `target_point_command`

**Training**:
- Max epochs: `15`
- Precision: `16-mixed` (fp16/bf16)

---

## 8. Important Notes

### Placeholder Token Replacement

- **`placeholder_values`**
- **`<TARGET_POINT>`**: Replaced via `placeholder_values` dict lookup
- **`<IMG_CONTEXT>`**: Replaced via token ID matching (`input_ids == img_context_token_id`)

### Template Formats

- **`conv_dict`**: Full conversation (user + assistant with answer), used for training
- **`question_dict`**: Question only (assistant=None), used for inference

### Waypoint Generation

- **Actual waypoints**: Generated from `speed_wps_prediction` and `route_prediction` (DrivingAdaptor heads)
- **Language loss**: Targets text generation (e.g., "Waypoints:"), separate from coordinate prediction

---

*Document generated from codebase analysis of `simlingo_training`*
