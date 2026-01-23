# SimLingo Base Training: Architecture and Training Analysis

## Overview

SimLingo Base Training is a simplified end-to-end autonomous driving model that predicts future waypoints from visual inputs, vehicle state, and route information. Unlike `simlingo_training`, this version does **not** use language commands or generate language outputs - it focuses purely on trajectory prediction.

---

## 1. Input
The model receives a `DrivingInput` tuple containing:

#### **Camera Images** (`camera_images`)
- **Shape**: `[B, T, N, C, H, W]` where:
  - `B`: Batch size
  - `T`: Temporal frames (typically 1)
  - `N`: Number of camera views (typically 3: front-forward, front-left, front-right)
  - `C`: Channels (3 for RGB)
  - `H, W`: Image height and width (original: 2048×1280)
- **Type**: `uint8` [0, 255]
- **Preprocessing**: 
  - Images are processed by LLaVA-NeXT encoder
  - May apply image shift augmentation
  - Bottom quarter may be cropped (configurable)

#### **Camera Intrinsics** (`camera_intrinsics`)
- **Shape**: `[B, N, 3, 3]`
- **Type**: `float32`
- **Purpose**: Camera calibration matrices for each view

#### **Camera Extrinsics** (`camera_extrinsics`)
- **Shape**: `[B, N, 4, 4]`
- **Type**: `float32`
- **Purpose**: Camera pose transformation matrices

#### **Vehicle Speed** (`vehicle_speed`)
- **Shape**: `[B, S]` where S is speed history length (typically 1)
- **Type**: `float32`
- **Unit**: meters per second (m/s)
- **Purpose**: Current vehicle speed
- **Encoding**: Encoded using `VectorInputAdaptor` with normalization to [0, 1] range
  - Normalization range: `(0.0, 64.0/3.6)` m/s (or `(0.0, 110.0/3.6)` with new normalization)

#### **Map Route** (`map_route`)
- **Shape**: `[B, 3, RH, RW]` (when `route_as != 'target_point'`)
- **Type**: `uint8` [0, 255]
- **Purpose**: Bird's-eye view (BEV) route image showing the navigation path
- **Alternative**: When `route_as == 'target_point'`, this is replaced by target point coordinates

#### **Target Point** (`target_point`)
- **Shape**: `[B, 2]`
- **Type**: `float32`
- **Purpose**: Navigation target point in ego coordinate system (x, y)
- **Usage**: 
  - When `route_as == 'target_point'` or `route_as == 'coords'`: Encoded directly as waypoint coordinates
  - Otherwise: Used to generate route image


---

## 2. Output

### 2.1 Output Components

The model produces:

#### **Speed Waypoints** (`speed_wps`)
- **Shape**: `[B, F, 2]` where F is future waypoint count (typically 10)
- **Type**: `float32`
- **Format**: 2D coordinates `(x, y)` in ego coordinate system
- **Mode**: 
  - `2d`: Full 2D waypoints (default)
  - `1d`: Distance-based waypoints (cumulative distance along path)
- **Prediction Method**: 
  - Model predicts **deltas** (incremental changes)
  - Deltas are cumulated using `cumsum` to get absolute waypoints
  - Formula: `waypoint[i] = cumsum(deltas[:i+1])`

#### **Route** (`route`)
- **Shape**: `[B, 20, 2]` (when `predict_route_as_wps=True`)
- **Type**: `float32`
- **Purpose**: High-level navigation path (20 waypoints)
- **Usage**: Represents the intended route for navigation
- **Note**: Only predicted when `predict_route_as_wps=True`

### 2.2 Output Processing

1. **Waypoint Decoding**:
   ```python
   # In DrivingAdaptor.get_predictions()
   feature = features[:, current_index: current_index + size]
   prediction = self.heads[input_type](feature).cumsum(1)  # Cumsum to get absolute waypoints
   ```

2. **Route Post-processing**:
   - Routes may be interpolated to equal spacing for evaluation

---

## 3. Model Architecture

### 3.1 High-Level Architecture

```
Input (Images + Speed + Route)
    ↓
[Vision Encoder] (LLaVA-NeXT)
    ↓
[Vision Features] → Project to hidden_size
    ↓
[Speed Encoder] → Speed embedding
[Route Encoder] → Route embedding (ResNet18 or MLP)
    ↓
[Concatenate] → Unified sequence
    ↓
[Language Model] (Tiny Llama, 50M)
    ↓
[Hidden States] → Split adaptor outputs
    ↓
[Driving Head] → Waypoint deltas
    ↓
[Cumsum] → Absolute waypoints
    ↓
Output (Waypoints + Route)
```

### 3.2 Component Details

#### **Vision Model** (`LLaVAnextEncoderModel`)
- **Base**: `llava-hf/llava-v1.6-mistral-7b-hf`
- **Architecture**: LLaVA-NeXT vision encoder (vision component only)
- **Key Features**:
  - Uses pretrained LLaVA vision encoder
  - Extracts patch embeddings from images
  - Applies temporal and camera encodings
  - Downsampling factor: `downsample_feature_grid_factor=2` (default)
- **Output**: Image embeddings `[B, N_tokens, embed_dim]`
- **Projection**: Linear layer to match language model hidden size

#### **Language Model** (`Llama`)
- **Variant**: `tiny` (50M parameters)
- **Architecture**: Transformer-based causal language model (Llama-style)
- **Configuration**:
  - Layers: 12
  - Attention heads: 8
  - Hidden size: 512
  - Intermediate size: 2048
- **Key Features**:
  - **No LoRA**: `lora=False` (default) - full fine-tuning
  - Tokenizer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
  - **No vocabulary**: `embed_tokens = None` - uses input embeddings directly
- **Forward Pass**:
  - Processes input embeddings (vision + speed + route)
  - Returns hidden states from last layer

#### **Route Encoder**

**Option 1: ResNet18** (`RouteEncode`)
- Used when `route_as != 'target_point'` and `route_as != 'coords'`
- **Architecture**: ResNet18 pretrained on ImageNet
- **Input**: Route image `[B, 3, H, W]`
- **Preprocessing**: Normalized to `[-1, 1]` range (`x / 128.0 - 1.0`)
- **Output**: Route embedding `[B, 1, hidden_size]`

**Option 2: WaypointInputAdaptor**
- Used when `route_as == 'target_point'` or `route_as == 'coords'`
- **Architecture**: MLP
  ```
  Input [B, N, 2] → Linear(2→256) → ReLU → Linear(256→hidden_size)
  ```
- **Normalization**: `NormZeroOne` with range `(-32.0, 32.0)` or `(-200.0, 200.0)`
- **Input**: Target point coordinates `[B, 2]` → expanded to `[B, 1, 2]`

#### **Speed Encoder** (`VectorInputAdaptor`)
- **Purpose**: Encodes vehicle speed scalar to embedding
- **Architecture**: MLP
  ```
  Input [B, 1] → NormZeroOne → Linear(1→256) → ReLU → Linear(256→hidden_size) → Unsqueeze
  ```
- **Normalization**: `NormZeroOne` with range `(0.0, 64.0/3.6)` m/s
- **Output**: `[B, 1, hidden_size]`

#### **Driving Adaptor** (`DrivingAdaptor`)
- **Purpose**: Creates learnable query embeddings that are processed by the LLM, then decodes waypoint predictions from LLM outputs
- **Components**:
  - **Query Embeddings**: Learnable parameters for waypoint queries (initialized randomly)
    - `query_embeds_speed`: `[1, 10, hidden_size]` for speed waypoints (10 queries)
    - `query_embeds_wps`: `[1, 20, hidden_size]` for route (if enabled, 20 queries)
  - **Prediction Heads**: MLPs to decode LLM output features to waypoints
    - `speed_wps_head`: `Linear(hidden_size → mlp_dim) → SiLU → Linear(mlp_dim → 2)`
    - `route_head`: `Linear(hidden_size → mlp_dim*2) → SiLU → Linear(mlp_dim*2 → mlp_dim) → SiLU → Linear(mlp_dim → 2)`
- **Forward Pass**:
  1. **Input Phase**: Creates learnable query embeddings for each prediction type
  2. **Concatenation**: Query embeddings are concatenated with vision/speed/route embeddings: `[vision, speed, route, queries]`
  3. **LLM Processing**: All embeddings (including queries) are fed into the language model together
  4. **Output Extraction**: After LLM forward pass, outputs are split to extract adaptor-specific features
  5. **Decoding**: Prediction heads decode LLM output features to waypoint deltas
  6. **Cumulation**: Deltas are cumulated using `cumsum()` to get absolute waypoints

#### **Language Projection**
- **Purpose**: Projects vision embeddings to language model hidden size
- **Architecture**: 
  - Identity if `vision_model.token_size == language_model.hidden_size`
  - Linear layer otherwise: `Linear(vision_token_size → hidden_size, bias=False)`

### 3.3 Forward Pass Flow

1. **Input Preparation**:
   ```python
   vision_embeds, _ = self.get_fixed_input_embeds(driving_input)
   # Returns: [vision_embeds, speed_embeds, route_embeds] concatenated
   ```

2. **Adaptor Query Creation**:
   ```python
   adaptor_dict = self.adaptors(example)
   adaptor_embeds = adaptor_dict["inputs"]  # Learnable query embeddings [B, N_queries, hidden_size]
   ```
   - Creates learnable query embeddings (e.g., 10 queries for speed_wps, 20 for route)

3. **Input Concatenation & LLM Forward**:
   ```python
   input_embeds = torch.cat((vision_embeds, adaptor_embeds), dim=1)  # [vision, speed, route, queries]
   outputs = self.language_model.forward(input_embeds)  # LLM processes all embeddings together
   ```
   - Query embeddings are concatenated **after** vision/speed/route embeddings
   - LLM processes the entire sequence, allowing queries to attend to vision/route context

   input_embeds = [vision_embeds, speed_embeds, route_embeds, query_embeds] 
   Query embeddings 是可学习参数（随机初始化，训练中更新，它们与条件信息（vision/speed/route）一起输入 LLM
   LLM 的输出中，对应 query 位置的部分被提取出来

4. **Output Splitting**:
   ```python
   vision_outputs, adaptor_outputs = outputs.split(
       [vision_size, adaptor_size], dim=1
   )
   ```
   - Extract adaptor-specific outputs (corresponding to query positions)

5. **Prediction Decoding**:
   ```python
   predictions = self.adaptors.driving.get_predictions(adaptor_outputs)
   # Returns: {'speed_wps': [B, 10, 2], 'route': [B, 20, 2]}
   ```
   - Prediction heads decode LLM output features to waypoint deltas
   - Deltas are cumulated to get absolute waypoints

---

## 4. Loss Functions

### 4.1 Loss Components

The model uses **MSE Loss** for waypoint prediction:

#### **Waypoint Loss** (`speed_wps_loss` / `route_loss`)
- **Type**: Mean Squared Error (MSE) Loss
- **Formula**:
  ```python
  loss = F.mse_loss(prediction, label, reduction="none").sum(-1).mean(-1)
  ```
- **Shape**: `[B]` (one loss value per batch item)
- **Computation**:
  1. Compute MSE Loss between predicted and ground truth waypoints (no reduction)
  2. Sum over spatial dimensions (last dimension: x, y)
  3. Mean over waypoint sequence dimension
  4. Result: per-sample loss
- **Ground Truth**:
  - `speed_wps`: `label.waypoints[:, :future_speed_waypoints]` (first N waypoints)
  - `route`: `label.route_adjusted` (route waypoints, when enabled)

### 4.2 Loss Aggregation

Losses are aggregated using `summarise_losses()`:

```python
def summarise_losses(loss_dict, weights=None):
    # Average each loss over valid samples
    loss_averages = {
        k: torch.where(n.sum() > 0, v.sum() / n.sum(), 0.0) 
        for k, (v, n) in loss_dict.items()
    }
    # Sum all losses (optionally weighted)
    total_loss = torch.stack(list(loss_averages.values())).sum()
    return TrainingOutput(loss=total_loss, ...)
```

- **Per-Loss Averaging**: Each loss is averaged over valid samples
- **Total Loss**: Sum of all averaged losses
- **Weighting**: Optional per-loss weights (not used by default)

### 4.3 Training Configuration

- **Optimizer**: AdamW (or FusedAdam with DeepSpeed)
  - Learning rate: `3e-5` (default)
  - Vision learning rate: `3e-5` (separate LR for vision model)
  - Weight decay: `0.1`
  - Betas: `(0.9, 0.999)`
- **Scheduler**: OneCycleLR
  - Max LR: Same as initial LR
  - Percentage warmup: `0.05` (5%)
- **Parameter Groups**:
  - Vision model: Uses `vision_lr`
  - Other components: Use `lr`

---

## 5. Key Design Choices

### 5.1 No Language Input/Output
- **Why**: Simplified architecture focusing on trajectory prediction
- **Benefit**: Faster training, simpler model, no language generation overhead
- **Difference from simlingo_training**: No language prompts or text generation

### 5.2 Incremental Waypoint Prediction
- **Why**: Predicts deltas instead of absolute coordinates
- **Benefit**: Easier to learn, more stable gradients
- **Implementation**: `cumsum()` to convert deltas to absolute waypoints

### 5.3 Route Encoding Options
- **ResNet18**: When route is provided as BEV image
- **MLP**: When route is provided as target point coordinates
- **Benefit**: Flexible input representation

### 5.4 Separate Learning Rates
- **Why**: Vision model may need different learning rate
- **Benefit**: Better fine-tuning control
- **Implementation**: Separate parameter groups with `vision_lr`

### 5.5 Query-Based Prediction
- **Why**: Fixed number of learnable queries for waypoints
- **Benefit**: Consistent output shape, easier batching
- **Implementation**: Learnable query embeddings

---

## 6. Data Flow Summary

```
Raw Data (CARLA)
    ↓
[Dataset] → Load images, measurements, waypoints
    ↓
[DataModule] → Batch and collate
    ↓
[DrivingInput] → Images, speed, map_route, target_point
    ↓
[Vision Encoder] → Image features
[Speed Encoder] → Speed embedding
[Route Encoder] → Route embedding
    ↓
[Concatenate] → Unified sequence (vision + speed + route)
[Driving Adaptor] → Create learnable query embeddings
    ↓
[Concatenate] → [vision, speed, route, queries]
    ↓
[Language Model] → Process all embeddings, output hidden states
    ↓
[Split Outputs] → Adaptor features
    ↓
[Driving Head] → Waypoint deltas → Cumsum → Waypoints
    ↓
[Compute Loss] → MSE Loss
    ↓
[Backprop] → Update parameters
```

---

## 7. Configuration Highlights

### Key Hyperparameters (from `local_training.yaml`)

- **Model**:
  - Vision: `llava-hf/llava-v1.6-mistral-7b-hf`
  - Language: `tiny` (50M parameters, no LoRA)
  - Learning rate: `3e-5`
  - Vision learning rate: `3e-5`
  - `predict_route_as_wps`: `True`
  - `speed_wps_mode`: `2d`
  - `route_as`: `target_point`

- **Data**:
  - Batch size: `4`
  - `hist_len`: `1` (temporal history)
  - `cut_bottom_quarter`: `False`
  - `use_global_img`: `False`

- **Training**:
  - Max epochs: `30`
  - Validation every: `10` epochs
  - Precision: `bf16` or `fp16` (depending on strategy)

---

## 8. Differences from SimLingo Training

| Aspect | SimLingo Base Training | SimLingo Training |
|--------|------------------------|-------------------|
| **Language Input** | ❌ No | ✅ Yes (prompts, commands) |
| **Language Output** | ❌ No | ✅ Yes (text generation) |
| **Vision Model** | LLaVA-NeXT | InternVL2-1B |
| **Language Model** | Tiny Llama (50M) | InternVL2-1B (1B) |
| **LoRA** | ❌ No (full fine-tuning) | ✅ Yes |
| **Route Encoding** | ResNet18 or MLP | WaypointInputAdaptor |
| **Loss Function** | MSE | Smooth L1 |
| **Speed Input** | ✅ Optional | ❌ Not used |
| **Architecture** | Simpler | More complex (multi-modal) |


## References

- **LLaVA-NeXT**: Vision-Language Model
- **Tiny Llama**: Small language model (50M parameters)
- **PyTorch Lightning**: Training framework
- **Hydra**: Configuration management

---

*Document generated from codebase analysis of `simlingo_base_training`*

