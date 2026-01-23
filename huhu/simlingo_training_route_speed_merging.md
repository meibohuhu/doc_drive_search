# simlingo_training 中 Route 和 Speed Token 与 Visual Token 的合并方式

## 概述

在 `simlingo_training` 中，route 和 speed 信息通过 **placeholder token 替换机制** 嵌入到文本序列中，而不是像 `simlingo_base_training` 那样直接拼接。这种方式允许 route/speed 信息出现在文本序列的任意位置。

**重要**: `simlingo_training` 引入了 language 输入，这实际上是一种 **instructional finetuning** 方法。模型通过对话格式（user-assistant）进行训练，学习遵循自然语言指令来执行驾驶任务。

**训练流程**: `simlingo_training` 采用 **单阶段（one-stage）训练**：
- 直接使用预训练的 vision model（如 InternVL2）和 language model（如 InternLM2）
- 一次性进行 instructional finetuning，同时训练所有组件（vision、language、adaptors）
- 没有分阶段训练策略（如先训练 vision 再训练 language）
- 虽然可以加载 checkpoint 继续训练，但这是用于恢复训练，不是分阶段训练

## 详细流程

### 1. **初始输入构建** (`LanguageAdaptor.forward`)

**位置**: `simlingo_training/models/adaptors/adaptors.py:238-257`

```python
def forward(self, example: DrivingExample, inference=False, **kwargs):
    # 从 prompt 中获取文本 token IDs
    ids = label.phrase_ids.long()
    
    # 将 token IDs 转换为 embeddings
    inputs = self.embed_tokens(ids.clamp(min=0, max=self.embed_tokens.num_embeddings - 1))
    
    return {"inputs": inputs, "inputs_mask": ids_valid, "_ids": ids, "_ids_mask": ids_mask}
```

**说明**:
- 文本序列中包含特殊 token，如 `<TARGET_POINT>`、`<IMG_CONTEXT>` 等
- 这些特殊 token 最初使用标准的 embedding 表示

### 2. **Placeholder Token 替换** (`replace_placeholder_tokens`)

**位置**: `simlingo_training/models/encoder/internvl2_model.py:17-144`

这是关键步骤，分为两个子步骤：

#### 2.1 替换 Route/Speed Token (Target Point)

**位置**: `internvl2_model.py:60-91`

```python
# 1. 找到特殊 token (如 <TARGET_POINT>) 的位置
special_ids = torch.tensor(list(set(input_ids[(input_ids >= smallest_added_id)].tolist())))    ### 转为list => set => list => tensor 
mask = input_ids == special_ids
first_occurrences = torch.argmax(first_occurrence_mask.float(), dim=2)

# 2. 从 placeholder_values 中获取坐标值
coords = [torch.tensor(placeholder_values[b_id][special_ids[key_id].item()], ...) 
          for key_id, b_id in zip(special_token_pos[:, 1], special_token_pos[:, 0])]

# 3. 使用 wp_encoder (WaypointInputAdaptor) 将坐标转换为 embeddings
wp_embeds = wp_encoder(coords.unsqueeze(0)).squeeze(0)

# 4. 在 inputs_embeds 中找到特殊 token 的位置，用 wp_embeds 替换
for i, (pos, first_occurrence) in enumerate(zip(special_token_pos, first_occurrences_filtered)):
    start = first_occurrence[pos[1]]
    end = start + coords_length_org[i]
    inputs_embeds[pos[0], start:end] = wp_embeds[i]  # 替换！
```

**说明**:
- `wp_encoder` 是 `WaypointInputAdaptor`，将坐标 `(x, y)` 转换为 embedding
- 替换发生在文本序列中 `<TARGET_POINT>` token 的位置
- 如果 target_point 是多个点，会替换多个位置

#### 2.2 替换 Image Token

**位置**: `internvl2_model.py:94-132`

```python
# 1. 提取图像特征
image_features = self.model.extract_feature(pixel_values_tmp)
image_features = image_features.reshape(-1, C_embed)

# 2. 找到 <IMG_CONTEXT> token 的位置
selected = (input_ids == self.img_context_token_id)

# 3. 用 vision features 替换 <IMG_CONTEXT> token 的 embeddings
inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C_embed)
```

**说明**:
- Vision features 替换文本序列中的 `<IMG_CONTEXT>` token
- 图像特征通过 vision model 提取

#### 2.3 更新 adaptor_dict

**位置**: `internvl2_model.py:138-142`

```python
adaptor_dict['language_inputs'] = inputs_embeds  # 更新后的 language inputs
start_id = adaptor_dict['perm'][:,0]

# 将更新后的 language_inputs 复制到 adaptor_dict['inputs'] 的相应位置
for b, i in enumerate(start_id):
    adaptor_dict['inputs'][b][:len(adaptor_dict['language_inputs'][b])-i] = inputs_embeds[b][i:]
```

### 3. **AdaptorList 合并** (`AdaptorList.forward`)

**位置**: `simlingo_training/models/adaptors/adaptors.py:301-331`

```python
def forward(self, example: DrivingExample, **kwargs):
    inputs_list: List[Tensor] = []
    
    # 1. 收集各个 adaptor 的输入
    for key, adaptor in self.adaptors.items():
        adaptor_input_dict = adaptor.forward(example, **kwargs)
        inputs_list.append(adaptor_input_dict["inputs"])
    
    # 2. 拼接 language_inputs 和 driving_inputs (learnable queries)
    inputs = torch.cat(inputs_list, dim=1)  # [B, L_lang + L_driving, D]
    
    # 3. 应用随机排列（训练时）
    perm = ...
    input_dict["inputs"] = inputs[arange, perm]
```

**说明**:
- `language_inputs`: 包含文本 + target_point (已替换) + images (已替换)
- `driving_inputs`: learnable query embeddings (用于预测 waypoints)
- 两者在序列维度上拼接

### 4. **最终输入到 LLM**

**位置**: `simlingo_training/models/driving.py:forward_model`

```python
def forward_model(self, driving_input: DrivingInput, adaptor_dict: Dict, ...):
    # adaptor_dict 已经包含了替换后的 embeddings
    adaptor_embeds = adaptor_dict["inputs"]  # [B, L_total, D]
    
    # 直接输入到 LLM
    outputs = self.language_model.model(
        inputs_embeds=adaptor_embeds,
        attention_mask=adaptor_mask,
        ...
    )
```

## Instructional Finetuning 特征

`simlingo_training` 中的 language 引入确实属于 **instructional finetuning**，具体特征如下：

### 1. **对话格式** (Conversation Format)

**位置**: `simlingo_training/dataloader/dataset_driving.py:299-313`

训练数据采用标准的 user-assistant 对话格式：
```python
conversation_all = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Current speed: 10.5 m/s. Command: go left. Predict the waypoints."},
            {"type": "image"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Waypoints:"},
        ],
    }
]
```

### 2. **Chat Template**

**位置**: `simlingo_training/utils/internvl2_utils.py:94-172`

使用 `internlm2-chat` 模板将对话转换为模型输入格式：
- 添加特殊 token（如 `<|im_start|>user`, `<|im_start|>assistant`, `<|im_end|>`）
- 将图像 token 替换为 `<IMG_CONTEXT>` placeholder
- 最终格式：`<|im_start|>user\n<img>...<IMG_CONTEXT>...</img>\n{prompt}<|im_end|><|im_start|>assistant\n{answer}<|im_end|>`

### 3. **多种任务类型**

**位置**: `simlingo_training/dataloader/dataset_driving.py:251-272`

训练时混合多种任务类型：
- **驾驶任务**: `"Current speed: X m/s. Command: Y. Predict the waypoints."` → `"Waypoints:"`
- **问答任务**: `"Current speed: X m/s. Command: Y. Q: {question}"` → `"A: {answer}"`
- **评论任务**: `"Current speed: X m/s. Command: Y. What should the ego do next?"` → `"{commentary} Waypoints:"`

### 4. **Loss 计算策略**

**位置**: `simlingo_training/models/adaptors/adaptors.py:259-274` 和 `simlingo_training/utils/internvl2_utils.py:29-50`

- 只对 **assistant 回复** 计算 loss（通过 `loss_mask` 实现）
- User prompt 部分不参与 loss 计算
- 这是 instruction tuning 的标准做法，确保模型学习如何响应指令

### 5. **指令遵循能力**

**位置**: `team_code/agent_simlingo.py:543-558`

推理时支持不同类型的指令：
- `<SAFETY>`: 安全相关指令
- `<INSTRUCTION_FOLLOWING>`: 指令遵循模式
- `use_cot`: Chain-of-Thought 推理（"What should the ego do next?"）

## 与 simlingo_base_training 的对比

| 特性 | simlingo_base_training | simlingo_training |
|------|------------------------|-------------------|
| **训练范式** | 端到端回归训练 | **Instructional Finetuning** |
| **训练阶段** | 单阶段训练 | **单阶段训练**（one-stage） |
| **预训练模型** | 使用预训练的 vision 和 language 模型 | 使用预训练的 vision（InternVL2）和 language（InternLM2）模型 |
| **Route/Speed 位置** | 固定位置（在 vision 之后） | 文本序列中的任意位置（通过 placeholder） |
| **合并方式** | 直接 `torch.cat([vision, speed, route])` | Placeholder token 替换 |
| **Speed 编码** | `VectorInputAdaptor` (1个 token) | 通过 `wp_encoder` 编码坐标 |
| **Route 编码** | `RouteEncode` (ResNet18) 或 `WaypointInputAdaptor` | `WaypointInputAdaptor` (坐标序列) |
| **文本输入** | 无 | 有（通过 `LanguageAdaptor`） |
| **对话格式** | 无 | User-Assistant 对话格式 |
| **Loss 计算** | 对所有输出计算 loss | 只对 assistant 回复计算 loss |
| **任务多样性** | 单一驾驶任务 | 驾驶 + 问答 + 评论多任务 |

## 关键代码位置总结

1. **Route/Speed 编码**: `simlingo_training/models/adaptors/adaptors.py:64-93` (`WaypointInputAdaptor`)
2. **Placeholder 替换**: `simlingo_training/models/encoder/internvl2_model.py:17-144` (`replace_placeholder_tokens`)
3. **Adaptor 合并**: `simlingo_training/models/adaptors/adaptors.py:301-331` (`AdaptorList.forward`)
4. **最终输入**: `simlingo_training/models/driving.py:192-257` (`forward_model`)

## 示例：输入序列结构

假设输入文本为：
```
"Current speed: 10.5 m/s. Target waypoint: <TARGET_POINT>. <IMG_CONTEXT> What should the ego do next?"
```

经过处理后，序列变为：
```
[text_embeds..., speed_embeds..., target_point_embeds..., image_embeds..., text_embeds..., driving_queries...]
     ↑              ↑                    ↑                    ↑              ↑              ↑
  文本部分      speed token         target_point      image features    文本部分      learnable queries
              (已替换)              (已替换)          (已替换)
```

## 优势

1. **灵活性**: Route/speed 可以出现在文本序列的任意位置
2. **上下文感知**: 模型可以看到 route/speed 在文本中的上下文
3. **统一处理**: 所有输入（文本、图像、坐标）都通过统一的 embedding 空间处理
4. **指令遵循能力**: 通过 instructional finetuning，模型能够理解和遵循自然语言指令
5. **多任务学习**: 同时学习驾驶、问答、评论等多种任务，提升模型的泛化能力
6. **更好的可解释性**: 模型通过生成文本响应来解释其行为（如 "What should the ego do next?"）


