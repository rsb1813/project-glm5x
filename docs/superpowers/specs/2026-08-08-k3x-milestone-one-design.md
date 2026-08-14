# K3X 첫 마일스톤 설계 명세

## 1. 목적

K3X의 첫 마일스톤은 Kimi K3 전체 weight 없이도 모델 그래프와 상태 전이, K3X 저장 형식 및 독립 runtime의 정확성을 검증하는 실행 가능한 축소 시스템을 만든다. 성능 최적화보다 oracle과의 재현성을 우선하며, 이후 모든 최적화가 우회할 수 있는 reference mode의 기반이 된다.

## 2. 승인된 범위

### 포함

- PyTorch FP32 reference text decoder.
- KDA와 causal depthwise ShortConv state.
- Gated MLA와 incremental KV state.
- Block Attention Residual.
- sigmoid router, correction bias, normalized Top-K weight.
- Stable LatentMoE의 down projection, routed experts, latent norm, up projection.
- dense 및 shared SiTU-GLU MLP.
- native MXFP4 expert payload decode와 byte-exact repack.
- full-prefix와 incremental greedy generation.
- K3X v1 streaming converter, reader, validator 및 resume.
- 독립 C++20 CPU runtime.
- 합성 모델의 실제 benchmark와 resource counters.

### 제외

- MoonViT-V2 실행. 문서에는 multimodal projector와 text embedding 연결 경계만 기록한다.
- 전체 1.56 TB checkpoint 또는 weight shard 다운로드.
- CUDA, io_uring, O_DIRECT 및 3-tier cache 구현.
- mixed-precision calibration, adaptive Top-K, pruning 및 proxy.
- speculative decoding과 expert-major verification.
- Cloud Run Job 실행 또는 유료 자원 생성.

## 3. 근거가 되는 source snapshot

다음 commit 또는 문서는 2026-08-08에 확인한 설계 근거이다.

| 출처 | 고정 revision | 이번 단계에서 확인한 내용 |
|---|---|---|
| MoonshotAI/Kimi-K3 | `3cb39dfd32e51c3328e2e4b4af21341247d06c43` | 공식 model summary와 technical report |
| moonshotai/Kimi-K3 config | Hugging Face `main`, 확인일 2026-08-08 | 실제 차원, layer map, quantization metadata |
| vllm-project/vllm | `44351f81d58861edc873c7678c500a4f40834450` | NVIDIA K3 graph, AttnRes, KDA 및 latent MoE serving 경로 |
| FareedKhan-dev/kimi-k3-in-c | `ff11dce858a2eb8a781224facdffd33a1fa48d25` | 독립 CPU graph, incremental state 및 oracle fixture 접근 |
| PipeNetwork/kimi-k3-mlx | `20a4fb101ce81380ab8af0036743d49e7256c521` | 공개 PyTorch reference graph과 streaming converter 비교 |
| MoonshotAI/FlashKDA | `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` | KDA kernel의 실제 layout과 recurrence 비교 |
| MoonshotAI/Attention-Residuals | `85e22310fe5ee860b4a023de312d791de8a5a5e6` | Attention Residual 원 논문과 구현 |
| deepseek-ai/DeepSpec | `005e03b81cec38b7da6399833d609ee89a2587f2` | DSpark 공개 구현과 학습·검증 interface |
| SpecMD | arXiv `2602.03921` | Least-Stale 정책의 원문. 첫 마일스톤에서는 구현하지 않음 |
| EcoSpec | arXiv `2607.12696` | marginal expert cost 기반 draft 선택. 첫 마일스톤에서는 구현하지 않음 |
| MoE-Spec | arXiv `2602.16052` | verification layer별 expert budget. 첫 마일스톤에서는 구현하지 않음 |
| AcceptMoE | arXiv `2608.02989` | commitment-weighted verifier expert set. 첫 마일스톤에서는 구현하지 않음 |

논문에 공개 code artifact가 연결되지 않은 기법은 논문만 확인한 상태로 표시한다. 후속 구현 단계에서 공개 artifact의 존재와 revision을 다시 확인하기 전에는 재현 완료로 간주하지 않는다.

## 4. 실제 Kimi K3 text graph 계약

공식 config는 hidden size 7168, 93 decoder layers, 96 heads, vocabulary 163840을 선언한다. 0-based runtime layer 기준으로 `0-2, 4-6, ..., 88-90`은 KDA이고 `3, 7, ..., 87, 91, 92`는 Gated MLA이다. 첫 layer만 dense MLP이며 layer 1-92는 routed MoE이다.

각 decoder layer는 ordinary PreNorm residual 대신 Block Attention Residual 경로를 사용한다.

1. 현재 block prefix sum과 이전 block residual sources를 결합한다.
2. 각 source를 RMS normalize하고 learned scalar projection으로 depth score를 얻는다.
3. depth 방향 softmax로 raw source를 가중합한다.
4. 그 결과를 input RMSNorm과 attention에 공급한다.
5. attention output을 현재 prefix sum에 더한다.
6. 다시 Attention Residual을 적용하고 post-attention RMSNorm과 MLP/MoE를 실행한다.
7. MLP/MoE output을 prefix sum에 더한다.
8. 매 12 layers 경계의 prefix representation을 block residual source로 보존한다.
9. 모든 layers 뒤 output Attention Residual과 final RMSNorm을 적용한다.

### 4.1 KDA

KDA layer는 hidden state에서 Q, K, V를 독립 projection하고 각 projection에 kernel size 4의 causal depthwise ShortConv와 SiLU를 적용한다. Q와 K는 head별 L2 normalization을 사용한다. Forget signal은 full-rank gate이며 lower bound `-5.0`을 갖는다. Recurrent state는 head별 key-channel × value-channel matrix이다.

한 token의 개념적 recurrence는 다음 순서를 보존한다.

1. channel-wise decay로 state row를 감쇠한다.
2. 감쇠된 state에서 `u = S^T k`를 읽는다.
3. delta rule `S += k ⊗ beta(v-u)`로 갱신한다.
4. 갱신된 state에서 `o = S^T q`를 읽는다.
5. output gate가 결합된 RMSNorm을 적용한 뒤 output projection한다.

Incremental state는 Q/K/V convolution histories와 recurrent matrix를 모두 포함한다.

### 4.2 Gated MLA

MLA의 query는 hidden 7168에서 rank 1536 projection, RMSNorm, head별 query projection을 거친다. 각 head query는 주 key subspace 128과 추가 key subspace 64로 나뉜다. Checkpoint 호환 field 이름은 각각 `qk_nope_head_dim`과 `qk_rope_head_dim`이지만 K3는 `mla_use_nope=true`이므로 어느 subspace에도 rotary embedding을 적용하지 않는다. KV projection은 rank 512 latent와 shared extra key 64를 함께 생성하고 latent에만 RMSNorm을 적용한다. Latent는 head별 주 key 128과 value 128로 확장된다.

Attention score는 두 NoPE key subspace를 모두 포함하며 전체 query head dimension 192의 제곱근으로 scale한다. Value output에는 hidden state에서 생성한 sigmoid output gate를 적용한 뒤 output projection한다. 첫 마일스톤 incremental cache는 correctness를 단순화하기 위해 확장된 per-head main K/V와 shared extra-key slot을 저장한다.

### 4.3 Router와 Stable LatentMoE

Router logits는 FP32 hidden state와 router matrix의 linear projection이다. score는 expert별 sigmoid이다. correction bias는 expert 선택 순위에만 더하고 최종 expert weight에는 bias 없는 score를 사용한다. 선택된 Top-16 weight는 합이 1이 되도록 normalize한 뒤 routed scaling factor를 곱한다.

공개 checkpoint config의 SiTU 상수는 `activation_situ_beta=4.0`, `activation_situ_linear_beta=25.0`이며 routed scaling factor는 1.0이다. 합성 모델도 같은 activation 상수를 유지한다.

Routed branch는 hidden 7168을 latent 3584로 down-project하고, 선택된 각 expert의 SiTU-GLU를 실행해 router weight로 합산한다. 합산된 latent를 RMSNorm하고 hidden 7168로 up-project한다. Shared branch는 원래 hidden state에서 2 shared experts 상당의 dense SiTU-GLU를 실행한다. 최종 MoE output은 routed branch와 shared branch의 합이다.

### 4.4 MXFP4

공식 checkpoint는 compressed-tensors의 `mxfp4-pack-quantized` 형식을 사용한다. 첫 마일스톤은 합성 routed expert의 packed data와 E8M0 scale을 K3X extent로 byte-exact 복사한다. Runtime은 별도 FP32 oracle과 비교 가능한 decode/matmul 경로를 제공한다. 잘못된 scale encoding, shape 또는 group alignment는 명시적으로 거부한다.

## 5. 합성 모델

합성 모델은 실제 위상과 packing 조건을 유지하면서 다음 크기로 축소한다.

| 항목 | 합성 값 |
|---|---:|
| vocabulary | 64 |
| hidden size | 64 |
| decoder layers | 4 |
| layer pattern | KDA, KDA, KDA, MLA |
| dense layers | layer 0 only |
| KDA heads × head dim | 4 × 16 |
| ShortConv kernel | 4 |
| MLA heads | 4 |
| q LoRA rank | 32 |
| KV latent rank | 32 |
| qk main / extra NoPE | 8 / 8 |
| value head dim | 8 |
| routed experts / Top-K | 8 / 2 |
| shared experts | 1 |
| routed latent size | 32 |
| expert intermediate size | 32 |
| dense intermediate size | 96 |
| SiTU beta / linear beta | 4 / 25 |
| Attention Residual block | 2 |

Attention Residual block size 2는 실제 값 12의 축소 대응이다. 4-layer fixture 안에서 복수 block source와 output mixing을 모두 실행하기 위한 유일한 topology 축소이며, 문서와 fixture metadata에 실제 값과 합성 값을 함께 기록한다.

## 6. 구성 요소와 책임

### 6.1 `reference/k3x_ref`

- 명시적 PyTorch eager 연산만으로 golden 결과를 생성한다.
- optimized path와 공유하지 않는 hand-derived fixtures를 제공한다.
- 모든 random tensor는 고정 seed와 canonical tensor naming을 사용한다.
- layer별 hidden state, router 결과, recurrent/conv/KV state, logits와 token IDs를 저장한다.

### 6.2 `converter/k3x_converter`

- source index에서 tensor 하나 또는 bounded chunk만 읽는다.
- canonical tensor role을 K3X IDs와 layer/expert directory entries로 변환한다.
- native MXFP4 payload는 byte repack만 한다.
- extent 기록 후 flush, checksum 검증, resume manifest 갱신 순서를 지킨다.
- 전체 source model을 RAM에 올리는 API를 제공하지 않는다.

### 6.3 `runtime`

- C++20과 표준 라이브러리만으로 K3X metadata와 payload를 읽는다.
- 첫 마일스톤 backend는 deterministic FP32 CPU reference이다.
- reader, tensor views, state, graph execution을 분리해 이후 CUDA backend가 graph 의미를 바꾸지 않고 교체될 수 있게 한다.
- unsupported required feature나 손상 artifact를 fail closed로 처리한다.

### 6.4 `tests`

- Python unit tests는 각 reference 연산과 converter를 검증한다.
- C++ unit tests는 reader와 kernel 경계를 검증한다.
- integration test는 Python이 생성한 artifact와 golden을 C++ executable에 전달해 결과를 비교한다.
- end-to-end test는 full-prefix와 incremental greedy tokens가 서로 및 golden과 exact match하는지 검사한다.

## 7. K3X v1 논리 형식

모든 integer는 little-endian fixed width이다. 파일 시작의 4096-byte superblock 뒤에 directory extents와 payload extents가 놓인다. 모든 extent offset은 superblock의 alignment 값에 맞춘다.

Superblock은 magic, major/minor version, file UUID, alignment, required/optional feature masks, directory offsets와 lengths, source fingerprint, finalized flag 및 root digest를 가진다. Tensor directory는 stable tensor ID, semantic role, dtype, rank, dimensions, quantization ID와 extent reference를 가진다. Layer directory는 실행 index, attention/FFN 종류 및 관련 tensor/expert ranges를 가진다. Expert directory는 layer/expert ID와 packed data/scale extent references를 가진다.

각 extent는 type, flags, logical length, stored length 및 CRC32C를 가진다. 완성 artifact에는 directories와 payload를 포함한 SHA-256 root digest를 기록한다. Unknown optional feature는 건너뛸 수 있지만 unknown required feature 또는 다른 major version은 거부한다.

## 8. Crash-safe streaming 변환

1. 최종 경로와 다른 `.partial` artifact와 resume manifest를 생성한다.
2. source identity, converter version, config digest와 planned extents를 기록한다.
3. source tensor 또는 bounded chunk를 읽고 변환한다.
4. aligned extent를 쓰고 flush한 뒤 read-back CRC32C를 검사한다.
5. 완료 extent의 source fingerprint와 output checksum을 manifest에 원자적으로 기록한다.
6. 재시작 시 두 fingerprint가 일치하는 extent만 재사용한다.
7. 모든 extents 뒤 directories와 provisional superblock을 기록한다.
8. 전체 SHA-256과 directory invariants를 검사한다.
9. finalized superblock을 기록하고 flush한다.
10. `.partial`을 최종 artifact 이름으로 원자적으로 rename한다.

## 9. 오류 정책

다음 조건에서는 값을 추측하거나 부분 실행하지 않는다.

- magic/version/required feature 불일치.
- directory 또는 extent가 파일 범위를 벗어남.
- extent overlap, alignment 위반 또는 duplicate semantic key.
- tensor shape, dtype 또는 quantization metadata 불일치.
- CRC32C나 root SHA-256 불일치.
- resume source/config/converter fingerprint 불일치.
- KDA/MLA state shape 또는 cache position 위반.
- MXFP4 group과 payload/scale length 불일치.

오류는 stable machine-readable code와 사람이 읽을 수 있는 context를 함께 반환한다.

## 10. Correctness 전략

모든 production behavior는 실패하는 테스트를 먼저 작성한다. 기대값은 implementation helper로 계산하지 않고 PyTorch reference 또는 작은 hand-derived literal에서 얻는다.

검증 단계는 primitive, stateful operator, decoder layer, full graph, storage round-trip 순서이다. FP32 수치 결과와 recurrent/KV state는 연산별 명시 tolerance를 사용한다. GEMM batch shape이나 독립 C++ reduction order가 다른 경로에 bitwise state digest 일치를 요구하지 않는다. MXFP4 payload와 scale round-trip, selected expert IDs 및 greedy tokens는 exact match여야 한다. Full-prefix와 incremental execution은 final token뿐 아니라 KDA convolution/recurrent state와 MLA KV contents도 수치 비교한다.

## 11. 측정 계획

첫 마일스톤 benchmark는 합성 모델에 대해서만 다음을 측정한다.

- converter input/output bytes와 peak RSS.
- cold/warm artifact open 시간.
- token당 positional read bytes.
- prefill 및 incremental decode latency.
- primitive와 layer별 CPU 시간.
- KDA/MLA state bytes.

`PERFORMANCE_MODEL.md`는 실제 K3 config로 trunk, selected expert, KV/KDA state 및 tier traffic의 식을 계산한다. Cache hit rate와 prefetch recall은 변수로 두고 가정 시나리오와 실제 측정을 분리한다. 측정 전 tok/s 목표 달성을 약속하지 않는다.

## 12. 완료 기준

- 공식 graph와 제안 memory/data flow 문서가 source revision을 포함한다.
- 합성 graph의 모든 필수 연산이 PyTorch와 C++에서 실행된다.
- K3X streaming round-trip, resume와 손상 거부 테스트가 통과한다.
- MXFP4 expert payload와 scales가 byte-exact 보존된다.
- layer outputs와 states가 tolerance 안에서 일치한다.
- full/incremental greedy token sequence가 exact match한다.
- Python/C++ 전체 test suite가 통과한다.
- 실제 합성 benchmark와 다음 병목이 기록된다.
- 전체 checkpoint와 유료 cloud resource를 사용하지 않는다.

