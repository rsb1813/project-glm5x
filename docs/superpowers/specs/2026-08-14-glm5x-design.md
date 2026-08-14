# GLM5X 설계

## 목표

GLM-5.2를 현재 기준 모델로 실행하고, GLM-5.3 가중치가 공개되면 모델 descriptor와 checkpoint manifest 교체만으로 동일한 out-of-core runtime을 재사용합니다.

## 경계

GLM5X는 기존 K3X의 검증된 공용 기술을 보존합니다.

- K3X storage core: aligned extents, directory, checksums, resumable streaming
- three-tier cache: VRAM, system RAM, NVMe
- deadline-aware prefetch와 task/session profile
- exact routing, cold rescue, expert-major verification interface
- benchmark schema와 ablation runner

Kimi K3 전용 graph는 GLM5X 기본 경로에 포함하지 않습니다.

- KDA recurrent graph
- Attention Residual graph
- 896 expert Top-16 assumptions
- native Kimi MXFP4 tensor naming

GLM adapter는 DSA/MLA, 256 routed experts, Top-8 routing, shared expert, MTP head를 descriptor로 노출합니다. 실제 tensor shape와 파일명은 checkpoint manifest에서 읽고 코드에 하드코딩하지 않습니다.

## 데이터 흐름

원본 GLM shard → bounded converter worker → GLM5X extent writer → NVMe artifact → RAM expert cache → VRAM resident/pinned staging → GLM execution.

MTP draft는 후보 token을 만들고, target verifier는 후보 token의 exact routing을 계산합니다. expert-major 모드는 candidate token 전체의 unique expert union을 만들고 weight fetch를 한 번만 수행합니다. acceptance와 quality는 natural routing strict mode를 기준으로 비교합니다.

## 교체 계약

GLM-5.3 전환 시 다음은 모델별로 재생성합니다.

- `config.json` 기반 descriptor
- tokenizer와 chat template
- tensor manifest와 SHA-256
- quantization calibration
- expert frequency/transition/PGO profile
- MTP acceptance profile

다음은 동일 interface를 유지하는 것을 목표로 합니다.

- storage superblock와 extent reader
- cache policy interface
- prefetch scheduler
- speculative verification interface
- benchmark JSON/CSV schema

## 정확성 계약

reference mode는 항상 유지합니다. 최적화 경로는 greedy token, layer output, routing IDs, MTP accept/commit 결과를 reference와 비교하며, proxy/pruning/AcceptMoE는 correctness mode에서 비활성화합니다.

## 상태 라벨

- Implemented: 기존 K3X에서 이식되어 테스트 가능한 공용 코드
- In progress: GLM descriptor·converter·reference graph
- Experimental: MTP scheduling, expert-major batching, mixed quantization
- Proposed: GLM-5.3 checkpoint adapter와 full CUDA fast path

