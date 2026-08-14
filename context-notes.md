# GLM5X 작업 컨텍스트

## 2026-08-14

- 프로젝트 목표를 Kimi K3 전용 K3X에서 GLM-5.2 우선, GLM-5.3 교체 가능 구조인 GLM5X로 전환했습니다.
- 기존 K3X의 저장 포맷, tiered cache, deadline prefetch, expert-major scheduling, speculative interface, benchmark ledger는 재사용 대상으로 분류했습니다.
- KDA, Attention Residual, 896-way Top-16, Kimi 전용 MXFP4 graph는 GLM adapter와 분리해야 하므로 신규 런타임의 기본 경로로 복사하지 않습니다.
- 현재 GLM-5.3 공개 가중치는 아직 없으므로 GLM-5.2를 실행 가능한 기준 checkpoint로 사용합니다.
- K3 공식 partial checkpoint와 파생 artifact는 기존 저장소의 `.worktrees/milestone-twenty-four-cuda-graph-cache/artifacts/m26-official`부터 `m37-local-foundry`까지를 명시적으로 삭제 대상으로 잡았습니다.
- synthetic K3 fixture는 실제 Kimi 가중치가 아니므로 기존 K3X 저장소에 보존합니다.
- 측정되지 않은 TPS는 README나 benchmark 문서에 쓰지 않습니다.
- 첫 검증에서 GLM descriptor/CLI Python 테스트 4개와 WSL CTest 14개가 통과했습니다. 전체 이식 Python suite는 역사적 results와 Windows build 경로에 의존하는 테스트가 있어 bootstrap 범위에서 제외했습니다.
