# K3 artifact deletion record

Date: 2026-08-14.

The following directories were explicitly removed from the old K3X worktree during the GLM5X migration.

```text
C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-twenty-four-cuda-graph-cache\artifacts\m26-official
C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-twenty-four-cuda-graph-cache\artifacts\m28-official-moe
C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-twenty-four-cuda-graph-cache\artifacts\m29-official-layer
C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-twenty-four-cuda-graph-cache\artifacts\m33-official-two-layer
C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-twenty-four-cuda-graph-cache\artifacts\m36-official-first-token
C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-twenty-four-cuda-graph-cache\artifacts\m37-local-foundry
```

The pre-delete inventory contained 3,250 files totaling 59,399,750,843 bytes, about 55.32 decimal GB. The deletion covered official K3 source shards, derived K3X artifacts, conversion blobs, and first-token materialization cache. Synthetic fixtures in other worktrees were intentionally left intact.

