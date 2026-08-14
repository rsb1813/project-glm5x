# K3X Milestone 6 Independent L2 Reader Design

Milestone 6 replaces the hot-path `std::ifstream`-per-extent implementation with an independently switchable Linux file-I/O layer. It must establish correctness and measurement boundaries before K3X selects buffered I/O, `io_uring`, or `O_DIRECT` as a default.

This milestone does not claim asynchronous NVMe-to-RAM overlap, deadline scheduling, physical NVMe traffic, or a production L2 policy until those behaviors are implemented and measured on native Linux. WSL2's `/mnt/c` path is a 9p/DrvFS development filesystem and is not performance authority for the P44 Pro.

## Current boundary

`Reader::read_extent` currently opens a new `std::ifstream`, seeks, allocates a vector, and synchronously reads for every data or auxiliary extent. One exact native expert causes six calls. Milestone 5 can remove repeated reads after admission, but every miss still pays this path before L1-to-L0 prefetch begins.

Metadata parsing and full-checkpoint verification are cold paths. They may continue using the portable buffered helper. Only post-open tensor and auxiliary reads use the new L2 data-plane object.

## Alternatives

### A. Replace every read with `io_uring` immediately

This has the smallest visible API but keeps the graph synchronous when each submission is immediately awaited. It also makes liburing mandatory and conflates queueing, page-cache behavior, and direct I/O. Rejected.

### B. Add a generic future/executor framework

This could model arbitrary deadlines and dependencies, but it would introduce cancellation, thread-safety, ownership, and scheduler contracts before a batched expert read has demonstrated value. Rejected for this milestone.

### C. Add one bounded batch API with independent engine and cache axes

Selected. The Reader retains exact single-extent wrappers and adds an ordered batch operation. The expert loader requests its six gate/up/down data and scale extents as one batch. The I/O engine and page-cache policy remain independent.

## Public configuration

Two axes describe actual behavior.

- `L2IoEngine::pread` uses one persistent file descriptor and exact positioned reads.
- `L2IoEngine::io_uring` submits a bounded batch and reaps every completion through liburing.
- `L2CacheMode::buffered` opens the file normally.
- `L2CacheMode::direct` opens with `O_DIRECT` and uses aligned bounce buffers.

The CLI spellings are `--l2-io pread|io-uring`, `--l2-cache buffered|direct`, and `--l2-queue-depth N`. Defaults remain `pread + buffered`. The existing `Reader::open(path, VerifyMode)` overload retains source compatibility and selects those defaults.

`io_uring` is a Linux-only optional build capability. Unsupported requested modes must fail at `Reader::open` with a stable error before model execution. Windows and CPU-only Linux builds retain the buffered baseline without liburing.

## Ordered batch contract

An `ExtentRequest` contains an offset and exact logical length. `read_extents` returns one byte vector per request in the same order.

The operation must satisfy all of the following.

- Reject integer overflow or a range beyond the K3X file before allocation or submission.
- Preserve request order even when completions arrive out of order.
- Treat a short read as `truncated_file`.
- Return no partial successful batch to the caller after any request fails.
- Count logical calls and bytes exactly as the single-extent API does.
- Separately count submitted and completed storage bytes, batch submissions, completions, short reads, and failures.
- Never label syscall or `/proc/self/io` counters as physical NVMe traffic.

Single tensor and auxiliary reads delegate to a one-element batch, so checksum and error behavior has one implementation.

## Buffered positioned I/O

The baseline opens one descriptor for the Reader lifetime and uses `pread` in a loop that handles `EINTR`, short progress, EOF, and the platform transfer-size limit. This removes repeated open/seek state without introducing asynchronous semantics.

The first TDD checkpoint implements this mode on Linux and keeps the existing portable fallback elsewhere. Exact K3X bytes, errors, and logical counters must remain unchanged.

## io_uring path

The first io_uring path uses ordinary per-request buffers and explicit file offsets. It batches up to configured queue depth, associates each SQE with a stable request index, submits, and validates each CQE result. Negative CQE results are errors directly rather than `errno` values.

Registered files, fixed buffers, SQPOLL, IOPOLL, provided-buffer rings, and persistent worker threads are excluded from the first comparison. They alter more than one variable and are only candidates for later ablations. The implementation must query opcode support and fail closed when required operations are unavailable.

## Direct I/O path

`O_DIRECT` is not assumed to be supported or faster. At open, Linux `statx` with `STATX_DIOALIGN` is used when available. Direct mode requires nonzero memory and offset/length alignment; absence of a trustworthy alignment contract is an unsupported capability, not silent buffered fallback.

Each logical request expands to an aligned file interval. A suitably aligned bounce buffer owns the submitted interval, and only the requested logical slice is returned. The reader records both logical and aligned storage bytes. A final aligned read may complete short at EOF only when all requested logical bytes are present; missing logical bytes remain `truncated_file`.

Buffered and direct descriptors are never mixed for overlapping hot-path reads inside one Reader. The runtime does not combine `mmap` with direct mode.

## Measurement boundary

B-0007 crosses the two axes when the filesystem supports them.

| Case | Engine | Cache mode |
|---|---|---|
| pread-buffered | pread | buffered |
| io-uring-buffered | io_uring | buffered |
| pread-direct | pread | direct |
| io-uring-direct | io_uring | direct |

The first correctness artifact remains the seeded synthetic K3X model. A second bounded I/O fixture with aligned multi-megabyte expert-like extents is proposed so queue depth and direct-I/O amplification can become representative without downloading Kimi K3; it is not implemented in Milestone 6.

Every row records model tokens/routing, logical Reader calls/bytes, submitted/completed storage bytes, batch count, completion count, queue depth, direct alignment, `/proc/self/io` `rchar` and `read_bytes` deltas when available, elapsed read time, and end-to-end timing. Cache state and filesystem/mount identity must be recorded. Dropping system caches is not automated because it is privileged and globally disruptive.

The default cannot change from `pread + buffered` on WSL2 evidence. Native Linux on the target P44 Pro must compare warm-cache and explicitly prepared cold-cache runs, disclose the preparation method, and repeat enough samples before accepting a mode.

## Failure and portability policy

- The default build has no mandatory liburing dependency.
- `K3X_ENABLE_IO_URING=ON` requires a discovered liburing package and Linux.
- Direct mode reports alignment and filesystem capability failures explicitly.
- Queue depth must be positive and bounded before allocating request state.
- Successful batches return only after all submitted requests are reaped. A submit or completion failure closes the ring while batch buffers are still alive, causing kernel shutdown cancellation, and marks that Reader's io_uring path unavailable for later calls.
- No paid resource, full checkpoint, privileged cache drop, or Cloud Run action belongs to this milestone.

## Acceptance

Milestone 6 is accepted only when exact single and batch reads, out-of-order completion mapping, short/error handling, counter accounting, unsupported-mode behavior, and model token/routing parity pass. B-0007 may select a default only from native-Linux measurements. WSL2 runs are correctness and plumbing evidence only.

## Primary references checked

- Linux `open(2)` documents that `O_DIRECT` alignment varies by filesystem and kernel, may fail or fall back for misaligned I/O, and should be queried with `STATX_DIOALIGN` when supported: <https://man7.org/linux/man-pages/man2/open.2.html>.
- Linux `statx(2)` defines `stx_dio_mem_align`, `stx_dio_offset_align`, and the filesystem-dependent support boundary: <https://man7.org/linux/man-pages/man2/statx.2.html>.
- Upstream liburing documents explicit offsets and negative error results in CQEs for `io_uring_prep_read`: <https://man7.org/linux/man-pages/man3/io_uring_prep_read.3.html>.
- Upstream liburing describes registered buffers as a separate optimization, especially with direct I/O, which is why they are excluded from the first engine comparison: <https://man7.org/linux/man-pages/man3/io_uring_register_buffers.3.html>.
- Linux's io_uring cancellation documentation states that closing a ring cancels all pending requests, which defines the fail-closed buffer-lifetime recovery path: <https://man7.org/linux/man-pages/man7/io_uring_cancelation.7.html>.
- The upstream liburing repository states that library and kernel versions are not locked together and newer operations require runtime/kernel capability handling: <https://github.com/axboe/liburing>.
