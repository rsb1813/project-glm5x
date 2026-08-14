# Security Policy

## Supported Versions

K3X is under active development and has not yet reached a stable release.

Security fixes are currently provided for the latest version of the `main`
branch only.

| Version | Supported |
| ------- | --------- |
| Latest `main` | Yes |
| Older commits and development branches | No |

## Reporting a Vulnerability

Please **do not open a public GitHub issue** for suspected security
vulnerabilities.

Use GitHub's **Private Vulnerability Reporting** feature instead:

1. Open the K3X repository on GitHub.
2. Go to **Security → Advisories**.
3. Select **Report a vulnerability**.
4. Include as much information as possible.

Useful information includes:

- affected K3X commit or version;
- operating system and architecture;
- CPU and GPU, if relevant;
- CUDA version, if relevant;
- exact runtime or converter command;
- minimal reproduction steps;
- malformed or crafted `.k3x` input, when applicable;
- expected and observed behavior;
- crash logs, sanitizer output, or stack traces;
- potential security impact.

Examples of security-sensitive areas include:

- malformed or malicious `.k3x` checkpoint handling;
- memory-safety issues in the C++ or CUDA runtime;
- integer overflow or out-of-bounds access;
- unsafe file or path handling;
- converter or checkpoint-parser vulnerabilities;
- vulnerabilities that may allow code execution, memory corruption,
  unauthorized file access, or denial of service.

Please avoid publicly disclosing a vulnerability before a fix can be
developed and coordinated.

## Security Response

Reports will be reviewed as soon as reasonably possible.

After confirming a vulnerability, maintainers may use a private GitHub
Security Advisory and temporary private fork to develop and validate a fix.

When appropriate, a security advisory will be published after a fix is
available.

## Scope

Performance regressions, model-quality differences, unsupported hardware,
and ordinary crashes that do not have a security impact should be reported
through the normal GitHub issue tracker.
