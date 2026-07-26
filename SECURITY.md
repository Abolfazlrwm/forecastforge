# Security Policy

## Supported Versions

ForecastForge is currently in a **pre-alpha, repository foundation**
phase (see `README.md`). There are no tagged releases yet, and no
version is currently designated as "supported" in a formal sense. Once
the project reaches a first tagged release, this table will be updated
to reflect which versions receive security fixes.

| Version | Supported |
| ------- | --------- |
| `0.1.x` (pre-release, `main`) | :white_check_mark: (best effort) |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub
issues.**

If you believe you have found a security vulnerability in ForecastForge,
please report it privately using one of the following channels:

1. **GitHub Security Advisories (preferred):** Use the
   ["Report a vulnerability"](https://github.com/forecastforge/forecastforge/security/advisories/new)
   feature under this repository's Security tab. This creates a private
   advisory visible only to maintainers until it is resolved.
2. **Email:** If GitHub Security Advisories are unavailable to you,
   contact the maintainers at `security@forecastforge.dev` with as much
   detail as possible.

Please include, where applicable:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, or a proof-of-concept.
- The affected version, commit, or branch.
- Any known mitigations.

### What to Expect

- **Acknowledgment:** We aim to acknowledge receipt of a vulnerability
  report within **5 business days**.
- **Assessment:** We will investigate and aim to provide an initial
  assessment (severity, affected components) within **10 business
  days** of acknowledgment.
- **Resolution:** Timelines for a fix depend on severity and complexity.
  We will keep the reporter informed of progress and coordinate on an
  appropriate disclosure timeline.
- **Credit:** With the reporter's permission, we are happy to credit
  them in the release notes or a security advisory once the issue is
  resolved.

### Scope

Because ForecastForge does not yet implement data ingestion, model
training, or a serving API, the current attack surface is limited
primarily to the repository's tooling and CI configuration (e.g.
supply-chain concerns in dependencies or GitHub Actions workflows).
Reports concerning dependencies, build configuration, or CI/CD security
are welcome and in scope. This section will be expanded as
functionality (data handling, model serving, etc.) is implemented.

Thank you for helping keep ForecastForge and its users safe.
