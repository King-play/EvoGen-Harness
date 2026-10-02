# Security and private information

Do not post API keys, credentials, private prompts, or unredacted model traces in public issues or pull requests. Keep `.env` and local run directories out of Git. Review generated artifacts before redistribution.

For a vulnerability, use GitHub's private vulnerability reporting interface **when enabled by the repository owner**. If it is unavailable, request a private reporting channel from a maintainer without publishing exploit details or sensitive data. No unverified security email address is listed here.

Model adapters and configured external commands execute code. In particular, the NVILA verifier uses `trust_remote_code=True`; inspect and pin model code before use. Do not load untrusted repositories or shell configurations with privileged credentials.

`open-source-check` and `check_release.py` are heuristic checks, not a full security audit or a guarantee of absence of secrets. Rotate a credential that has been exposed; deleting a file in a later commit does not remove it from Git history.
