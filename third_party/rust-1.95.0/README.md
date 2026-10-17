# Rust 1.95.0 license supplement

Unmodified root license texts from the Rust compiler source commit reported by
`rustc -Vv`: `59807616e1fa2540724bfbac14d7976d7e4a3860`. Retrieved 2026-10-17.
The rustup `rust-docs` component supplies `COPYRIGHT-library.html` but does not
always include these two root text files. The packaging tool uses this fallback
only for exactly Rust 1.95.0 and always obtains the library-specific copyright
notice from the actual installed toolchain.

| File | Upstream | SHA256 |
| --- | --- | --- |
| LICENSE-APACHE | https://github.com/rust-lang/rust/blob/59807616e1fa2540724bfbac14d7976d7e4a3860/LICENSE-APACHE | `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a` |
| LICENSE-MIT | https://github.com/rust-lang/rust/blob/59807616e1fa2540724bfbac14d7976d7e4a3860/LICENSE-MIT | `b71bd43a069ca0641a9ecfe585ca7b3c53b5cc1608f8b68321168698e28b5ea1` |

These texts retain their upstream terms. Review the toolchain's copyright notice
and native runtime obligations per target before public distribution. An upgrade
to a compiler without its root license texts needs a new reviewed supplement.
