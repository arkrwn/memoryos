//! Secret masking for commits (SPEC section 3). A host model is told not to save secrets, but a
//! statement or a quoted source can still carry one, and memory outlives the conversation that
//! leaked it. Every match becomes `[redacted]` before anything is stored. The patterns favor
//! precision: a token is only masked when its shape is distinctive, so ids, hashes and prose stay.

use std::sync::LazyLock;

use regex::{Captures, Regex};

pub(crate) const MASK: &str = "[redacted]";

/// In order: a pattern that runs earlier masks text a later one would otherwise see. The `keep`
/// group, where present, is the label in front of the secret, which stays readable.
static PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        // A PEM private key block, or its start up to the end of the text if it was cut off.
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----(?s:.*?)(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\z)",
        // A JSON Web Token: base64url header and payload, both JSON objects, then a signature.
        r"\beyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*",
        r"(?i)(?P<keep>\bbearer\s+)[A-Za-z0-9._~+/=-]{16,}",
        // OpenAI and Anthropic style keys: sk-..., sk-proj-..., sk-ant-...
        r"\bsk-[A-Za-z0-9_-]{20,}",
        r"\bgh[pousr]_[A-Za-z0-9]{30,}",
        r"\bgithub_pat_[A-Za-z0-9_]{22,}",
        r"\bAKIA[0-9A-Z]{16}\b",
        // MemoryOS API keys.
        r"\bmos_[A-Za-z0-9_-]{20,}",
        r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
        // `password=...`, `db_password: ...`, `secret=...`, `access_token=...`, `api_key=...`. A
        // value already masked starts with `[` and is left alone.
        r#"(?i)(?P<keep>\b[a-z0-9_]*(?:password|passwd|secret|token|api_?key)["']?\s*[:=]\s*)(?:"[^"\n]+"|'[^'\n]+'|[^\s"',;\[][^\s"',;]*)"#,
    ]
    .iter()
    .map(|pattern| Regex::new(pattern).expect("a valid secret pattern"))
    .collect()
});

/// Masks every secret in `text`. Returns whether anything was masked.
pub(crate) fn redact(text: &mut String) -> bool {
    let mut changed = false;
    for pattern in PATTERNS.iter() {
        if !pattern.is_match(text) {
            continue;
        }
        let masked = pattern
            .replace_all(text, |caps: &Captures<'_>| {
                let keep = caps.name("keep").map_or("", |m| m.as_str());
                format!("{keep}{MASK}")
            })
            .into_owned();
        changed |= masked != *text;
        *text = masked;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn masked(text: &str) -> String {
        let mut text = text.to_owned();
        assert!(redact(&mut text), "nothing masked in {text:?}");
        text
    }

    fn untouched(text: &str) {
        let mut copy = text.to_owned();
        assert!(!redact(&mut copy), "masked {text:?} as {copy:?}");
        assert_eq!(copy, text);
    }

    #[test]
    fn provider_keys_are_masked() {
        let cases = [
            (
                "key sk-proj-AbCdEfGhIjKlMnOpQrStUvWx12 here",
                "key [redacted] here",
            ),
            ("sk-ant-api03-abcdefghijklmnopqrstuv", "[redacted]"),
            (
                "token ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789 used",
                "token [redacted] used",
            ),
            (
                "github_pat_11ABCDEFG0123456789_abcdefghijklmnop",
                "[redacted]",
            ),
            ("AWS key AKIAIOSFODNN7EXAMPLE.", "AWS key [redacted]."),
            (
                "use mos_4f8Kx2LmQ9vTzR1aB7cD3eF5gH6iJ0kL to connect",
                "use [redacted] to connect",
            ),
            ("xoxb-1234567890-abcdefghij", "[redacted]"),
            ("slack xoxp-12345-67890-abcdef", "slack [redacted]"),
        ];
        for (input, expected) in cases {
            assert_eq!(masked(input), expected, "{input:?}");
        }
    }

    #[test]
    fn jwt_and_bearer_tokens_are_masked() {
        let jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";
        assert_eq!(
            masked(&format!("the jwt {jwt} expired")),
            "the jwt [redacted] expired"
        );
        assert_eq!(
            masked("Authorization: Bearer abcdef0123456789ABCDEF"),
            "Authorization: Bearer [redacted]"
        );
        assert_eq!(
            masked(&format!("Authorization: Bearer {jwt}")),
            "Authorization: Bearer [redacted]"
        );
    }

    #[test]
    fn pem_private_keys_are_masked_whole() {
        let pem = "before\n-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\nabc\n-----END RSA PRIVATE KEY-----\nafter";
        assert_eq!(masked(pem), "before\n[redacted]\nafter");
        let cut = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkq";
        assert_eq!(masked(cut), "[redacted]");
    }

    #[test]
    fn key_value_secrets_keep_their_label() {
        let cases = [
            ("password=hunter2", "password=[redacted]"),
            ("db passwd: s3cr3t!", "db passwd: [redacted]"),
            ("secret = 'abc def'", "secret = [redacted]"),
            ("client_secret=\"xyz\"", "client_secret=[redacted]"),
            ("access_token=abc123&x=1", "access_token=[redacted]"),
            ("API_KEY: 1234abcd", "API_KEY: [redacted]"),
            ("DB_PASSWORD=pa55, then", "DB_PASSWORD=[redacted], then"),
        ];
        for (input, expected) in cases {
            let mut text = input.to_owned();
            redact(&mut text);
            assert_eq!(text, expected, "{input:?}");
        }
    }

    #[test]
    fn masking_twice_changes_nothing_more() {
        let mut text = masked("password=hunter2 and sk-abcdefghijklmnopqrstuvwxyz");
        assert_eq!(text, "password=[redacted] and [redacted]");
        assert!(!redact(&mut text));
    }

    #[test]
    fn ordinary_text_and_identifiers_are_left_alone() {
        for text in [
            "Ada lives in Bandung",
            "Reset your password from the settings page",
            "The token was lost, and the secret is out",
            "Fact 0192f3a4-5b6c-7d8e-9f01-23456789abcd was superseded",
            "Fixed in commit 3f2a9c1e8b7d6f5a4c3b2a1908f7e6d5c4b3a291",
            "Short sha 3f2a9c1",
            "She uses scikit-learn and sk-learn style names",
            "The bearer of bad news",
            "ghp_short",
            "AKIA is a prefix",
            "mos_ alone, or mos_short",
            "token: [redacted]",
            "eyJ is how base64 JSON starts",
            "x-ray, task-list, sky-high",
        ] {
            untouched(text);
        }
    }
}
