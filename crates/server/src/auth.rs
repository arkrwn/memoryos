//! Bearer tokens and scope patterns (SPEC section 6).

use axum::extract::{Request, State};
use axum::http::header::AUTHORIZATION;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use memoryos_core::{ApiKey, MemoryStore, RequestContext, Scope};
use sha2::{Digest, Sha256};

use crate::app::AppState;
use crate::error::{ApiError, ApiResult};

const TOKEN_PREFIX: &str = "mos_";
const TOKEN_BYTES: usize = 32;
const MAX_SCOPES_PER_REQUEST: usize = 50;

/// `mos_` and 43 url-safe characters from 32 random bytes.
pub fn generate_token() -> Result<String, String> {
    let mut bytes = [0u8; TOKEN_BYTES];
    getrandom::fill(&mut bytes).map_err(|e| format!("no system randomness: {e}"))?;
    Ok(format!("{TOKEN_PREFIX}{}", URL_SAFE_NO_PAD.encode(bytes)))
}

/// Lowercase hex SHA-256, the only form of a token that is stored.
pub fn hash_token(token: &str) -> String {
    Sha256::digest(token.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// `*` allows all, a trailing `*` matches a prefix, and anything else must be equal.
pub fn pattern_matches(pattern: &str, scope: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => scope.starts_with(prefix),
        None => pattern == scope,
    }
}

/// For `key create`: `*`, `<kind>:*`, `<kind>:<prefix>*`, or an exact scope. A typo here would
/// otherwise create a key that silently matches nothing.
pub fn validate_pattern(pattern: &str) -> Result<(), String> {
    if pattern == "*" {
        return Ok(());
    }
    let candidate = match pattern.strip_suffix('*') {
        Some(prefix) if prefix.ends_with(':') => format!("{prefix}x"),
        Some(prefix) => prefix.to_owned(),
        None => pattern.to_owned(),
    };
    candidate
        .parse::<Scope>()
        .map(|_| ())
        .map_err(|_| format!("invalid scope pattern: {pattern}"))
}

/// What a request may list, after `Caller::select_scopes`: every scope it covers is one the key
/// allows.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ScopeSelection {
    pub exact: Vec<Scope>,
    pub prefixes: Vec<String>,
}

/// Adds the overlap of the prefix `wanted` and one key pattern. Returns whether there is one.
fn narrow(wanted: &str, pattern: &str, selection: &mut ScopeSelection) -> ApiResult<bool> {
    match pattern.strip_suffix('*') {
        Some(allowed) if allowed.starts_with(wanted) => {
            selection.prefixes.push(allowed.to_owned());
            Ok(true)
        }
        Some(allowed) if wanted.starts_with(allowed) => {
            selection.prefixes.push(wanted.to_owned());
            Ok(true)
        }
        None if pattern.starts_with(wanted) => {
            let scope = pattern.parse::<Scope>().map_err(ApiError::from)?;
            selection.exact.push(scope);
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// The authenticated key. Every check of SPEC section 6 goes through here.
#[derive(Debug, Clone)]
pub struct Caller {
    pub key: ApiKey,
    pub ctx: RequestContext,
}

impl Caller {
    pub fn new(key: ApiKey) -> Self {
        let ctx = RequestContext {
            tenant: key.tenant.clone(),
            actor: format!("key:{}", key.id),
        };
        Self { key, ctx }
    }

    fn allows(&self, scope: &Scope) -> bool {
        let scope = scope.to_string();
        self.key
            .scopes
            .iter()
            .any(|pattern| pattern_matches(pattern, &scope))
    }

    /// For a scope the request names: the caller already knows it, so refusing reveals nothing.
    pub fn require_scope(&self, scope: &Scope) -> ApiResult<()> {
        if self.allows(scope) {
            Ok(())
        } else {
            Err(ApiError::forbidden(format!(
                "this key does not allow scope {scope}"
            )))
        }
    }

    pub fn require_scopes(&self, scopes: &[Scope]) -> ApiResult<()> {
        if scopes.is_empty() || scopes.len() > MAX_SCOPES_PER_REQUEST {
            return Err(ApiError::invalid_input(format!(
                "scopes must have 1 to {MAX_SCOPES_PER_REQUEST} entries, got {}",
                scopes.len()
            )));
        }
        scopes
            .iter()
            .try_for_each(|scope| self.require_scope(scope))
    }

    /// For a request that may name patterns. An exact scope is checked as `require_scope` does. A
    /// pattern is narrowed to its overlap with the key's patterns, so `user:*` from a key limited
    /// to `user:a` covers `user:a` alone. A pattern with no overlap is refused.
    pub fn select_scopes(&self, requested: &[String]) -> ApiResult<ScopeSelection> {
        if requested.is_empty() || requested.len() > MAX_SCOPES_PER_REQUEST {
            return Err(ApiError::invalid_input(format!(
                "scopes must have 1 to {MAX_SCOPES_PER_REQUEST} entries, got {}",
                requested.len()
            )));
        }
        let mut selection = ScopeSelection::default();
        for item in requested {
            validate_pattern(item).map_err(ApiError::invalid_input)?;
            let Some(wanted) = item.strip_suffix('*') else {
                let scope = item.parse::<Scope>().map_err(ApiError::from)?;
                self.require_scope(&scope)?;
                selection.exact.push(scope);
                continue;
            };
            let mut overlaps = false;
            for pattern in &self.key.scopes {
                overlaps |= narrow(wanted, pattern, &mut selection)?;
            }
            if !overlaps {
                return Err(ApiError::forbidden(format!(
                    "this key does not allow any scope matching {item}"
                )));
            }
        }
        Ok(selection)
    }

    /// Whether the key may read a row in this scope.
    pub fn may_read(&self, scope: &Scope) -> bool {
        self.allows(scope)
    }

    /// For a row loaded by id: a mismatch looks the same as an id that does not exist.
    pub fn require_visible(&self, scope: &Scope) -> ApiResult<()> {
        if self.allows(scope) {
            Ok(())
        } else {
            Err(ApiError::not_found())
        }
    }

    pub fn require_admin(&self) -> ApiResult<()> {
        if self.key.is_admin {
            Ok(())
        } else {
            Err(ApiError::forbidden("this operation requires an admin key"))
        }
    }
}

/// Resolves the token on every request, so a revocation applies at once.
pub async fn authenticate(
    State(state): State<AppState>,
    mut request: Request,
    next: Next,
) -> Response {
    // Owned, because a request is not `Sync` and so cannot be borrowed across the lookup.
    let token = bearer_token(&request).map(str::to_owned);
    let caller = match resolve(&state, token).await {
        Ok(caller) => caller,
        Err(error) => return error.into_response(),
    };
    if let Err(wait) = state.limiter.check(caller.key.id) {
        return ApiError::rate_limited(wait.as_secs_f64().ceil().max(1.0) as u64).into_response();
    }
    request.extensions_mut().insert(caller);
    next.run(request).await
}

async fn resolve(state: &AppState, token: Option<String>) -> ApiResult<Caller> {
    let token = token.ok_or_else(ApiError::unauthorized)?;
    let hash = hash_token(&token);
    state.failed_auth.check(&hash).map_err(|wait| {
        ApiError::too_many_failed_tokens(wait.as_secs_f64().ceil().max(1.0) as u64)
    })?;
    match state.services.store.find_api_key(&hash).await? {
        Some(key) => {
            state.failed_auth.succeeded(&hash);
            Ok(Caller::new(key))
        }
        None => {
            state.failed_auth.failed();
            Err(ApiError::unauthorized())
        }
    }
}

fn bearer_token(request: &Request) -> Option<&str> {
    let value = request.headers().get(AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| token.trim())
}

#[cfg(test)]
mod tests {
    use memoryos_core::Trust;
    use uuid::Uuid;

    use super::*;

    fn select(key: &[&str], requested: &[&str]) -> ApiResult<ScopeSelection> {
        let requested: Vec<String> = requested.iter().map(|s| (*s).to_owned()).collect();
        caller(key).select_scopes(&requested)
    }

    fn selection(exact: &[&str], prefixes: &[&str]) -> ScopeSelection {
        ScopeSelection {
            exact: exact.iter().map(|s| s.parse().expect("scope")).collect(),
            prefixes: prefixes.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    #[test]
    fn a_wildcard_narrows_to_what_the_key_allows() {
        let cases: [(&[&str], &str, ScopeSelection); 7] = [
            (&["user:a"], "user:*", selection(&["user:a"], &[])),
            (&["user:a"], "*", selection(&["user:a"], &[])),
            (&["user:a*"], "user:*", selection(&[], &["user:a"])),
            (&["user:*"], "user:a*", selection(&[], &["user:a"])),
            (&["user:*"], "user:*", selection(&[], &["user:"])),
            (&["*"], "user:*", selection(&[], &["user:"])),
            (&["*"], "*", selection(&[], &[""])),
        ];
        for (key, requested, expected) in cases {
            assert_eq!(select(key, &[requested]).expect(requested), expected);
        }
    }

    #[test]
    fn a_wildcard_with_no_overlap_is_refused() {
        for (key, requested) in [
            (&["user:a"][..], "user:b*"),
            (&["user:a"][..], "agent:*"),
            (&["user:a*"][..], "user:b*"),
            (&["user:a"][..], "user:ab*"),
        ] {
            assert!(select(key, &[requested]).is_err(), "{requested}");
        }
    }

    #[test]
    fn exact_scopes_are_still_checked() {
        assert_eq!(
            select(&["user:*"], &["user:a"]).expect("allowed"),
            selection(&["user:a"], &[])
        );
        assert!(select(&["user:a"], &["user:b"]).is_err());
    }

    #[test]
    fn a_malformed_pattern_is_invalid_and_the_count_is_limited() {
        for bad in ["team:*", "us*er:a", "user:a*b", "user:a**", "use*", ""] {
            assert!(select(&["*"], &[bad]).is_err(), "{bad:?}");
        }
        assert!(select(&["*"], &[]).is_err());
        assert!(select(&["*"], &vec!["user:a"; 51]).is_err());
    }

    fn caller(scopes: &[&str]) -> Caller {
        Caller::new(ApiKey {
            id: Uuid::now_v7(),
            tenant: "t".into(),
            name: "n".into(),
            scopes: scopes.iter().map(|s| (*s).to_owned()).collect(),
            max_trust: Trust::Agent,
            is_admin: false,
        })
    }

    #[test]
    fn token_has_the_documented_shape_and_hash() {
        let token = generate_token().expect("token");
        assert_eq!(token.len(), 4 + 43);
        assert!(token.starts_with("mos_"));
        assert!(token[4..]
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
        assert_ne!(generate_token().expect("token"), token);
        // Known answer for SHA-256("abc").
        assert_eq!(
            hash_token("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn patterns_match_all_prefix_and_exact() {
        assert!(pattern_matches("*", "user:a"));
        assert!(pattern_matches("user:*", "user:a"));
        assert!(!pattern_matches("user:*", "agent:a"));
        assert!(!pattern_matches("user:*", "users:a"));
        assert!(pattern_matches("user:a", "user:a"));
        assert!(!pattern_matches("user:a", "user:ab"));
        assert!(pattern_matches("user:a*", "user:ab"));
    }

    #[test]
    fn create_refuses_patterns_that_could_never_match() {
        for good in ["*", "user:*", "shared:team-*", "agent:a.b"] {
            assert!(validate_pattern(good).is_ok(), "{good}");
        }
        for bad in ["", "user", "team:*", "user:a b", "us*", "user:*x"] {
            assert!(validate_pattern(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn named_scope_is_403_and_loaded_row_is_404() {
        let key = caller(&["user:a"]);
        let other = Scope::user("b");
        assert!(key.require_scope(&Scope::user("a")).is_ok());
        assert_eq!(
            key.require_scope(&other).err().map(|e| e.status.as_u16()),
            Some(403)
        );
        assert_eq!(
            key.require_visible(&other).err().map(|e| e.status.as_u16()),
            Some(404)
        );
    }

    #[test]
    fn scope_lists_must_have_1_to_50_entries() {
        let key = caller(&["*"]);
        assert!(key.require_scopes(&[]).is_err());
        let fifty: Vec<Scope> = (0..50).map(|i| Scope::user(format!("u{i}"))).collect();
        assert!(key.require_scopes(&fifty).is_ok());
        let many: Vec<Scope> = (0..51).map(|i| Scope::user(format!("u{i}"))).collect();
        assert_eq!(
            key.require_scopes(&many).err().map(|e| e.status.as_u16()),
            Some(400)
        );
    }

    #[test]
    fn one_forbidden_scope_refuses_the_whole_list() {
        let key = caller(&["user:a"]);
        let scopes = [Scope::user("a"), Scope::user("b")];
        assert_eq!(
            key.require_scopes(&scopes).err().map(|e| e.status.as_u16()),
            Some(403)
        );
    }
}
