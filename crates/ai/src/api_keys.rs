pub use crate::aws_credentials::{AwsCredentials, AwsCredentialsState};
use serde::{Deserialize, Serialize};
use warp_multi_agent_api as api;
use warpui::{Entity, ModelContext, SingletonEntity};
use warpui_extras::secure_storage::{self, AppContextExt};

const SECURE_STORAGE_KEY: &str = "AiApiKeys";

/// Emitted when user-provided API keys are updated in-memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiKeyManagerEvent {
    KeysUpdated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApiKeyProvider {
    Anthropic,
    OpenAI,
    Google,
    OpenRouter,
}

/// User-provided API keys for AI providers.
///
/// These are used for "Bring Your Own API Key" functionality, allowing
/// users to use their own API keys instead of Warp's.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ApiKeys {
    pub google: Option<String>,
    pub anthropic: Option<String>,
    pub openai: Option<String>,
    pub open_router: Option<String>,
    pub openai_base_url: Option<String>,
    pub openai_model: Option<String>,
    pub anthropic_base_url: Option<String>,
    pub anthropic_model: Option<String>,
}

impl ApiKeys {
    pub fn has_any_key(&self) -> bool {
        self.openai.is_some()
            || self.anthropic.is_some()
            || self.google.is_some()
            || self.open_router.is_some()
    }

    pub fn has_any_local_api_key(&self) -> bool {
        self.has_any_key()
    }
}

fn non_empty_trimmed(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn valid_model_id(value: &str) -> bool {
    const MAX_MODEL_ID_LEN: usize = 128;

    !value.is_empty()
        && value.len() <= MAX_MODEL_ID_LEN
        && !value.starts_with("sk-")
        && !value.starts_with("AIza")
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-' | ':' | '/'))
}

fn normalized_model_id(value: Option<String>) -> Option<String> {
    non_empty_trimmed(value).filter(|value| valid_model_id(value))
}

/// Controls how AWS credentials are refreshed by [`ApiKeyManager`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AwsCredentialsRefreshStrategy {
    /// Load credentials from the local AWS credential chain (~/.aws). This is the default.
    #[default]
    LocalChain,
    /// Credentials are managed externally via OIDC/STS.
    /// The task ID is used to scope the STS AssumeRoleWithWebIdentity session.
    /// The role ARN is the IAM role to assume via STS.
    OidcManaged {
        task_id: Option<String>,
        role_arn: String,
    },
}

/// A structure that manages API keys for AI providers.
pub struct ApiKeyManager {
    keys: ApiKeys,
    pub(crate) aws_credentials_state: AwsCredentialsState,
    aws_credentials_refresh_strategy: AwsCredentialsRefreshStrategy,
}

impl ApiKeyManager {
    pub fn new(ctx: &mut ModelContext<Self>) -> Self {
        let keys = Self::load_keys_from_secure_storage(ctx);
        Self {
            keys,
            aws_credentials_state: AwsCredentialsState::Missing,
            aws_credentials_refresh_strategy: AwsCredentialsRefreshStrategy::default(),
        }
    }

    pub fn keys(&self) -> &ApiKeys {
        &self.keys
    }

    pub fn set_google_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.google = non_empty_trimmed(key);
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_anthropic_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.anthropic = non_empty_trimmed(key);
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_openai_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.openai = non_empty_trimmed(key);
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_openai_base_url(&mut self, base_url: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.openai_base_url = non_empty_trimmed(base_url);
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_openai_model(&mut self, model: Option<String>, ctx: &mut ModelContext<Self>) {
        let model = normalized_model_id(model);
        self.keys.openai_model = if model == self.keys.anthropic_model {
            None
        } else {
            model
        };
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_anthropic_base_url(
        &mut self,
        base_url: Option<String>,
        ctx: &mut ModelContext<Self>,
    ) {
        self.keys.anthropic_base_url = non_empty_trimmed(base_url);
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_anthropic_model(&mut self, model: Option<String>, ctx: &mut ModelContext<Self>) {
        let model = normalized_model_id(model);
        self.keys.anthropic_model = if model == self.keys.openai_model {
            None
        } else {
            model
        };
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_open_router_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.open_router = non_empty_trimmed(key);
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_aws_credentials_state(
        &mut self,
        state: AwsCredentialsState,
        ctx: &mut ModelContext<Self>,
    ) {
        self.aws_credentials_state = state;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
    }

    pub fn aws_credentials_state(&self) -> &AwsCredentialsState {
        &self.aws_credentials_state
    }

    pub fn aws_credentials_refresh_strategy(&self) -> AwsCredentialsRefreshStrategy {
        self.aws_credentials_refresh_strategy.clone()
    }

    pub fn set_aws_credentials_refresh_strategy(
        &mut self,
        strategy: AwsCredentialsRefreshStrategy,
    ) {
        self.aws_credentials_refresh_strategy = strategy;
    }

    pub fn api_keys_for_request(
        &self,
        include_byo_keys: bool,
        include_aws_bedrock_credentials: bool,
        providers: &[ApiKeyProvider],
    ) -> Option<api::request::settings::ApiKeys> {
        let include_provider_key =
            |candidate_provider| include_byo_keys && providers.contains(&candidate_provider);
        let anthropic = include_provider_key(ApiKeyProvider::Anthropic)
            .then(|| self.keys.anthropic.clone())
            .flatten()
            .unwrap_or_default();
        let openai = include_provider_key(ApiKeyProvider::OpenAI)
            .then(|| self.keys.openai.clone())
            .flatten()
            .unwrap_or_default();
        let google = include_provider_key(ApiKeyProvider::Google)
            .then(|| self.keys.google.clone())
            .flatten()
            .unwrap_or_default();
        let open_router = include_provider_key(ApiKeyProvider::OpenRouter)
            .then(|| self.keys.open_router.clone())
            .flatten()
            .unwrap_or_default();
        // Also include credentials when running with OIDC-managed Bedrock inference, regardless
        // of the per-user setting flag (which only applies to the local credential chain path).
        let include_aws = include_aws_bedrock_credentials
            || matches!(
                self.aws_credentials_refresh_strategy,
                AwsCredentialsRefreshStrategy::OidcManaged { .. }
            );
        let aws_credentials = include_aws
            .then(|| match self.aws_credentials_state {
                AwsCredentialsState::Loaded {
                    ref credentials, ..
                } => Some(credentials.clone().into()),
                _ => None,
            })
            .flatten();

        if anthropic.is_empty()
            && openai.is_empty()
            && google.is_empty()
            && open_router.is_empty()
            && aws_credentials.is_none()
        {
            None
        } else {
            Some(api::request::settings::ApiKeys {
                anthropic,
                openai,
                google,
                open_router,
                allow_use_of_warp_credits: false,
                aws_credentials,
            })
        }
    }

    fn load_keys_from_secure_storage(ctx: &mut ModelContext<Self>) -> ApiKeys {
        let key_json = match ctx.secure_storage().read_value(SECURE_STORAGE_KEY) {
            Ok(json) => json,
            Err(e) => {
                if !matches!(e, secure_storage::Error::NotFound) {
                    log::error!("Failed to read API keys from secure storage: {e:#}");
                }
                return ApiKeys::default();
            }
        };

        let keys = match serde_json::from_str(&key_json) {
            Ok(keys) => keys,
            Err(e) => {
                log::error!("Failed to deserialize API keys: {e:#}");
                ApiKeys::default()
            }
        };

        keys
    }

    fn write_keys_to_secure_storage(&mut self, ctx: &mut ModelContext<Self>) {
        let keys = self.keys.clone();

        let json = match serde_json::to_string(&keys) {
            Ok(json) => json,
            Err(e) => {
                log::error!("Failed to serialize API keys: {e:#}");
                return;
            }
        };

        if let Err(e) = ctx.secure_storage().write_value(SECURE_STORAGE_KEY, &json) {
            log::error!("Failed to write API keys to secure storage: {e:#}");
        }
    }
}

impl Entity for ApiKeyManager {
    type Event = ApiKeyManagerEvent;
}

impl SingletonEntity for ApiKeyManager {}

#[cfg(test)]
mod tests {
    use super::{normalized_model_id, ApiKeyManager, ApiKeyProvider, AwsCredentialsState};

    #[test]
    fn normalized_model_id_accepts_common_model_ids() {
        assert_eq!(
            normalized_model_id(Some("  claude-sonnet-4-5  ".to_string())),
            Some("claude-sonnet-4-5".to_string())
        );
        assert_eq!(
            normalized_model_id(Some("openai/gpt-4o:latest".to_string())),
            Some("openai/gpt-4o:latest".to_string())
        );
    }

    #[test]
    fn normalized_model_id_rejects_key_like_or_unsafe_values() {
        assert_eq!(normalized_model_id(Some("sk-test".to_string())), None);
        assert_eq!(normalized_model_id(Some("AIza-test".to_string())), None);
        assert_eq!(
            normalized_model_id(Some(
                "model
name"
                    .to_string()
            )),
            None
        );
        assert_eq!(normalized_model_id(Some("a".repeat(129))), None);
    }

    #[test]
    fn api_keys_for_request_only_includes_matching_provider_key() {
        let manager = ApiKeyManager {
            keys: super::ApiKeys {
                openai: Some("openai-key".to_string()),
                anthropic: Some("anthropic-key".to_string()),
                google: Some("google-key".to_string()),
                open_router: Some("open-router-key".to_string()),
                ..Default::default()
            },
            aws_credentials_state: AwsCredentialsState::Missing,
            aws_credentials_refresh_strategy: Default::default(),
        };

        let keys = manager
            .api_keys_for_request(true, false, &[ApiKeyProvider::Anthropic])
            .expect("expected matching provider key");

        assert_eq!(keys.anthropic, "anthropic-key");
        assert!(keys.openai.is_empty());
        assert!(keys.google.is_empty());
        assert!(keys.open_router.is_empty());
    }
}
