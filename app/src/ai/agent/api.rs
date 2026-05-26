pub(crate) mod convert_conversation;
mod convert_from;
mod convert_to;
mod r#impl;

pub use ai::agent::convert::ConvertToAPITypeError;
use ai::api_keys::{ApiKeyManager, ApiKeyProvider};
pub use convert_from::{
    user_inputs_from_messages, ConversionParams, ConvertAPIMessageToClientOutputMessage,
    MaybeAIAgentOutputMessage, MessageToAIAgentOutputMessageError,
};

pub use r#impl::generate_multi_agent_output;

use futures_lite::Stream;
use serde::Serialize;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use warp_core::channel::ChannelState;
use warp_core::execution_mode::AppExecutionMode;
use warp_core::features::FeatureFlag;

use crate::ai::agent::conversation::AIConversationId;
use crate::ai::ambient_agents::AmbientAgentTaskId;
use crate::{
    ai::{
        blocklist::SessionContext,
        llms::{LLMId, LLMPreferences, LLMProvider},
    },
    server::server_api::AIApiError,
};

use super::{AIAgentInput, MCPContext, MCPServer, RequestMetadata, Suggestions};
use crate::ai::blocklist::{BlocklistAIPermissions, RequestInput};
use crate::ai::execution_profiles::profiles::AIExecutionProfilesModel;
use crate::ai::mcp::templatable_manager::TemplatableMCPServerInfo;
use crate::ai::mcp::TemplatableMCPServerManager;
use crate::settings::AISettings;
use crate::terminal::safe_mode_settings::get_secret_obfuscation_mode;
use crate::workspaces::user_workspaces::UserWorkspaces;
use warp_core::user_preferences::GetUserPreferences;
use warpui::{AppContext, EntityId, SingletonEntity as _};

/// Unique, server-generated conversation-scoped token to be roundtripped to the API when sending
/// requests that follow-up within a given conversation.
#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ServerConversationToken(String);

impl ServerConversationToken {
    pub fn new(id: String) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn debug_link(&self) -> String {
        format!(
            "{}/debug/maa/{}",
            ChannelState::server_root_url(),
            self.as_str()
        )
    }

    pub fn conversation_link(&self) -> String {
        format!(
            "{}/conversation/{}",
            ChannelState::server_root_url(),
            self.as_str()
        )
    }
}

impl From<ServerConversationToken> for String {
    fn from(value: ServerConversationToken) -> Self {
        value.0
    }
}

// Conversions between AI ServerConversationToken and protocol ServerConversationToken
impl From<session_sharing_protocol::common::ServerConversationToken> for ServerConversationToken {
    fn from(token: session_sharing_protocol::common::ServerConversationToken) -> Self {
        Self(token.to_string())
    }
}

impl TryFrom<ServerConversationToken>
    for session_sharing_protocol::common::ServerConversationToken
{
    type Error = uuid::Error;

    fn try_from(token: ServerConversationToken) -> Result<Self, Self::Error> {
        token.as_str().parse()
    }
}

#[derive(Debug, Clone)]
pub struct RequestParams {
    pub input: Vec<AIAgentInput>,
    pub conversation_token: Option<ServerConversationToken>,
    pub forked_from_conversation_token: Option<ServerConversationToken>,
    pub ambient_agent_task_id: Option<AmbientAgentTaskId>,
    pub tasks: Vec<warp_multi_agent_api::Task>,
    pub existing_suggestions: Option<Suggestions>,
    pub metadata: Option<RequestMetadata>,
    pub session_context: SessionContext,
    pub model: LLMId,
    #[allow(unused)]
    pub coding_model: LLMId,
    pub cli_agent_model: LLMId,
    pub computer_use_model: LLMId,
    pub is_memory_enabled: bool,
    pub warp_drive_context_enabled: bool,
    pub context_window_limit: Option<u32>,
    pub mcp_context: Option<MCPContext>,
    pub planning_enabled: bool,
    should_redact_secrets: bool,

    /// User-provided API keys for AI providers (BYO API Key).
    pub api_keys: Option<warp_multi_agent_api::request::settings::ApiKeys>,
    pub allow_use_of_warp_credits_with_byok: bool,
    pub local_direct_config: Option<super::local::LocalDirectConfig>,
    pub autonomy_level: warp_multi_agent_api::AutonomyLevel,
    pub isolation_level: warp_multi_agent_api::IsolationLevel,
    pub web_search_enabled: bool,
    pub computer_use_enabled: bool,
    pub ask_user_question_enabled: bool,
    pub research_agent_enabled: bool,
    pub orchestration_enabled: bool,
    pub supported_tools_override: Option<Vec<warp_multi_agent_api::ToolType>>,
    /// The conversation ID of the parent agent that spawned this child agent, if any.
    pub parent_agent_id: Option<String>,
    /// The display name for this agent (e.g. "Agent 1"), assigned by the orchestrator.
    pub agent_name: Option<String>,
}

pub type Event = Result<warp_multi_agent_api::ResponseEvent, Arc<AIApiError>>;

#[cfg(not(target_family = "wasm"))]
pub type ResponseStream = Pin<Box<dyn Stream<Item = Event> + Send + 'static>>;

// The WASM version of this type has no bound on `Send`, which is an unnecessary bound when
// targeting wasm because the browser is single-threaded (and we don't leverage WebWorkers for async
// execution in WoW).
#[cfg(target_family = "wasm")]
pub type ResponseStream = Pin<Box<dyn Stream<Item = Event>>>;

fn local_openai_direct_config_for_model(
    api_keys: &ai::api_keys::ApiKeys,
    model_id: &LLMId,
    vision_supported: bool,
) -> Option<super::local::LocalDirectConfig> {
    let (api_key, base_url, model) = api_keys.local_openai_config()?;
    (model_id.as_str() == model).then(|| super::local::LocalDirectConfig {
        api_key: api_key.to_string(),
        base_url: base_url.to_string(),
        model: model.to_string(),
        vision_supported,
        cost_telemetry: Default::default(),
    })
}

#[derive(Debug, Clone)]
pub struct ConversationData {
    pub id: AIConversationId,
    pub tasks: Vec<warp_multi_agent_api::Task>,
    pub server_conversation_token: Option<ServerConversationToken>,
    pub forked_from_conversation_token: Option<ServerConversationToken>,
    pub ambient_agent_task_id: Option<AmbientAgentTaskId>,
    pub existing_suggestions: Option<Suggestions>,
}

impl RequestParams {
    pub fn new(
        terminal_view_id: Option<EntityId>,
        session_context: SessionContext,
        request_input: &RequestInput,
        conversation: ConversationData,
        metadata: Option<RequestMetadata>,
        app: &AppContext,
    ) -> Self {
        let ai_settings = AISettings::as_ref(app);
        let is_memory_enabled = ai_settings.is_memory_enabled(app);
        let warp_drive_context_enabled = ai_settings.is_warp_drive_context_enabled(app);

        // Build MCP context - either grouped by server or flat lists based on feature flag
        let mcp_context = if FeatureFlag::MCPGroupedServerContext.is_enabled() {
            // Group MCP tools and resources by server
            let templatable_manager = TemplatableMCPServerManager::as_ref(app);

            let mut active_servers: Vec<&TemplatableMCPServerInfo> = templatable_manager
                .get_active_templatable_servers()
                .values()
                .copied()
                .collect();

            // If file-based MCP servers are enabled, add active servers in scope of
            // the user's current working directory
            if let Some(cwd) = session_context.current_working_directory() {
                active_servers.extend(
                    templatable_manager
                        .get_active_file_based_servers(Path::new(cwd), app)
                        .values(),
                );
            }

            // Include any ephemeral MCP servers started via the Oz CLI.
            active_servers.extend(
                templatable_manager
                    .get_active_cli_spawned_servers()
                    .values(),
            );

            let servers: Vec<MCPServer> = active_servers
                .into_iter()
                .map(|server| MCPServer {
                    name: server.name().to_string(),
                    description: server.description().unwrap_or_default().to_string(),
                    id: server.installation_id().to_string(),
                    resources: server.resources().to_vec(),
                    tools: server.tools().to_vec(),
                })
                .collect();

            if servers.is_empty() {
                None
            } else {
                #[allow(deprecated)]
                Some(MCPContext {
                    resources: vec![],
                    tools: vec![],
                    servers,
                })
            }
        } else {
            // Flat lists of resources and tools
            let templatable_mcp_manager = TemplatableMCPServerManager::as_ref(app);
            let resources = templatable_mcp_manager
                .resources()
                .cloned()
                .collect::<Vec<_>>();
            let tools = templatable_mcp_manager.tools().cloned().collect::<Vec<_>>();

            #[allow(deprecated)]
            (!resources.is_empty() || !tools.is_empty()).then_some(MCPContext {
                resources,
                tools,
                servers: vec![],
            })
        };

        let should_redact_secrets = get_secret_obfuscation_mode(app).should_redact_secret();

        let user_workspaces = UserWorkspaces::as_ref(app);
        let include_byo_keys = user_workspaces.is_byo_api_key_enabled(app);
        let mut model_providers = Vec::new();
        let llm_preferences = LLMPreferences::as_ref(app);
        for model_id in [
            &request_input.model_id,
            &request_input.coding_model_id,
            &request_input.cli_agent_model_id,
            &request_input.computer_use_model_id,
        ] {
            let Some(provider) =
                llm_preferences
                    .get_llm_info(model_id)
                    .and_then(|info| match &info.provider {
                        LLMProvider::Anthropic => Some(ApiKeyProvider::Anthropic),
                        LLMProvider::OpenAI => Some(ApiKeyProvider::OpenAI),
                        LLMProvider::Google => Some(ApiKeyProvider::Google),
                        LLMProvider::Xai | LLMProvider::Unknown => None,
                    })
            else {
                continue;
            };
            if !model_providers.contains(&provider) {
                model_providers.push(provider);
            }
        }
        let api_key_manager = ApiKeyManager::as_ref(app);
        let api_keys = api_key_manager.api_keys_for_request(
            include_byo_keys,
            user_workspaces.is_aws_bedrock_credentials_enabled(app),
            &model_providers,
        );
        let vision_supported = llm_preferences
            .get_llm_info(&request_input.model_id)
            .map(|info| info.vision_supported)
            .unwrap_or(true);
        let local_direct_config = local_openai_direct_config_for_model(
            api_key_manager.keys(),
            &request_input.model_id,
            vision_supported,
        );
        let allow_use_of_warp_credits_with_byok =
            *AISettings::as_ref(app).can_use_warp_credits_with_byok;

        let app_execution_mode = AppExecutionMode::as_ref(app);
        let autonomy_level = if app_execution_mode.is_autonomous() {
            warp_multi_agent_api::AutonomyLevel::Unsupervised
        } else {
            warp_multi_agent_api::AutonomyLevel::Supervised
        };

        let isolation_level = if app_execution_mode.is_sandboxed() {
            warp_multi_agent_api::IsolationLevel::Sandbox
        } else {
            warp_multi_agent_api::IsolationLevel::None
        };

        let web_search_enabled =
            BlocklistAIPermissions::as_ref(app).get_web_search_enabled(app, terminal_view_id);
        let research_agent_enabled = app
            .private_user_preferences()
            .read_value("ResearchAgentEnabled")
            .ok()
            .flatten()
            .and_then(|s| s.parse().ok())
            .unwrap_or_default();
        let is_ambient_agent = conversation.ambient_agent_task_id.is_some();
        let computer_use_enabled = FeatureFlag::AgentModeComputerUse.is_enabled()
            && BlocklistAIPermissions::as_ref(app)
                .get_computer_use_setting(app, terminal_view_id)
                .is_enabled()
            && computer_use::is_supported_on_current_platform()
            && (FeatureFlag::LocalComputerUse.is_enabled() || is_ambient_agent);
        let ask_user_question_enabled = BlocklistAIPermissions::as_ref(app)
            .get_ask_user_question_setting(app, terminal_view_id)
            != crate::ai::execution_profiles::AskUserQuestionPermission::Never;

        let orchestration_enabled = ai_settings.is_orchestration_enabled(app)
            && session_context
                .session_type()
                .as_ref()
                .is_none_or(|t| matches!(t, crate::terminal::model::session::SessionType::Local));

        // Reconcile the persisted override against the active base model's
        // current `LLMContextWindow` instead of trusting whatever was stored
        // last. If the active model isn't configurable or has been removed
        // server-side, drop the override; otherwise clamp it to the model's
        // current `[min, max]` range. This closes the window between an
        // in-flight model metadata refresh and the next request.
        let context_window_limit = {
            let profile_data = AIExecutionProfilesModel::as_ref(app)
                .active_profile(terminal_view_id, app)
                .data()
                .clone();
            profile_data
                .configurable_context_window(app)
                .and_then(|cw| {
                    profile_data
                        .context_window_limit
                        .map(|v| v.clamp(cw.min, cw.max))
                })
        };

        Self {
            input: request_input.all_inputs().cloned().collect(),
            conversation_token: conversation.server_conversation_token,
            forked_from_conversation_token: conversation.forked_from_conversation_token,
            ambient_agent_task_id: conversation.ambient_agent_task_id,
            tasks: conversation.tasks,
            existing_suggestions: conversation.existing_suggestions,
            context_window_limit,
            metadata,
            session_context,
            model: request_input.model_id.clone(),
            coding_model: request_input.coding_model_id.clone(),
            cli_agent_model: request_input.cli_agent_model_id.clone(),
            computer_use_model: request_input.computer_use_model_id.clone(),
            is_memory_enabled,
            warp_drive_context_enabled,
            mcp_context,
            planning_enabled: true,
            should_redact_secrets,
            api_keys,
            allow_use_of_warp_credits_with_byok,
            local_direct_config,
            autonomy_level,
            isolation_level,
            web_search_enabled,
            computer_use_enabled,
            ask_user_question_enabled,
            research_agent_enabled,
            orchestration_enabled,
            supported_tools_override: request_input.supported_tools_override.clone(),
            parent_agent_id: None,
            agent_name: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::local_openai_direct_config_for_model;
    use crate::ai::llms::LLMId;

    #[test]
    fn local_direct_config_uses_dedicated_local_openai_fields() {
        let keys = ai::api_keys::ApiKeys {
            openai: Some("hosted-openai-key".to_string()),
            openai_base_url: Some("https://api.openai.com/v1".to_string()),
            openai_model: Some("gpt-4o".to_string()),
            local_openai_api_key: Some("local-key".to_string()),
            local_openai_base_url: Some("http://localhost:11434/v1".to_string()),
            local_openai_model: Some("qwen2.5-coder".to_string()),
            ..Default::default()
        };

        let config =
            local_openai_direct_config_for_model(&keys, &LLMId::from("qwen2.5-coder"), false)
                .expect("expected local direct config");

        assert_eq!(config.api_key, "local-key");
        assert_eq!(config.base_url, "http://localhost:11434/v1");
        assert_eq!(config.model, "qwen2.5-coder");
        assert!(!config.vision_supported);
    }

    #[test]
    fn local_direct_config_does_not_route_hosted_openai_model_when_local_model_differs() {
        let keys = ai::api_keys::ApiKeys {
            openai: Some("hosted-openai-key".to_string()),
            local_openai_api_key: Some("local-key".to_string()),
            local_openai_base_url: Some("http://localhost:11434/v1".to_string()),
            local_openai_model: Some("qwen2.5-coder".to_string()),
            ..Default::default()
        };

        assert_eq!(
            local_openai_direct_config_for_model(&keys, &LLMId::from("gpt-4o"), true),
            None
        );
    }

    #[test]
    fn local_direct_config_preserves_legacy_openai_triple_fallback() {
        let keys = ai::api_keys::ApiKeys {
            openai: Some("legacy-local-key".to_string()),
            openai_base_url: Some("http://localhost:11434/v1".to_string()),
            openai_model: Some("legacy-local-model".to_string()),
            ..Default::default()
        };

        let config =
            local_openai_direct_config_for_model(&keys, &LLMId::from("legacy-local-model"), true)
                .expect("expected legacy local direct config");

        assert_eq!(config.api_key, "legacy-local-key");
        assert_eq!(config.base_url, "http://localhost:11434/v1");
        assert_eq!(config.model, "legacy-local-model");
        assert!(config.vision_supported);
    }
}
