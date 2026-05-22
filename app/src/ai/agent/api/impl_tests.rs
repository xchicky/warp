use crate::ai::agent::{
    api::{RequestParams, ServerConversationToken},
    local::LocalDirectConfig,
    task::TaskId,
    AIAgentActionId, AIAgentActionResult, AIAgentActionResultType, AIAgentContext, AIAgentInput,
    MCPContext, MCPServer, ReadFilesResult,
};
use crate::ai::blocklist::SessionContext;
use crate::ai::llms::LLMId;
use warp_core::features::FeatureFlag;
use warp_multi_agent_api as api;

use super::{get_supported_tools, local_direct_config_for_request, local_mcp_context_for_request};

fn request_params_with_ask_user_question_enabled(ask_user_question_enabled: bool) -> RequestParams {
    let model = LLMId::from("test-model");

    RequestParams {
        input: vec![],
        conversation_token: None,
        forked_from_conversation_token: None,
        ambient_agent_task_id: None,
        tasks: vec![],
        existing_suggestions: None,
        metadata: None,
        session_context: SessionContext::new_for_test(),
        model: model.clone(),
        coding_model: model.clone(),
        cli_agent_model: model.clone(),
        computer_use_model: model,
        is_memory_enabled: false,
        warp_drive_context_enabled: false,
        context_window_limit: None,
        mcp_context: None,
        planning_enabled: true,
        should_redact_secrets: false,
        api_keys: None,
        allow_use_of_warp_credits_with_byok: false,
        local_direct_config: None,
        autonomy_level: api::AutonomyLevel::Supervised,
        isolation_level: api::IsolationLevel::None,
        web_search_enabled: false,
        computer_use_enabled: false,
        ask_user_question_enabled,
        research_agent_enabled: false,
        orchestration_enabled: false,
        supported_tools_override: None,
        parent_agent_id: None,
        agent_name: None,
    }
}

fn local_direct_config() -> LocalDirectConfig {
    LocalDirectConfig {
        api_key: "test-key".to_string(),
        base_url: "http://127.0.0.1:1/v1".to_string(),
        model: "test-model".to_string(),
        vision_supported: true,
        cost_telemetry: Default::default(),
    }
}

fn action_result_input() -> AIAgentInput {
    AIAgentInput::ActionResult {
        result: AIAgentActionResult {
            id: AIAgentActionId::from("action".to_string()),
            task_id: TaskId::new("task".to_string()),
            result: AIAgentActionResultType::ReadFiles(ReadFilesResult::Cancelled),
        },
        context: std::sync::Arc::from(Vec::<AIAgentContext>::new().into_boxed_slice()),
    }
}

fn resume_conversation_input() -> AIAgentInput {
    AIAgentInput::ResumeConversation {
        context: std::sync::Arc::from(Vec::<AIAgentContext>::new().into_boxed_slice()),
    }
}

fn mcp_context_with_server(name: &str) -> MCPContext {
    #[allow(deprecated)]
    MCPContext {
        resources: Vec::new(),
        tools: Vec::new(),
        servers: vec![MCPServer {
            id: format!("{name}-id"),
            name: name.to_string(),
            description: "test server".to_string(),
            resources: Vec::new(),
            tools: Vec::new(),
        }],
    }
}

#[test]
fn local_route_continues_for_non_user_query_with_token() {
    let mut params = request_params_with_ask_user_question_enabled(false);
    let local_config = local_direct_config();
    params.local_direct_config = Some(local_config.clone());
    params.conversation_token = Some(ServerConversationToken::new(
        "local-conversation".to_string(),
    ));
    params.input = vec![action_result_input()];

    assert_eq!(local_direct_config_for_request(&params), Some(local_config));
}

#[test]
fn local_mcp_context_is_flag_gated() {
    let _flag = FeatureFlag::LocalAgentMcp.override_enabled(false);

    assert_eq!(
        local_mcp_context_for_request(Some(mcp_context_with_server("server"))),
        None
    );
}

#[test]
fn local_mcp_context_uses_existing_grouped_server_context_when_enabled() {
    let _flag = FeatureFlag::LocalAgentMcp.override_enabled(true);

    let context = local_mcp_context_for_request(Some(mcp_context_with_server("server"))).unwrap();

    assert_eq!(context.servers().len(), 1);
    assert_eq!(context.servers()[0].name(), "server");
}

#[test]
fn local_mcp_context_is_empty_without_active_servers_or_cwd_scoped_context() {
    let _flag = FeatureFlag::LocalAgentMcp.override_enabled(true);
    #[allow(deprecated)]
    let empty_context = MCPContext {
        resources: Vec::new(),
        tools: Vec::new(),
        servers: Vec::new(),
    };

    assert_eq!(local_mcp_context_for_request(None), None);
    assert_eq!(local_mcp_context_for_request(Some(empty_context)), None);
}

#[test]
fn local_route_skipped_for_non_user_query_without_token() {
    let mut params = request_params_with_ask_user_question_enabled(false);
    params.local_direct_config = Some(local_direct_config());
    params.conversation_token = None;
    params.input = vec![resume_conversation_input()];

    assert_eq!(local_direct_config_for_request(&params), None);
}

#[test]
fn supported_tools_omits_ask_user_question_when_disabled() {
    let params = request_params_with_ask_user_question_enabled(false);
    let supported_tools = get_supported_tools(&params);

    assert!(!supported_tools.contains(&api::ToolType::AskUserQuestion));
}

#[test]
fn supported_tools_includes_ask_user_question_when_enabled_and_feature_flag_is_enabled() {
    if !FeatureFlag::AskUserQuestion.is_enabled() {
        return;
    }

    let params = request_params_with_ask_user_question_enabled(true);
    let supported_tools = get_supported_tools(&params);

    assert!(supported_tools.contains(&api::ToolType::AskUserQuestion));
}

#[test]
fn supported_tools_include_upload_artifact_when_feature_flag_is_enabled() {
    let _flag = FeatureFlag::ArtifactCommand.override_enabled(true);
    let params = request_params_with_ask_user_question_enabled(false);
    let supported_tools = get_supported_tools(&params);

    assert!(supported_tools.contains(&api::ToolType::UploadFileArtifact));
}

#[test]
fn supported_tools_omit_upload_artifact_when_feature_flag_is_disabled() {
    let _flag = FeatureFlag::ArtifactCommand.override_enabled(false);
    let params = request_params_with_ask_user_question_enabled(false);
    let supported_tools = get_supported_tools(&params);

    assert!(!supported_tools.contains(&api::ToolType::UploadFileArtifact));
}
