use std::collections::HashMap;

use super::{
    artifact_from_fork_proto, AIConversation, AIConversationAutoexecuteMode, AIConversationId,
    RequestCost,
};
use crate::ai::artifacts::Artifact;
use crate::persistence::model::{AgentConversationData, PRIMARY_AGENT_CATEGORY};
use warp_core::features::FeatureFlag;
use warp_multi_agent_api as api;

fn restored_conversation(conversation_data: Option<AgentConversationData>) -> AIConversation {
    AIConversation::new_restored(
        AIConversationId::new(),
        vec![api::Task {
            id: "root-task".to_string(),
            messages: vec![],
            dependencies: None,
            description: String::new(),
            summary: String::new(),
            server_data: String::new(),
        }],
        conversation_data,
    )
    .unwrap()
}

fn user_query_message(id: &str, request_id: &str, query: &str) -> api::Message {
    api::Message {
        id: id.to_string(),
        task_id: "root-task".to_string(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::UserQuery(api::message::UserQuery {
            query: query.to_string(),
            context: None,
            referenced_attachments: HashMap::new(),
            mode: None,
            intended_agent: Default::default(),
        })),
        request_id: request_id.to_string(),
        timestamp: None,
    }
}

fn agent_output_message(id: &str, request_id: &str) -> api::Message {
    api::Message {
        id: id.to_string(),
        task_id: "root-task".to_string(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::AgentOutput(
            api::message::AgentOutput {
                text: "Done".to_string(),
            },
        )),
        request_id: request_id.to_string(),
        timestamp: None,
    }
}

fn local_byok_token_usage(
    model_id: &str,
    total_input: u32,
    output: u32,
    cost_in_cents: f32,
) -> api::response_event::stream_finished::TokenUsage {
    api::response_event::stream_finished::TokenUsage {
        model_id: model_id.to_string(),
        total_input,
        output,
        input_cache_read: 0,
        input_cache_write: 0,
        cost_in_cents,
    }
}

#[allow(deprecated)]
fn local_byok_usage_metadata(
    model_id: &str,
    total_tokens: u32,
    cost_in_cents: f32,
) -> api::response_event::stream_finished::ConversationUsageMetadata {
    let mut usage_by_category = HashMap::new();
    usage_by_category.insert(PRIMARY_AGENT_CATEGORY.to_string(), total_tokens);

    let mut byok_token_usage = HashMap::new();
    byok_token_usage.insert(
        model_id.to_string(),
        api::response_event::stream_finished::ModelTokenUsage {
            model_id: model_id.to_string(),
            total_tokens,
            token_usage_by_category: usage_by_category,
        },
    );

    api::response_event::stream_finished::ConversationUsageMetadata {
        context_window_usage: 0.0,
        summarized: false,
        credits_spent: cost_in_cents,
        token_usage: vec![],
        tool_usage_metadata: None,
        warp_token_usage: HashMap::new(),
        byok_token_usage,
    }
}

fn restored_conversation_with_queries(queries: &[&str]) -> AIConversation {
    let messages = queries
        .iter()
        .enumerate()
        .flat_map(|(index, query)| {
            let request_id = format!("request-{index}");
            [
                user_query_message(&format!("user-{index}"), &request_id, query),
                agent_output_message(&format!("agent-{index}"), &request_id),
            ]
        })
        .collect();

    AIConversation::new_restored(
        AIConversationId::new(),
        vec![api::Task {
            id: "root-task".to_string(),
            messages,
            dependencies: None,
            description: String::new(),
            summary: String::new(),
            server_data: String::new(),
        }],
        None,
    )
    .unwrap()
}

#[test]
fn local_byok_usage_metadata_accumulates_across_requests() {
    let mut conversation = AIConversation::new(false);

    conversation
        .update_cost_and_usage_for_request(
            Some(RequestCost::new(0.25)),
            vec![local_byok_token_usage("local-model", 10, 5, 0.25)],
            Some(local_byok_usage_metadata("local-model", 15, 0.25)),
            true,
        )
        .unwrap();
    conversation
        .update_cost_and_usage_for_request(
            Some(RequestCost::new(0.5)),
            vec![local_byok_token_usage("local-model", 20, 10, 0.5)],
            Some(local_byok_usage_metadata("local-model", 30, 0.5)),
            true,
        )
        .unwrap();

    let usage_metadata = conversation.usage_metadata();
    assert_eq!(usage_metadata.credits_spent, 0.75);
    assert_eq!(conversation.credits_spent_for_last_block(), Some(0.5));
    assert_eq!(usage_metadata.token_usage.len(), 1);
    assert_eq!(usage_metadata.token_usage[0].model_id, "local-model");
    assert_eq!(usage_metadata.token_usage[0].byok_tokens, 45);
    assert_eq!(
        usage_metadata.token_usage[0].byok_token_usage_by_category[PRIMARY_AGENT_CATEGORY],
        45
    );
}

#[test]
fn latest_user_query_returns_latest_non_empty_user_query() {
    let conversation =
        restored_conversation_with_queries(&["write unit tests", "fix the failing test"]);

    assert_eq!(
        conversation.latest_user_query(),
        Some("fix the failing test".to_string())
    );
}

#[test]
fn latest_user_query_trims_and_skips_empty_queries() {
    let conversation = restored_conversation_with_queries(&["  write unit tests  ", "  "]);

    assert_eq!(
        conversation.latest_user_query(),
        Some("write unit tests".to_string())
    );
}

#[test]
fn restored_conversation_defaults_autoexecute_override_when_not_persisted() {
    let _flag = FeatureFlag::RememberFastForwardState.override_enabled(true);
    let conversation_data: AgentConversationData =
        serde_json::from_str(r#"{"server_conversation_token":null}"#).unwrap();

    let conversation = restored_conversation(Some(conversation_data));

    assert_eq!(
        conversation.autoexecute_override(),
        AIConversationAutoexecuteMode::RespectUserSettings
    );
}

#[test]
fn restored_conversation_uses_persisted_last_event_sequence() {
    let conversation_data: AgentConversationData =
        serde_json::from_str(r#"{"server_conversation_token":null,"last_event_sequence":42}"#)
            .unwrap();

    let conversation = restored_conversation(Some(conversation_data));

    assert_eq!(conversation.last_event_sequence(), Some(42));
}

#[test]
fn restored_conversation_uses_persisted_remote_child_marker() {
    let conversation_data: AgentConversationData =
        serde_json::from_str(r#"{"server_conversation_token":null,"is_remote_child":true}"#)
            .unwrap();

    let conversation = restored_conversation(Some(conversation_data));

    assert!(conversation.is_remote_child());
}

#[test]
fn child_conversation_detection_uses_parent_agent_id() {
    let conversation_data: AgentConversationData = serde_json::from_str(
        r#"{"server_conversation_token":null,"parent_agent_id":"parent-run-id"}"#,
    )
    .unwrap();

    let conversation = restored_conversation(Some(conversation_data));

    assert!(conversation.is_child_agent_conversation());
    assert_eq!(conversation.parent_conversation_id(), None);
}

#[test]
fn restored_conversation_defaults_unknown_persisted_autoexecute_override() {
    let _flag = FeatureFlag::RememberFastForwardState.override_enabled(true);
    let conversation_data: AgentConversationData = serde_json::from_str(
        r#"{"server_conversation_token":null,"autoexecute_override":"UnexpectedValue"}"#,
    )
    .unwrap();

    let conversation = restored_conversation(Some(conversation_data));

    assert_eq!(
        conversation.autoexecute_override(),
        AIConversationAutoexecuteMode::RespectUserSettings
    );
}

#[test]
fn restored_conversation_uses_persisted_autoexecute_override_when_enabled() {
    let _flag = FeatureFlag::RememberFastForwardState.override_enabled(true);
    let conversation_data: AgentConversationData = serde_json::from_str(
        r#"{"server_conversation_token":null,"autoexecute_override":"RunToCompletion"}"#,
    )
    .unwrap();

    let conversation = restored_conversation(Some(conversation_data));

    assert_eq!(
        conversation.autoexecute_override(),
        AIConversationAutoexecuteMode::RunToCompletion
    );
}

#[test]
fn restored_conversation_ignores_persisted_autoexecute_override_when_disabled() {
    let _flag = FeatureFlag::RememberFastForwardState.override_enabled(false);
    let conversation_data: AgentConversationData = serde_json::from_str(
        r#"{"server_conversation_token":null,"autoexecute_override":"RunToCompletion"}"#,
    )
    .unwrap();

    let conversation = restored_conversation(Some(conversation_data));

    assert_eq!(
        conversation.autoexecute_override(),
        AIConversationAutoexecuteMode::RespectUserSettings
    );
}

#[test]
fn fork_artifacts_adds_file_artifacts_to_conversation() {
    let proto_artifact = api::message::artifact_event::ConversationArtifact {
        artifact: Some(
            api::message::artifact_event::conversation_artifact::Artifact::File(
                api::message::artifact_event::FileArtifact {
                    artifact_uid: "artifact-file-1".to_string(),
                    filepath: "outputs/report.txt".to_string(),
                    mime_type: "text/plain".to_string(),
                    size_bytes: 42,
                    description: "Daily summary".to_string(),
                },
            ),
        ),
    };

    assert_eq!(
        artifact_from_fork_proto(&proto_artifact),
        Some(Artifact::File {
            artifact_uid: "artifact-file-1".to_string(),
            filepath: "outputs/report.txt".to_string(),
            filename: "report.txt".to_string(),
            mime_type: "text/plain".to_string(),
            description: Some("Daily summary".to_string()),
            size_bytes: Some(42),
        })
    );
}
