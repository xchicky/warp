//! This module contains model, controller, and view logic for Blocklist AI.
use warpui::SingletonEntity;

mod action_model;
pub mod agent_view;
pub mod block;
pub mod code_block;
mod context_model;
mod controller;
#[cfg(feature = "local_fs")]
pub(crate) mod handoff;

pub(crate) fn is_local_to_cloud_handoff_available() -> bool {
    use crate::features::FeatureFlag;
    FeatureFlag::OzHandoff.is_enabled()
        && FeatureFlag::HandoffLocalCloud.is_enabled()
        && cfg!(all(feature = "local_fs", not(target_family = "wasm")))
}

pub(crate) fn conversation_latest_exchange_local_openai_model_id(
    history_model: &history_model::BlocklistAIHistoryModel,
    conversation_id: &crate::ai::agent::conversation::AIConversationId,
) -> Option<crate::ai::llms::LLMId> {
    let model_id = history_model
        .conversation(conversation_id)?
        .latest_exchange()?
        .model_id
        .clone();
    crate::ai::llms::local_openai_model_from_llm_id(&model_id)
        .is_some()
        .then_some(model_id)
}

pub(crate) fn selected_local_full_terminal_use_model_id(
    app: &warpui::AppContext,
    terminal_view_id: warpui::EntityId,
) -> Option<crate::ai::llms::LLMId> {
    if !crate::features::FeatureFlag::LocalAgentFullTerminalUse.is_enabled() {
        return None;
    }

    let model = crate::ai::llms::LLMPreferences::as_ref(app)
        .get_active_cli_agent_model_with_local(app, Some(terminal_view_id));
    crate::ai::llms::local_openai_model_from_llm_id(&model.id)
        .is_some()
        .then_some(model.id)
}

pub(crate) fn current_rendered_conversation_local_openai_model_id(
    terminal_model: &crate::terminal::TerminalModel,
    maybe_agent_view_controller: Option<&agent_view::AgentViewController>,
    history_model: &history_model::BlocklistAIHistoryModel,
    selected_local_model: Option<&crate::ai::llms::LLMId>,
) -> Option<crate::ai::llms::LLMId> {
    let agent_view_state = maybe_agent_view_controller
        .map(|controller| controller.agent_view_state())
        .unwrap_or_else(|| terminal_model.block_list().agent_view_state());

    if agent_view_state.is_active() {
        let conversation_id = agent_view_state
            .fullscreen_conversation_id()
            .or_else(|| agent_view_state.active_conversation_id())?;
        let conversation = history_model.conversation(&conversation_id)?;
        if conversation.latest_exchange().is_none() {
            return selected_local_model.and_then(|model_id| {
                crate::ai::llms::local_openai_model_from_llm_id(model_id)
                    .is_some()
                    .then_some(model_id.clone())
            });
        }

        return conversation_latest_exchange_local_openai_model_id(history_model, &conversation_id);
    }

    let conversation_id = terminal_model
        .block_list()
        .active_block()
        .ai_conversation_id()?;
    conversation_latest_exchange_local_openai_model_id(history_model, &conversation_id)
}
pub(crate) mod orchestration_event_streamer;
pub(crate) mod orchestration_events;
mod passive_suggestions;
pub(crate) mod task_status_sync_model;
pub(super) use controller::RequestInput;
pub mod history_model;
pub mod inline_action;
mod input_model;
mod permissions;
mod persistence;
pub mod prompt;
pub mod suggested_agent_mode_workflow_modal;
pub mod suggested_rule_modal;
mod suggestion_chip_view;
pub mod summarization_cancel_dialog;
pub(crate) mod telemetry;
pub mod usage;

pub(crate) mod codebase_index_speedbump_banner;
pub(crate) mod telemetry_banner;
pub(super) mod view_util;

#[cfg_attr(target_family = "wasm", allow(unused_imports))]
pub(crate) use action_model::{
    apply_edits, read_local_file_context, BlocklistAIActionEvent, BlocklistAIActionModel,
    FileReadResult, ReadFileContextResult, RequestFileEditsFormatKind, ShellCommandExecutor,
    ShellCommandExecutorEvent, StartAgentExecutor, StartAgentExecutorEvent, StartAgentRequest,
    StartAgentRequestId,
};

#[cfg(any(test, feature = "integration_tests"))]
pub(crate) use block::model::testing::FakeAIBlockModel;
pub(crate) use block::{init, model, AIBlock, AIBlockEvent, RequestedEditResolution};

pub(crate) use context_model::{
    block_context_from_terminal_model, AttachmentType, BlocklistAIContextEvent,
    BlocklistAIContextModel, PendingAttachment, PendingFile, PendingQueryState,
};
pub(crate) use controller::{
    response_stream::ResponseStreamId, BlocklistAIController, BlocklistAIControllerEvent,
    ClientIdentifiers, SessionContext, SlashCommandRequest,
};
pub(crate) use history_model::{
    AIQueryHistory, AIQueryHistoryOutputStatus, BlocklistAIHistoryEvent, BlocklistAIHistoryModel,
    ConversationStatusUpdate, FORK_PREFIX, PRE_REWIND_PREFIX,
};
pub(crate) use input_model::{
    BlocklistAIInputEvent, BlocklistAIInputModel, InputConfig, InputType,
};
pub(crate) use passive_suggestions::{
    LegacyPassiveSuggestionsEvent, LegacyPassiveSuggestionsModel, MaaPassiveSuggestionsEvent,
    MaaPassiveSuggestionsModel, PassiveSuggestionsModels,
};
#[cfg_attr(target_family = "wasm", allow(unused))]
pub(crate) use persistence::PersistedAIInputType;
pub(crate) use persistence::{PersistedAIInput, SerializedBlockListItem};
pub(crate) use view_util::{
    ai_brand_color, ai_indicator_height, get_ai_block_overflow_menu_element_position_id,
    get_attached_blocks_chip_element_position_id, render_ai_agent_mode_icon,
    render_ai_follow_up_icon, ATTACH_AS_AGENT_MODE_CONTEXT_TEXT, CLAUDE_ORANGE,
    NEW_AGENT_PANE_LABEL,
};

pub(crate) use view_util::format_credits;

pub use crate::ai::blocklist::block::{secret_redaction, AIBlockResponseRating, TextLocation};
pub use block::keyboard_navigable_buttons;
pub use block::toggleable_items;
pub use controller::input_context::{
    BLOCK_CONTEXT_ATTACHMENT_REGEX, DIFF_HUNK_ATTACHMENT_REGEX, DRIVE_OBJECT_ATTACHMENT_REGEX,
};
pub use permissions::{BlocklistAIPermissions, CommandExecutionPermissionAllowedReason};
pub use suggestion_chip_view::*;
pub use view_util::error_color;
