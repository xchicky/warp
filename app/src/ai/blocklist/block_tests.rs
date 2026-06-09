use super::{
    populate_local_direct_completed_file_edit_diffs, received_message_collapsible_id,
    CollapsibleElementState, CollapsibleExpansionState, LocalDirectCompletedFileEditDiffsContext,
};
use crate::ai::agent::conversation::AIConversationId;
use crate::ai::agent::task::TaskId;
use crate::ai::agent::{
    AIAgentAction, AIAgentActionId, AIAgentActionResult, AIAgentActionResultType,
    AIAgentActionType, AIAgentOutput, AIAgentOutputMessage, AIAgentOutputMessageType,
    AIIdentifiers, MessageId, RequestFileEditsResult, StartAgentExecutionMode,
};
use crate::ai::blocklist::action_model::{
    compose_run_agents_child_prompt, run_agents_to_start_agent_mode, BlocklistAIActionModel,
    RequestFileEditsFormatKind,
};
use crate::ai::blocklist::agent_view::{AgentViewController, EphemeralMessageModel};
use crate::ai::blocklist::block::model::testing::FakeAIBlockModel;
use crate::ai::blocklist::context_model::BlocklistAIContextModel;
use crate::ai::blocklist::inline_action::code_diff_view::{
    CodeDiffState, CodeDiffView, CodeDiffViewAction,
};
use crate::ai::blocklist::input_model::BlocklistAIInputModel;
use crate::ai::blocklist::{BlocklistAIController, BlocklistAIHistoryModel, ClientIdentifiers};
use crate::ai::get_relevant_files::controller::GetRelevantFilesController;
use crate::settings::AISettings;
use crate::terminal::find::model::TerminalFindModel;
use crate::terminal::model::session::active_session::ActiveSession;
use crate::terminal::model::session::Sessions;
use crate::terminal::model::terminal_model::TerminalModel;
use crate::terminal::model_events::ModelEventDispatcher;
use crate::test_util::settings::initialize_settings_for_tests;
use crate::test_util::terminal::{add_window_with_terminal, initialize_app_for_terminal_view};
use crate::FileEdit;
use crate::NotebookKeybindings;
use ai::agent::action::{RunAgentsAgentRunConfig, RunAgentsExecutionMode};
use ai::skills::SkillReference;
use parking_lot::FairMutex;
use settings::Setting;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use warp_files::FileModel;
use warpui::platform::WindowStyle;
use warpui::{App, EntityId, ModelHandle, SingletonEntity, TypedActionView, ViewContext};

#[test]
fn reasoning_auto_collapses_when_user_has_not_manually_toggled() {
    App::test((), |mut app| async move {
        initialize_settings_for_tests(&mut app);
        let mut state = CollapsibleElementState::default();
        app.update(|ctx| {
            state.finish_reasoning(ctx);
        });

        assert!(matches!(
            state.expansion_state,
            CollapsibleExpansionState::Collapsed
        ));
    });
}

#[test]
fn always_show_thinking_stays_expanded_after_finish() {
    App::test((), |mut app| async move {
        initialize_settings_for_tests(&mut app);
        AISettings::handle(&app).update(&mut app, |settings, ctx| {
            settings
                .thinking_display_mode
                .set_value(crate::settings::ThinkingDisplayMode::AlwaysShow, ctx)
                .unwrap();
        });

        let mut state = CollapsibleElementState::default();
        app.update(|ctx| {
            state.finish_reasoning(ctx);
        });

        assert!(matches!(
            state.expansion_state,
            CollapsibleExpansionState::Expanded {
                is_finished: true,
                scroll_pinned_to_bottom: false
            }
        ));
    });
}

#[test]
fn manual_collapse_while_streaming_stays_collapsed_after_finish() {
    App::test((), |mut app| async move {
        initialize_settings_for_tests(&mut app);
        let mut state = CollapsibleElementState::default();

        state.toggle_expansion();
        app.update(|ctx| {
            state.finish_reasoning(ctx);
        });

        assert!(matches!(
            state.expansion_state,
            CollapsibleExpansionState::Collapsed
        ));
    });
}

#[test]
fn manual_reexpand_while_streaming_stays_expanded_after_finish() {
    App::test((), |mut app| async move {
        initialize_settings_for_tests(&mut app);
        let mut state = CollapsibleElementState::default();

        state.toggle_expansion();
        state.toggle_expansion();
        app.update(|ctx| {
            state.finish_reasoning(ctx);
        });

        assert!(matches!(
            state.expansion_state,
            CollapsibleExpansionState::Expanded {
                is_finished: true,
                scroll_pinned_to_bottom: false
            }
        ));
    });
}

#[test]
fn received_message_collapsible_id_prefixes_row_ids() {
    let first = received_message_collapsible_id("message-1");
    let second = received_message_collapsible_id("message-2");

    assert_eq!(&*first, "received-message:message-1");
    assert_eq!(&*second, "received-message:message-2");
    assert_ne!(first, second);
}

#[test]
fn compose_child_prompt_concatenates_when_both_non_empty() {
    let composed = compose_run_agents_child_prompt("base", "do X");
    assert_eq!(composed, "base\n\ndo X");
}

#[test]
fn compose_child_prompt_uses_base_only_when_per_agent_empty() {
    let composed = compose_run_agents_child_prompt("base", "");
    assert_eq!(composed, "base");
}

#[test]
fn compose_child_prompt_uses_per_agent_only_when_base_empty() {
    let composed = compose_run_agents_child_prompt("", "do X");
    assert_eq!(composed, "do X");
}

#[test]
fn compose_child_prompt_returns_empty_when_both_empty() {
    let composed = compose_run_agents_child_prompt("", "");
    assert_eq!(composed, "");
}

#[test]
fn compose_child_prompt_treats_whitespace_only_base_as_empty() {
    let composed = compose_run_agents_child_prompt("   \n", "do X");
    assert_eq!(composed, "do X");
}

fn agent_cfg() -> RunAgentsAgentRunConfig {
    RunAgentsAgentRunConfig {
        name: "child".to_string(),
        prompt: "do X".to_string(),
        title: "Child".to_string(),
    }
}

#[test]
fn remote_arm_propagates_skills_into_skill_references() {
    let skills = vec![
        SkillReference::BundledSkillId("writing-pr-descriptions".to_string()),
        SkillReference::Path(PathBuf::from("/tmp/skill/SKILL.md")),
    ];
    let mode = run_agents_to_start_agent_mode(
        &RunAgentsExecutionMode::Remote {
            environment_id: "env-1".to_string(),
            worker_host: "warp".to_string(),
            computer_use_enabled: true,
        },
        "oz",
        "auto",
        &skills,
        &agent_cfg(),
    )
    .expect("Remote+oz must convert");
    let StartAgentExecutionMode::Remote {
        skill_references,
        environment_id,
        worker_host,
        harness_type,
        model_id,
        computer_use_enabled,
        title,
    } = mode
    else {
        panic!("expected Remote start-agent mode");
    };
    assert_eq!(skill_references, skills);
    assert_eq!(environment_id, "env-1");
    assert_eq!(worker_host, "warp");
    assert_eq!(harness_type, "oz");
    assert_eq!(model_id, "auto");
    assert!(computer_use_enabled);
    assert_eq!(title, "Child");
}

#[test]
fn remote_arm_with_empty_skills_propagates_empty_vec() {
    let mode = run_agents_to_start_agent_mode(
        &RunAgentsExecutionMode::Remote {
            environment_id: "env-1".to_string(),
            worker_host: "warp".to_string(),
            computer_use_enabled: false,
        },
        "claude",
        "auto",
        &[],
        &agent_cfg(),
    )
    .expect("Remote+claude must convert");
    let StartAgentExecutionMode::Remote {
        skill_references, ..
    } = mode
    else {
        panic!("expected Remote start-agent mode");
    };
    assert!(skill_references.is_empty());
}

#[test]
fn remote_arm_rejects_opencode() {
    let err = run_agents_to_start_agent_mode(
        &RunAgentsExecutionMode::Remote {
            environment_id: "env-1".to_string(),
            worker_host: "warp".to_string(),
            computer_use_enabled: false,
        },
        "opencode",
        "auto",
        &[],
        &agent_cfg(),
    )
    .expect_err("Remote+opencode must be rejected");
    assert!(err.to_lowercase().contains("opencode"));
}

fn create_action_model(app: &mut App) -> ModelHandle<BlocklistAIActionModel> {
    let terminal_model = Arc::new(FairMutex::new(TerminalModel::mock(None, None)));
    let sessions = app.add_model(|_| Sessions::new_for_test());
    let (_, model_events_rx) = async_channel::unbounded();
    let model_event_dispatcher =
        app.add_model(|ctx| ModelEventDispatcher::new(model_events_rx, sessions.clone(), ctx));
    let active_session =
        app.add_model(|ctx| ActiveSession::new(sessions, model_event_dispatcher.clone(), ctx));
    let get_relevant_files_controller = app.add_model(GetRelevantFilesController::new);
    app.add_model(|ctx| {
        BlocklistAIActionModel::new(
            terminal_model,
            active_session,
            &model_event_dispatcher,
            get_relevant_files_controller,
            EntityId::new(),
            ctx,
        )
    })
}

fn create_file_edit() -> FileEdit {
    FileEdit::Create {
        file: Some("created.txt".to_string()),
        content: Some("contents".to_string()),
    }
}

fn request_file_edits_action(action_id: &AIAgentActionId) -> AIAgentAction {
    AIAgentAction {
        id: action_id.clone(),
        task_id: TaskId::new("task".to_string()),
        action: AIAgentActionType::RequestFileEdits {
            file_edits: vec![create_file_edit()],
            title: Some("Create file".to_string()),
        },
        requires_result: true,
    }
}

fn successful_file_edit_result(action_id: AIAgentActionId) -> AIAgentActionResult {
    AIAgentActionResult {
        id: action_id,
        task_id: TaskId::new("task".to_string()),
        result: AIAgentActionResultType::RequestFileEdits(RequestFileEditsResult::Success {
            diff: String::new(),
            updated_files: vec![],
            deleted_files: vec![],
            lines_added: 1,
            lines_removed: 0,
        }),
    }
}

fn code_diff_view_for_test(
    action_id: AIAgentActionId,
    conversation_id: AIConversationId,
    action_model: ModelHandle<BlocklistAIActionModel>,
    ctx: &mut ViewContext<CodeDiffView>,
) -> CodeDiffView {
    let mut output = AIAgentOutput::default();
    output.messages.push(AIAgentOutputMessage {
        id: MessageId::new("action-message".to_string()),
        message: AIAgentOutputMessageType::Action(request_file_edits_action(&action_id)),
        citations: vec![],
    });
    let model = FakeAIBlockModel::new(vec![], output);
    CodeDiffView::new(
        &action_id,
        &model,
        Some("Create file".to_string()),
        AIIdentifiers {
            client_conversation_id: Some(conversation_id),
            ..Default::default()
        },
        RequestFileEditsFormatKind::Unknown,
        false,
        action_model,
        None,
        ctx,
    )
}

fn ai_block_for_test(
    conversation_id: AIConversationId,
    action_model: ModelHandle<BlocklistAIActionModel>,
    terminal_view_handle: warpui::ViewHandle<crate::terminal::view::TerminalView>,
    ctx: &mut ViewContext<super::AIBlock>,
) -> super::AIBlock {
    let terminal_model = Arc::new(FairMutex::new(TerminalModel::mock(None, None)));
    let terminal_view_id = EntityId::new();
    let sessions = ctx.add_model(|_| Sessions::new_for_test());
    let (_, model_events_rx) = async_channel::unbounded();
    let model_event_dispatcher =
        ctx.add_model(|ctx| ModelEventDispatcher::new(model_events_rx, sessions.clone(), ctx));
    let active_session = ctx
        .add_model(|ctx| ActiveSession::new(sessions.clone(), model_event_dispatcher.clone(), ctx));
    let ephemeral_message_model = ctx.add_model(|_| EphemeralMessageModel::new());
    let agent_view_controller = ctx.add_model(|_| {
        AgentViewController::new(
            terminal_model.clone(),
            terminal_view_id,
            ephemeral_message_model.clone(),
        )
    });
    let context_model = ctx.add_model(|ctx| {
        BlocklistAIContextModel::new(
            sessions,
            &model_event_dispatcher,
            terminal_model.clone(),
            terminal_view_id,
            agent_view_controller.clone(),
            ctx,
        )
    });
    let input_model = ctx.add_model(|ctx| {
        BlocklistAIInputModel::new(
            terminal_model.clone(),
            agent_view_controller.clone(),
            context_model.clone(),
            terminal_view_id,
            ctx,
        )
    });
    let controller = ctx.add_model(|ctx| {
        BlocklistAIController::new(
            input_model,
            context_model.clone(),
            action_model.clone(),
            active_session.clone(),
            agent_view_controller.clone(),
            terminal_model.clone(),
            terminal_view_id,
            ctx,
        )
    });
    let cli_subagent_controller = ctx.add_model(|ctx| {
        super::cli_controller::CLISubagentController::new(
            &controller,
            &action_model,
            Some(agent_view_controller.clone()),
            terminal_model.clone(),
            &model_event_dispatcher,
            terminal_view_id,
            ctx,
        )
    });
    let get_relevant_files_controller = ctx.add_model(GetRelevantFilesController::new);
    let find_model = ctx.add_model(|_| TerminalFindModel::new(terminal_model.clone()));
    let block_model = Rc::new(FakeAIBlockModel::new(vec![], AIAgentOutput::default()));

    super::AIBlock::new(
        block_model,
        terminal_model,
        ClientIdentifiers {
            client_exchange_id: Default::default(),
            conversation_id,
            response_stream_id: None,
        },
        controller,
        get_relevant_files_controller,
        None,
        None,
        action_model,
        context_model,
        find_model,
        active_session,
        &cli_subagent_controller,
        &model_event_dispatcher,
        agent_view_controller,
        None,
        terminal_view_handle.downgrade(),
        terminal_view_id,
        ctx,
    )
}

#[test]
fn local_direct_result_first_completed_file_edit_populates_diffs_and_expands() {
    App::test((), |mut app| async move {
        initialize_app_for_terminal_view(&mut app);
        app.add_singleton_model(NotebookKeybindings::new);
        app.add_singleton_model(FileModel::new);
        let action_model = create_action_model(&mut app);
        let action_id = AIAgentActionId::from("apply-file-diff-1".to_string());

        let terminal_view = add_window_with_terminal(&mut app, None);
        let terminal_view_handle = terminal_view.clone();
        let terminal_view_id = terminal_view.read(&app, |view, _| view.id());
        let conversation_id = BlocklistAIHistoryModel::handle(&app)
            .update(&mut app, |history, ctx| {
                history.start_new_conversation(terminal_view_id, false, false, ctx)
            });
        action_model.update(&mut app, |action_model, ctx| {
            action_model.apply_finished_local_direct_file_edit_result(
                conversation_id,
                successful_file_edit_result(action_id.clone()),
                ctx,
            );
        });

        let (_, ai_block) = app.add_window(WindowStyle::NotStealFocus, |ctx| {
            ai_block_for_test(
                conversation_id,
                action_model.clone(),
                terminal_view_handle,
                ctx,
            )
        });

        let code_diff_view = ai_block.update(&mut app, |block, ctx| {
            block.handle_requested_edit_complete_for_test(&action_id, vec![create_file_edit()], ctx)
        });

        app.read(|ctx| {
            code_diff_view.read(ctx, |view, _| {
                assert!(matches!(
                    view.state_for_test(),
                    CodeDiffState::Accepted(None)
                ));
                assert!(!view.is_pending_diffs_empty());
                assert!(!view.is_expanded());
            });
        });

        code_diff_view.update(&mut app, |view, ctx| {
            view.handle_action(&CodeDiffViewAction::ToggleRequestedEditVisibility, ctx);
        });

        app.read(|ctx| {
            code_diff_view.read(ctx, |view, _| {
                assert!(!view.is_pending_diffs_empty());
                assert!(view.is_expanded());
            });
        });
    });
}

#[test]
fn local_direct_result_first_completed_file_edit_population_is_conversation_scoped() {
    App::test((), |mut app| async move {
        initialize_app_for_terminal_view(&mut app);
        app.add_singleton_model(NotebookKeybindings::new);
        app.add_singleton_model(FileModel::new);
        let action_model = create_action_model(&mut app);
        let action_id = AIAgentActionId::from("apply-file-diff-1".to_string());
        let conversation_a = AIConversationId::new();
        let conversation_b = AIConversationId::new();

        action_model.update(&mut app, |action_model, ctx| {
            action_model.apply_finished_local_direct_file_edit_result(
                conversation_a,
                successful_file_edit_result(action_id.clone()),
                ctx,
            );
        });

        let (_, code_diff_view) = app.add_window(WindowStyle::NotStealFocus, |ctx| {
            code_diff_view_for_test(action_id.clone(), conversation_b, action_model.clone(), ctx)
        });
        let shell_launch_data = None;
        let current_working_directory = Some("/tmp".to_string());

        populate_local_direct_completed_file_edit_diffs(
            &action_model,
            conversation_b,
            &action_id,
            &code_diff_view,
            vec![create_file_edit()],
            LocalDirectCompletedFileEditDiffsContext {
                shell_launch_data: &shell_launch_data,
                current_working_directory: &current_working_directory,
            },
            &mut app,
        );

        app.read(|ctx| {
            code_diff_view.read(ctx, |view, _| {
                assert!(matches!(
                    view.state_for_test(),
                    CodeDiffState::WaitingForUser
                ));
                assert!(view.is_pending_diffs_empty());
            });
        });
    });
}
