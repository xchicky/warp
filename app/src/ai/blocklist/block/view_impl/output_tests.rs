use ai::agent::action::UploadArtifactRequest;
use indexmap::IndexMap;
use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::ai::agent::{AIAgentOutput, UploadArtifactResult};
use crate::ai::blocklist::block::keyboard_navigable_buttons::{
    KeyboardNavigableButtonBuilder, KeyboardNavigableButtons,
};
use crate::ai::blocklist::block::model::testing::FakeAIBlockModel;
use crate::ai::blocklist::block::{
    AIBlockStateHandles, AutonomySettingSpeedbump, KeyboardNavigableButtonsKind, RequestedEdit,
};
use crate::ai::blocklist::secret_redaction::SecretRedactionState;
use crate::ai::blocklist::BlocklistAIActionModel;
use crate::ai::get_relevant_files::controller::GetRelevantFilesController;
use crate::settings::ThinkingDisplayMode;
use crate::terminal::model::session::active_session::ActiveSession;
use crate::terminal::model::session::Sessions;
use crate::terminal::model::TerminalModel;
use crate::terminal::model_events::ModelEventDispatcher;
use crate::terminal::shared_session::SharedSessionStatus;
use crate::test_util::terminal::initialize_app_for_terminal_view;
use crate::util::link_detection::DetectedLinksState;
use parking_lot::FairMutex;
use warp_core::ui::appearance::Appearance;
use warpui::elements::MouseStateHandle;
use warpui::platform::WindowStyle;
use warpui::ui_components::button::ButtonVariant;
use warpui::{
    App, AppContext, Element, Entity, EntityId, Presenter, SingletonEntity, TypedActionView, View,
    ViewContext, ViewHandle, WindowInvalidation,
};

use super::{format_upload_artifact_text, Props};

#[test]
fn pending_plan_approval_renders_keyboard_navigable_buttons() {
    App::test((), |mut app| async move {
        initialize_app_for_terminal_view(&mut app);
        let action_model = create_action_model(&mut app);
        let (window_id, root_view) = app.add_window(WindowStyle::NotStealFocus, |ctx| {
            PendingPlanApprovalOutputTestView::new(action_model.clone(), ctx)
        });

        app.read(|ctx| {
            let text_content = root_view
                .as_ref(ctx)
                .render(ctx)
                .debug_text_content()
                .unwrap_or_default();
            assert!(text_content.contains("Review the plan before implementation."));
        });

        let root_view_id = root_view.id();
        let buttons_view_id =
            app.read(|ctx| root_view.read(ctx, |root, _| root.keyboard_navigable_buttons.id()));
        let mut presenter = Presenter::new(window_id);
        let mut updated = HashSet::new();
        updated.insert(root_view_id);
        updated.insert(buttons_view_id);
        app.update(|ctx| {
            presenter.invalidate(
                WindowInvalidation {
                    updated,
                    ..Default::default()
                },
                ctx,
            );
            presenter.build_scene(
                pathfinder_geometry::vector::vec2f(1000., 1000.),
                1.,
                None,
                ctx,
            );
        });

        app.read(|ctx| {
            let rendered_buttons_text = ctx
                .render_view(window_id, buttons_view_id)
                .expect("buttons child view should be renderable")
                .debug_text_content()
                .unwrap_or_default();
            assert!(rendered_buttons_text.contains("Approve plan and continue"));
            assert!(rendered_buttons_text.contains("Revise plan"));

            let descendants = presenter.descendants(root_view_id);
            assert!(
                descendants.contains(&buttons_view_id),
                "pending plan approval buttons should be wired into the real output render tree"
            );
        });
    });
}

struct PendingPlanApprovalOutputTestView {
    model: FakeAIBlockModel,
    state_handles: AIBlockStateHandles,
    action_model: warpui::ModelHandle<BlocklistAIActionModel>,
    detected_links_state: DetectedLinksState,
    secret_redaction_state: SecretRedactionState,
    requested_edits: IndexMap<crate::ai::agent::AIAgentActionId, RequestedEdit>,
    autonomy_setting_speedbump: AutonomySettingSpeedbump,
    suggested_rules: Vec<ViewHandle<crate::ai::blocklist::SuggestionChipView>>,
    suggested_agent_mode_workflow: Option<ViewHandle<crate::ai::blocklist::SuggestionChipView>>,
    manage_rules_button: ViewHandle<crate::view_components::action_button::ActionButton>,
    keyboard_navigable_buttons: ViewHandle<KeyboardNavigableButtons>,
    response_rating: OnceCell<crate::ai::blocklist::AIBlockResponseRating>,
    review_changes_button: ViewHandle<crate::view_components::action_button::ActionButton>,
    open_all_comments_button: ViewHandle<crate::view_components::action_button::ActionButton>,
    dismiss_suggestion_button: ViewHandle<crate::view_components::action_button::ActionButton>,
    disable_rule_suggestions_button:
        ViewHandle<crate::view_components::action_button::ActionButton>,
    shared_session_status: SharedSessionStatus,
}

impl PendingPlanApprovalOutputTestView {
    fn new(
        action_model: warpui::ModelHandle<BlocklistAIActionModel>,
        ctx: &mut ViewContext<Self>,
    ) -> Self {
        Self {
            model: FakeAIBlockModel::new(vec![], AIAgentOutput::default()),
            state_handles: AIBlockStateHandles::default(),
            action_model,
            detected_links_state: DetectedLinksState::default(),
            secret_redaction_state: SecretRedactionState::default(),
            requested_edits: IndexMap::new(),
            autonomy_setting_speedbump: AutonomySettingSpeedbump::default(),
            suggested_rules: vec![],
            suggested_agent_mode_workflow: None,
            manage_rules_button: ctx.add_view(|_| {
                crate::view_components::action_button::ActionButton::new(
                    "Manage rules",
                    crate::view_components::action_button::NakedTheme,
                )
            }),
            keyboard_navigable_buttons: ctx.add_typed_action_view(|_| {
                KeyboardNavigableButtons::new(vec![
                    test_button("Approve plan and continue"),
                    test_button("Revise plan"),
                ])
            }),
            response_rating: OnceCell::new(),
            review_changes_button: ctx.add_view(|_| {
                crate::view_components::action_button::ActionButton::new(
                    "Review changes",
                    crate::view_components::action_button::SecondaryTheme,
                )
            }),
            open_all_comments_button: ctx.add_view(|_| {
                crate::view_components::action_button::ActionButton::new(
                    "Open all in code review",
                    crate::view_components::action_button::SecondaryTheme,
                )
            }),
            dismiss_suggestion_button: ctx.add_view(|_| {
                crate::view_components::action_button::ActionButton::new(
                    "Dismiss",
                    crate::ai::blocklist::block::SuggestionDismissButtonTheme,
                )
            }),
            disable_rule_suggestions_button: ctx.add_view(|_| {
                crate::view_components::action_button::ActionButton::new(
                    "Don't show again",
                    crate::ai::blocklist::block::SuggestionDismissButtonTheme,
                )
            }),
            shared_session_status: SharedSessionStatus::NotShared,
        }
    }
}

impl Entity for PendingPlanApprovalOutputTestView {
    type Event = ();
}

impl View for PendingPlanApprovalOutputTestView {
    fn ui_name() -> &'static str {
        "PendingPlanApprovalOutputTestView"
    }

    fn render(&self, app: &AppContext) -> Box<dyn Element> {
        super::render(
            Props {
                model: &self.model,
                state_handles: &self.state_handles,
                action_buttons: &HashMap::new(),
                view_screenshot_buttons: &HashMap::new(),
                action_model: &self.action_model,
                editor_views: &[],
                current_working_directory: None,
                shell_launch_data: None,
                detected_links_state: &self.detected_links_state,
                secret_redaction_state: &self.secret_redaction_state,
                requested_commands: &HashMap::new(),
                requested_mcp_tools: &HashMap::new(),
                requested_edits: &self.requested_edits,
                unit_test_suggestions: &HashMap::new(),
                todo_list_states: &HashMap::new(),
                collapsible_block_states: &HashMap::new(),
                is_selecting_text: false,
                is_ai_input_enabled: false,
                find_context: None,
                is_references_section_open: false,
                autonomy_setting_speedbump: &self.autonomy_setting_speedbump,
                suggested_rules: &self.suggested_rules,
                suggested_agent_mode_workflow: &self.suggested_agent_mode_workflow,
                manage_rules_button: &self.manage_rules_button,
                keyboard_navigable_buttons: Some(&self.keyboard_navigable_buttons),
                keyboard_navigable_buttons_kind: Some(
                    KeyboardNavigableButtonsKind::PendingPlanApproval,
                ),
                response_rating: &self.response_rating,
                request_refunded_count: None,
                search_codebase_view: &HashMap::new(),
                web_search_views: &HashMap::new(),
                web_fetch_views: &HashMap::new(),
                review_changes_button: &self.review_changes_button,
                open_all_comments_button: &self.open_all_comments_button,
                dismiss_suggestion_button: &self.dismiss_suggestion_button,
                disable_rule_suggestions_button: &self.disable_rule_suggestions_button,
                current_todo_list: None,
                has_accepted_edits: false,
                finish_reason: None,
                is_usage_footer_expanded: false,
                shared_session_status: &self.shared_session_status,
                terminal_view_id: EntityId::new(),
                is_conversation_transcript_viewer: false,
                aws_bedrock_credentials_error_view: None,
                imported_comments: &HashMap::new(),
                run_agents_card_views: &HashMap::new(),
                #[cfg(feature = "local_fs")]
                resolved_code_block_paths: &HashMap::new(),
                #[cfg(feature = "local_fs")]
                resolved_blocklist_image_sources: &Default::default(),
                thinking_display_mode: ThinkingDisplayMode::ShowAndCollapse,
                conversation_has_imported_comments: false,
                ask_user_question_view: None,
                is_cloud_agent_pre_first_exchange: false,
            },
            app,
        )
    }
}

impl TypedActionView for PendingPlanApprovalOutputTestView {
    type Action = ();
}

fn create_action_model(app: &mut App) -> warpui::ModelHandle<BlocklistAIActionModel> {
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

fn test_button(label: &'static str) -> KeyboardNavigableButtonBuilder {
    KeyboardNavigableButtonBuilder::new(
        move |_is_selected, app| {
            Appearance::as_ref(app)
                .ui_builder()
                .button(ButtonVariant::Secondary, MouseStateHandle::default())
                .with_text_label(label.to_string())
        },
        |_ctx| {},
    )
}

#[test]
fn format_upload_artifact_text_includes_request_details() {
    let request = UploadArtifactRequest {
        file_path: "reports/daily.txt".to_string(),
        description: Some("Daily summary".to_string()),
    };

    let text = format_upload_artifact_text(&request, None);

    assert_eq!(
        text,
        "Upload artifact: reports/daily.txt\nDescription: Daily summary"
    );
}

#[test]
fn format_upload_artifact_text_includes_success_summary() {
    let request = UploadArtifactRequest {
        file_path: "reports/daily.txt".to_string(),
        description: Some("Daily summary".to_string()),
    };
    let result = UploadArtifactResult::Success {
        artifact_uid: "artifact-123".to_string(),
        filepath: Some("reports/daily.txt".to_string()),
        mime_type: "text/plain".to_string(),
        description: Some("Daily summary".to_string()),
        size_bytes: 128,
    };

    let text = format_upload_artifact_text(&request, Some(&result));

    assert_eq!(
        text,
        "Upload artifact: reports/daily.txt\nDescription: Daily summary\nStatus: uploaded artifact artifact-123\nUploaded file: reports/daily.txt"
    );
}

#[test]
fn format_upload_artifact_text_includes_terminal_status() {
    let request = UploadArtifactRequest {
        file_path: "reports/daily.txt".to_string(),
        description: None,
    };

    let error_text = format_upload_artifact_text(
        &request,
        Some(&UploadArtifactResult::Error(
            "permission denied".to_string(),
        )),
    );
    assert_eq!(
        error_text,
        "Upload artifact: reports/daily.txt\nStatus: upload failed: permission denied"
    );

    let cancelled_text =
        format_upload_artifact_text(&request, Some(&UploadArtifactResult::Cancelled));
    assert_eq!(cancelled_text, "Upload artifact: reports/daily.txt");
}
