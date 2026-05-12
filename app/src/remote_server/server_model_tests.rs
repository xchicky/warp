use crate::auth::auth_state::AuthState;
use ::ai::index::full_source_code_embedding::manager::{
    CodebaseIndexFinishedStatus, CodebaseIndexingError,
};
use ::ai::index::full_source_code_embedding::SyncProgress;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use warp_util::standardized_path::StandardizedPath;

use super::super::proto::{Authenticate, CodebaseIndexStatusState, Initialize};
use super::super::server_buffer_tracker::ServerBufferTracker;
use super::{
    codebase_index_status_state_from_parts, failure_message_from_last_sync_result,
    progress_from_sync_progress, PendingFileOps, ServerModel,
};

fn test_model() -> ServerModel {
    ServerModel {
        connection_senders: HashMap::new(),
        snapshot_sent_roots_by_connection: HashMap::new(),
        authorized_codebase_index_roots_by_connection: HashMap::new(),
        grace_timer_cancel: None,
        in_progress: HashMap::new(),
        host_id: "test-host-id".to_string(),
        executors: HashMap::new(),
        pending_file_ops: PendingFileOps::new(),
        auth_state: Arc::new(AuthState::new_logged_out_for_test()),
        buffers: ServerBufferTracker::new(),
    }
}

#[test]
fn codebase_index_root_authorization_is_connection_scoped() {
    let mut model = test_model();
    let authorized_conn = uuid::Uuid::new_v4();
    let unauthorized_conn = uuid::Uuid::new_v4();
    let repo_path = StandardizedPath::from_local_canonicalized(Path::new("/")).unwrap();

    model
        .authorized_codebase_index_roots_by_connection
        .insert(authorized_conn, HashSet::new());
    model
        .authorized_codebase_index_roots_by_connection
        .insert(unauthorized_conn, HashSet::new());
    model.authorize_codebase_index_root_for_connection(authorized_conn, repo_path.clone());

    assert!(model.codebase_index_root_authorized_for_connection(authorized_conn, &repo_path));
    assert!(!model.codebase_index_root_authorized_for_connection(unauthorized_conn, &repo_path));
}

#[test]
fn removing_authorized_codebase_index_root_clears_all_connections() {
    let mut model = test_model();
    let first_conn = uuid::Uuid::new_v4();
    let second_conn = uuid::Uuid::new_v4();
    let repo_path = StandardizedPath::from_local_canonicalized(Path::new("/")).unwrap();

    model.authorize_codebase_index_root_for_connection(first_conn, repo_path.clone());
    model.authorize_codebase_index_root_for_connection(second_conn, repo_path.clone());
    model.remove_authorized_codebase_index_root(&repo_path);

    assert!(!model.codebase_index_root_authorized_for_connection(first_conn, &repo_path));
    assert!(!model.codebase_index_root_authorized_for_connection(second_conn, &repo_path));
}

#[test]
fn fresh_model_starts_without_auth_token() {
    let model = test_model();

    assert_eq!(model.auth_token().as_deref(), None);
    assert_eq!(model.auth_state.user_id(), None);
    assert_eq!(model.auth_state.user_email(), None);
}

#[test]
fn initialize_with_auth_token_stores_token() {
    let mut model = test_model();

    model.apply_initialize_auth(&Initialize {
        auth_token: "initial-token".to_string(),
        user_id: "test-user-id".to_string(),
        user_email: "test@example.com".to_string(),
        crash_reporting_enabled: true,
    });

    assert_eq!(model.auth_token().as_deref(), Some("initial-token"));
    assert_eq!(
        model.auth_state.user_id().unwrap().as_string(),
        "test-user-id"
    );
    assert_eq!(
        model.auth_state.user_email().as_deref(),
        Some("test@example.com")
    );
}

#[test]
fn empty_initialize_clears_auth_context() {
    let mut model = test_model();
    model.apply_initialize_auth(&Initialize {
        auth_token: "initial-token".to_string(),
        user_id: "test-user-id".to_string(),
        user_email: "test@example.com".to_string(),
        crash_reporting_enabled: true,
    });

    model.apply_initialize_auth(&Initialize {
        auth_token: String::new(),
        user_id: String::new(),
        user_email: String::new(),
        crash_reporting_enabled: true,
    });

    assert_eq!(model.auth_token().as_deref(), None);
    assert_eq!(model.auth_state.user_id(), None);
    assert_eq!(model.auth_state.user_email(), None);
}

#[test]
fn authenticate_with_auth_token_replaces_auth_token() {
    let mut model = test_model();
    model.apply_initialize_auth(&Initialize {
        auth_token: "initial-token".to_string(),
        user_id: String::new(),
        user_email: String::new(),
        crash_reporting_enabled: true,
    });

    model.handle_authenticate(Authenticate {
        auth_token: "rotated-token".to_string(),
    });

    assert_eq!(model.auth_token().as_deref(), Some("rotated-token"));
}

#[test]
fn empty_authenticate_clears_auth_token() {
    let mut model = test_model();
    model.apply_initialize_auth(&Initialize {
        auth_token: "initial-token".to_string(),
        user_id: String::new(),
        user_email: String::new(),
        crash_reporting_enabled: true,
    });

    model.handle_authenticate(Authenticate {
        auth_token: String::new(),
    });

    assert_eq!(model.auth_token().as_deref(), None);
}

#[test]
fn pending_codebase_index_without_synced_version_maps_to_indexing() {
    assert_eq!(
        codebase_index_status_state_from_parts(true, false, None),
        CodebaseIndexStatusState::Indexing
    );
}

#[test]
fn pending_codebase_index_with_synced_version_maps_to_stale() {
    assert_eq!(
        codebase_index_status_state_from_parts(true, true, None),
        CodebaseIndexStatusState::Stale
    );
}

#[test]
fn completed_codebase_index_maps_to_ready() {
    let result = CodebaseIndexFinishedStatus::Completed;

    assert_eq!(
        codebase_index_status_state_from_parts(false, true, Some(&result)),
        CodebaseIndexStatusState::Ready
    );
}

#[test]
fn failed_codebase_index_maps_to_failed_and_includes_message() {
    let result = CodebaseIndexFinishedStatus::Failed(CodebaseIndexingError::BuildTreeError);

    assert_eq!(
        codebase_index_status_state_from_parts(false, false, Some(&result)),
        CodebaseIndexStatusState::Failed
    );
    assert_eq!(
        failure_message_from_last_sync_result(Some(&result)).as_deref(),
        Some("Build tree error")
    );
}

#[test]
fn sync_progress_maps_to_remote_progress_fields() {
    assert_eq!(
        progress_from_sync_progress(Some(&SyncProgress::Discovering { total_nodes: 5 })),
        (Some(0), Some(5))
    );
    assert_eq!(
        progress_from_sync_progress(Some(&SyncProgress::Syncing {
            completed_nodes: 3,
            total_nodes: 8,
        })),
        (Some(3), Some(8))
    );
    assert_eq!(progress_from_sync_progress(None), (None, None));
}
