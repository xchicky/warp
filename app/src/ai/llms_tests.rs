use super::*;

fn add_local_full_terminal_use_llm_test_models(app: &mut warpui::App) {
    app.add_singleton_model(|_| crate::auth::AuthStateProvider::new_for_test());
    app.add_singleton_model(|_| crate::server::server_api::ServerApiProvider::new_for_test());
    app.add_singleton_model(crate::auth::auth_manager::AuthManager::new_for_test);
    app.add_singleton_model(crate::server::sync_queue::SyncQueue::mock);
    app.add_singleton_model(|_| crate::network::NetworkStatus::new());
    app.add_singleton_model(crate::workspaces::team_tester::TeamTesterStatus::mock);
    app.add_singleton_model(crate::server::cloud_objects::update_manager::UpdateManager::mock);
    app.add_singleton_model(crate::cloud_object::model::persistence::CloudModel::mock);
    app.add_singleton_model(|_| {
        crate::ai::mcp::templatable_manager::TemplatableMCPServerManager::default()
    });
    app.add_singleton_model(crate::settings::PrivacySettings::mock);
    app.add_singleton_model(crate::workspaces::user_workspaces::UserWorkspaces::default_mock);
    app.add_singleton_model(|ctx| {
        crate::ai::execution_profiles::profiles::AIExecutionProfilesModel::new(
            &crate::LaunchMode::new_for_unit_test(),
            ctx,
        )
    });
    app.add_singleton_model(LLMPreferences::new);
}

// -- DisableReason::should_clear_preference tests --

#[test]
fn should_clear_preference_admin_disabled() {
    // AdminDisabled always clears, regardless of BYOK status.
    assert!(DisableReason::AdminDisabled.should_clear_preference(false));
    assert!(DisableReason::AdminDisabled.should_clear_preference(true));
}

#[test]
fn should_clear_preference_unavailable() {
    assert!(DisableReason::Unavailable.should_clear_preference(false));
    assert!(DisableReason::Unavailable.should_clear_preference(true));
}

#[test]
fn should_not_clear_preference_out_of_requests() {
    // Transient — never clears.
    assert!(!DisableReason::OutOfRequests.should_clear_preference(false));
    assert!(!DisableReason::OutOfRequests.should_clear_preference(true));
}

#[test]
fn should_not_clear_preference_provider_outage() {
    // Transient — never clears.
    assert!(!DisableReason::ProviderOutage.should_clear_preference(false));
    assert!(!DisableReason::ProviderOutage.should_clear_preference(true));
}

#[test]
fn should_clear_preference_requires_upgrade_without_byok() {
    // No BYOK key → server will reject → clear.
    assert!(DisableReason::RequiresUpgrade.should_clear_preference(false));
}

#[test]
fn should_not_clear_preference_requires_upgrade_with_byok() {
    // BYOK key present → server allows → keep.
    assert!(!DisableReason::RequiresUpgrade.should_clear_preference(true));
}

#[test]
fn llm_info_deserializes_without_base_model_name() {
    let raw = r#"{
            "display_name": "gpt-4o",
            "id": "gpt-4o",
            "usage_metadata": {
                "request_multiplier": 1,
                "credit_multiplier": null
            },
            "description": null,
            "disable_reason": null,
            "vision_supported": false,
            "spec": null,
            "provider": "Unknown"
        }"#;

    let info: LLMInfo = serde_json::from_str(raw).expect("should deserialize");
    assert_eq!(info.display_name, "gpt-4o");
    assert_eq!(info.base_model_name, "gpt-4o");
}

#[test]
fn llm_info_deserializes_host_configs_as_vec() {
    // Wire format from server: host_configs is a Vec
    let raw = r#"{
            "display_name": "gpt-4o",
            "id": "gpt-4o",
            "usage_metadata": { "request_multiplier": 1, "credit_multiplier": null },
            "provider": "OpenAI",
            "host_configs": [
                { "enabled": true, "model_routing_host": "DirectApi" },
                { "enabled": false, "model_routing_host": "AwsBedrock" }
            ]
        }"#;

    let info: LLMInfo = serde_json::from_str(raw).expect("should deserialize vec format");
    assert_eq!(info.display_name, "gpt-4o");
    assert_eq!(info.host_configs.len(), 2);
    assert!(
        info.host_configs
            .get(&LLMModelHost::DirectApi)
            .unwrap()
            .enabled
    );
    assert!(
        !info
            .host_configs
            .get(&LLMModelHost::AwsBedrock)
            .unwrap()
            .enabled
    );
}

#[test]
fn llm_info_round_trip_serializes_and_deserializes() {
    // Start with wire format (Vec)
    let wire_json = r#"{
            "display_name": "claude-3",
            "base_model_name": "claude-3",
            "id": "claude-3",
            "usage_metadata": { "request_multiplier": 2, "credit_multiplier": 1.5 },
            "description": "A powerful model",
            "vision_supported": true,
            "provider": "Anthropic",
            "host_configs": [
                { "enabled": true, "model_routing_host": "DirectApi" }
            ]
        }"#;

    // Deserialize from wire format
    let info: LLMInfo = serde_json::from_str(wire_json).expect("should deserialize");

    // Serialize (produces HashMap format)
    let serialized = serde_json::to_string(&info).expect("should serialize");

    // Deserialize again (from HashMap format)
    let round_tripped: LLMInfo =
        serde_json::from_str(&serialized).expect("should deserialize after round trip");

    assert_eq!(info, round_tripped);
}

#[test]
fn add_custom_models_makes_local_openai_the_default_agent_model() {
    let mut models = ModelsByFeature::default();
    let api_keys = ai::api_keys::ApiKeys {
        local_openai_api_key: Some("local-key".to_string()),
        local_openai_base_url: Some("http://localhost:11434/v1".to_string()),
        local_openai_model: Some("qwen2.5-coder".to_string()),
        ..Default::default()
    };

    add_custom_models(&mut models, &api_keys);

    let default = models.agent_mode.default_llm_info();
    assert_eq!(default.id, local_openai_llm_id("qwen2.5-coder"));
    assert_eq!(default.display_name, "Local OpenAI: qwen2.5-coder");
}

#[test]
fn add_custom_models_allows_hosted_openai_key_to_coexist_with_local_openai() {
    let mut models = ModelsByFeature::default();
    let api_keys = ai::api_keys::ApiKeys {
        openai: Some("hosted-openai-key".to_string()),
        local_openai_api_key: Some("local-key".to_string()),
        local_openai_base_url: Some("http://localhost:11434/v1".to_string()),
        local_openai_model: Some("qwen2.5-coder".to_string()),
        ..Default::default()
    };

    add_custom_models(&mut models, &api_keys);

    let default = models.agent_mode.default_llm_info();
    assert_eq!(default.id, local_openai_llm_id("qwen2.5-coder"));
    assert_eq!(default.display_name, "Local OpenAI: qwen2.5-coder");
}

#[test]
fn add_custom_models_preserves_legacy_local_openai_fallback() {
    let mut models = ModelsByFeature::default();
    let api_keys = ai::api_keys::ApiKeys {
        openai: Some("legacy-local-key".to_string()),
        openai_base_url: Some("http://localhost:11434/v1".to_string()),
        openai_model: Some("legacy-local-model".to_string()),
        ..Default::default()
    };

    add_custom_models(&mut models, &api_keys);

    let default = models.agent_mode.default_llm_info();
    assert_eq!(default.id, local_openai_llm_id("legacy-local-model"));
    assert_eq!(default.display_name, "Local OpenAI: legacy-local-model");
}

#[test]
fn add_custom_models_namespaces_local_openai_id_when_model_matches_hosted_id() {
    let mut models = ModelsByFeature::default();
    models.agent_mode.choices.push(custom_llm_info(
        "Hosted OpenAI",
        "gpt-4o",
        LLMProvider::OpenAI,
    ));
    let api_keys = ai::api_keys::ApiKeys {
        local_openai_api_key: Some("local-key".to_string()),
        local_openai_base_url: Some("http://localhost:11434/v1".to_string()),
        local_openai_model: Some("gpt-4o".to_string()),
        ..Default::default()
    };

    add_custom_models(&mut models, &api_keys);

    let default = models.agent_mode.default_llm_info();
    assert_eq!(default.id, local_openai_llm_id("gpt-4o"));
    assert_eq!(default.display_name, "Local OpenAI: gpt-4o");
    assert!(models
        .agent_mode
        .choices
        .iter()
        .any(|choice| choice.id == LLMId::from("gpt-4o")));
    assert!(models
        .agent_mode
        .choices
        .iter()
        .any(|choice| choice.id == local_openai_llm_id("gpt-4o")));
}

#[test]
fn local_full_terminal_use_choice_is_flag_gated() {
    warpui::App::test((), |mut app| async move {
        crate::test_util::settings::initialize_settings_for_tests(&mut app);
        add_local_full_terminal_use_llm_test_models(&mut app);

        ApiKeyManager::handle(&app).update(&mut app, |manager, ctx| {
            manager.set_local_openai_api_key(Some("local-key".to_string()), ctx);
            manager.set_local_openai_base_url(Some("http://localhost:11434/v1".to_string()), ctx);
            manager.set_local_openai_model(Some("qwen2.5-coder".to_string()), ctx);
        });

        let _flag = FeatureFlag::LocalAgentFullTerminalUse.override_enabled(false);
        let choice = LLMPreferences::handle(&app).read(&app, |preferences, ctx| {
            preferences.local_full_terminal_use_choice(ctx)
        });
        assert!(choice.is_none());
    });
}

#[test]
fn local_full_terminal_use_choice_defaults_and_preserves_external_choices() {
    warpui::App::test((), |mut app| async move {
        crate::test_util::settings::initialize_settings_for_tests(&mut app);
        add_local_full_terminal_use_llm_test_models(&mut app);
        let _flag = FeatureFlag::LocalAgentFullTerminalUse.override_enabled(true);

        ApiKeyManager::handle(&app).update(&mut app, |manager, ctx| {
            manager.set_local_openai_api_key(Some("local-key".to_string()), ctx);
            manager.set_local_openai_base_url(Some("http://localhost:11434/v1".to_string()), ctx);
            manager.set_local_openai_model(Some("qwen2.5-coder".to_string()), ctx);
        });

        let (active, choices) = LLMPreferences::handle(&app).read(&app, |preferences, ctx| {
            (
                preferences.get_active_cli_agent_model_with_local(ctx, None),
                preferences
                    .get_cli_agent_llm_choices_with_local(ctx)
                    .collect::<Vec<_>>(),
            )
        });
        assert_eq!(active.id, local_openai_llm_id("qwen2.5-coder"));
        assert_eq!(active.display_name, "Local Agent (qwen2.5-coder)");

        assert_eq!(choices[0].id, local_openai_llm_id("qwen2.5-coder"));
        assert!(choices
            .iter()
            .skip(1)
            .any(|choice| choice.id == LLMId::from("cli-agent-auto")));
    });
}

#[test]
fn local_full_terminal_use_keeps_explicit_external_cli_agent_model() {
    warpui::App::test((), |mut app| async move {
        crate::test_util::settings::initialize_settings_for_tests(&mut app);
        add_local_full_terminal_use_llm_test_models(&mut app);
        let _flag = FeatureFlag::LocalAgentFullTerminalUse.override_enabled(true);

        ApiKeyManager::handle(&app).update(&mut app, |manager, ctx| {
            manager.set_local_openai_api_key(Some("local-key".to_string()), ctx);
            manager.set_local_openai_base_url(Some("http://localhost:11434/v1".to_string()), ctx);
            manager.set_local_openai_model(Some("qwen2.5-coder".to_string()), ctx);
        });
        crate::ai::execution_profiles::profiles::AIExecutionProfilesModel::handle(&app).update(
            &mut app,
            |profiles, ctx| {
                let profile_id = profiles.default_profile_id();
                profiles.set_cli_agent_model(profile_id, Some(LLMId::from("cli-agent-auto")), ctx);
            },
        );

        let active = LLMPreferences::handle(&app).read(&app, |preferences, ctx| {
            preferences.get_active_cli_agent_model_with_local(ctx, None)
        });

        assert_eq!(active.id, LLMId::from("cli-agent-auto"));
    });
}

#[test]
fn local_full_terminal_use_choice_is_available_as_llm_info() {
    warpui::App::test((), |mut app| async move {
        crate::test_util::settings::initialize_settings_for_tests(&mut app);
        add_local_full_terminal_use_llm_test_models(&mut app);
        let _flag = FeatureFlag::LocalAgentFullTerminalUse.override_enabled(true);

        ApiKeyManager::handle(&app).update(&mut app, |manager, ctx| {
            manager.set_local_openai_api_key(Some("local-key".to_string()), ctx);
            manager.set_local_openai_base_url(Some("http://localhost:11434/v1".to_string()), ctx);
            manager.set_local_openai_model(Some("qwen2.5-coder".to_string()), ctx);
        });

        let info = LLMPreferences::handle(&app)
            .read(&app, |preferences, ctx| {
                preferences.get_llm_info_for_request(&local_openai_llm_id("qwen2.5-coder"), ctx)
            })
            .expect("local FTU model should be visible to request metadata");

        assert_eq!(info.display_name, "Local Agent (qwen2.5-coder)");
    });
}
