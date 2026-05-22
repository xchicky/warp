use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LocalAgentCostTelemetryConfig {
    #[serde(default)]
    pub pricing_overrides: HashMap<String, LocalAgentModelPricing>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LocalAgentModelPricing {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_cents_per_million: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_cents_per_million: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_cents_per_million: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_cents_per_million: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LocalAgentUsageCounts {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub prompt_cache_read_tokens: u32,
    pub prompt_cache_write_tokens: u32,
}

impl LocalAgentUsageCounts {
    pub fn new(
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
        prompt_cache_read_tokens: u32,
        prompt_cache_write_tokens: u32,
    ) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens,
            prompt_cache_read_tokens,
            prompt_cache_write_tokens,
        }
    }
}

#[derive(Clone, Debug)]
struct BuiltinPricingRule {
    model_pattern: &'static str,
    effective_date: &'static str,
    pricing: LocalAgentModelPricing,
}

const BUILTIN_PRICING_RULES: &[BuiltinPricingRule] = &[
    // Effective 2026-05-22: common OpenAI-compatible local defaults used by the
    // local-agent image/cost telemetry tests.
    BuiltinPricingRule {
        model_pattern: "gpt-4o",
        effective_date: "2026-05-22",
        pricing: LocalAgentModelPricing {
            input_cents_per_million: Some(250.0),
            output_cents_per_million: Some(1_000.0),
            cache_read_cents_per_million: Some(25.0),
            cache_write_cents_per_million: Some(250.0),
        },
    },
    // Effective 2026-05-22: DeepSeek-compatible local deployments often use
    // this family during testing.
    BuiltinPricingRule {
        model_pattern: "deepseek-chat",
        effective_date: "2026-05-22",
        pricing: LocalAgentModelPricing {
            input_cents_per_million: Some(27.5),
            output_cents_per_million: Some(110.0),
            cache_read_cents_per_million: Some(2.75),
            cache_write_cents_per_million: Some(27.5),
        },
    },
    // Effective 2026-05-22: Qwen-compatible local deployments.
    BuiltinPricingRule {
        model_pattern: "qwen-max",
        effective_date: "2026-05-22",
        pricing: LocalAgentModelPricing {
            input_cents_per_million: Some(30.0),
            output_cents_per_million: Some(120.0),
            cache_read_cents_per_million: Some(3.0),
            cache_write_cents_per_million: Some(30.0),
        },
    },
];

pub fn estimate_local_request_cost_cents(
    model_id: &str,
    usage: LocalAgentUsageCounts,
    config: &LocalAgentCostTelemetryConfig,
) -> Option<f64> {
    let pricing = config
        .pricing_overrides
        .get(model_id)
        .or_else(|| builtin_pricing_for_model(model_id))?;

    let mut total_cost = 0.0;
    let mut saw_any_pricing = false;

    if let Some(input_price) = pricing.input_cents_per_million {
        total_cost += cents_for_tokens(usage.prompt_tokens, input_price);
        saw_any_pricing = true;
    }
    if let Some(output_price) = pricing.output_cents_per_million {
        total_cost += cents_for_tokens(usage.completion_tokens, output_price);
        saw_any_pricing = true;
    }
    if let Some(cache_read_price) = pricing.cache_read_cents_per_million {
        total_cost += cents_for_tokens(usage.prompt_cache_read_tokens, cache_read_price);
        saw_any_pricing = true;
    }
    if let Some(cache_write_price) = pricing.cache_write_cents_per_million {
        total_cost += cents_for_tokens(usage.prompt_cache_write_tokens, cache_write_price);
        saw_any_pricing = true;
    }

    saw_any_pricing.then_some(total_cost)
}

fn cents_for_tokens(tokens: u32, cents_per_million: f64) -> f64 {
    (tokens as f64 * cents_per_million) / 1_000_000.0
}

fn builtin_pricing_for_model(model_id: &str) -> Option<&'static LocalAgentModelPricing> {
    BUILTIN_PRICING_RULES
        .iter()
        .find(|rule| model_matches_pattern(model_id, rule.model_pattern))
        .map(|rule| {
            let _ = rule.effective_date;
            &rule.pricing
        })
}

fn model_matches_pattern(model_id: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return model_id.starts_with(prefix);
    }
    model_id == pattern
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_known_model_cost_and_cache_tokens() {
        let usage = LocalAgentUsageCounts::new(2_000_000, 1_000_000, 3_000_000, 100_000, 50_000);
        let cost = estimate_local_request_cost_cents(
            "gpt-4o",
            usage,
            &LocalAgentCostTelemetryConfig::default(),
        )
        .unwrap();

        assert!(cost > 0.0);
    }

    #[test]
    fn prefers_user_override_and_returns_none_without_any_pricing() {
        let mut config = LocalAgentCostTelemetryConfig::default();
        config.pricing_overrides.insert(
            "custom-model".to_string(),
            LocalAgentModelPricing {
                input_cents_per_million: Some(10.0),
                output_cents_per_million: None,
                cache_read_cents_per_million: None,
                cache_write_cents_per_million: None,
            },
        );

        assert_eq!(
            estimate_local_request_cost_cents(
                "custom-model",
                LocalAgentUsageCounts::new(1_000_000, 0, 1_000_000, 0, 0),
                &config
            ),
            Some(10.0)
        );

        assert_eq!(
            estimate_local_request_cost_cents(
                "unknown-model",
                LocalAgentUsageCounts::new(1_000_000, 0, 1_000_000, 0, 0),
                &config
            ),
            None
        );
    }
}
