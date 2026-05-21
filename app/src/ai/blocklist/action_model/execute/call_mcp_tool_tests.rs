//! Unit tests for the `coerce_integer_args` helper.

use super::*;
use serde_json::json;
use std::time::Duration;

fn obj(value: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    match value {
        serde_json::Value::Object(m) => m,
        _ => panic!("expected a JSON object"),
    }
}

#[test]
fn whole_float_is_coerced_when_schema_declares_integer() {
    let mut args = obj(json!({ "line": 5.0 }));
    let schema = obj(json!({
        "properties": { "line": { "type": "integer" } }
    }));

    coerce_integer_args(&mut args, &schema);

    // Serialized as "5", not "5.0", and round-trips as i64.
    assert_eq!(serde_json::to_string(&args["line"]).unwrap(), "5");
    assert_eq!(args["line"].as_i64(), Some(5));
}

#[test]
fn no_coercion_when_not_typed_as_integer() {
    // Three scenarios that should all preserve the original float value:
    //   * schema declares `"type": "number"` (explicit float)
    //   * schema has no `properties` at all
    //   * schema property lacks a `"type"` key
    let cases = [
        json!({ "properties": { "x": { "type": "number" } } }),
        json!({}),
        json!({ "properties": { "x": { "description": "no type" } } }),
    ];

    for schema_value in cases {
        let mut args = obj(json!({ "x": 1.0 }));
        let schema = obj(schema_value);

        coerce_integer_args(&mut args, &schema);

        assert_eq!(args["x"].as_f64(), Some(1.0));
        assert_eq!(serde_json::to_string(&args["x"]).unwrap(), "1.0");
    }
}

#[test]
fn timeout_error_is_shaped_as_timeout_status() {
    let result = call_mcp_tool_error_result(&rmcp::ServiceError::Timeout {
        timeout: Duration::from_secs(3),
    });

    let CallMCPToolResult::Timeout(message) = result else {
        panic!("expected timeout result");
    };
    assert!(message.contains("status: timeout"));
    assert!(message.contains("3 seconds"));
}

#[test]
fn transport_closed_after_retry_is_shaped_as_unavailable() {
    let result = call_mcp_tool_error_result(&rmcp::ServiceError::TransportClosed);

    let CallMCPToolResult::Unavailable(message) = result else {
        panic!("expected unavailable result");
    };
    assert!(message.contains("status: unavailable"));
    assert!(message.contains("one reconnect retry"));
}

#[test]
fn reconnect_failure_is_shaped_as_unavailable() {
    let result = call_mcp_tool_error_result(&rmcp::ServiceError::McpError(
        rmcp::model::ErrorData::internal_error("Reconnection failed: server exited", None),
    ));

    let CallMCPToolResult::Unavailable(message) = result else {
        panic!("expected unavailable result");
    };
    assert!(message.contains("status: unavailable"));
    assert!(message.contains("Reconnection failed"));
}

#[test]
fn server_error_is_shaped_without_raw_transport_details() {
    let result = call_mcp_tool_error_result(&rmcp::ServiceError::UnexpectedResponse);

    let CallMCPToolResult::ServerError(message) = result else {
        panic!("expected server error result");
    };
    assert!(message.contains("status: server-error"));
    assert!(!message.contains("stderr"));
    assert!(!message.contains("stdout"));
}
