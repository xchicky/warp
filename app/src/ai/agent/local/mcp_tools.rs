use std::collections::HashSet;

use serde_json::{json, Map, Value};

use super::{LocalMcpContext, OpenAIChatTool, OpenAIChatToolFunction};

const MAX_MCP_TOOL_DESCRIPTION_CHARS: usize = 600;
const MAX_MCP_SCHEMA_DIAGNOSTIC_CHARS: usize = 240;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct LocalMcpToolCatalog {
    entries: Vec<LocalMcpToolCatalogEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct LocalMcpToolCatalogEntry {
    pub openai_name: String,
    pub server_id: String,
    pub mcp_tool_name: String,
    pub tool: OpenAIChatTool,
}

impl LocalMcpToolCatalog {
    pub fn into_tool_definitions(self) -> Vec<OpenAIChatTool> {
        self.entries.into_iter().map(|entry| entry.tool).collect()
    }

    #[cfg(test)]
    pub fn entries(&self) -> &[LocalMcpToolCatalogEntry] {
        &self.entries
    }
}

pub(super) fn mcp_tool_catalog(
    context: &LocalMcpContext,
    local_tools: &[OpenAIChatTool],
) -> Option<LocalMcpToolCatalog> {
    let mut used_names = local_tools
        .iter()
        .map(|tool| tool.function.name.to_string())
        .collect::<HashSet<_>>();
    let mut entries = Vec::new();

    for server in context.servers() {
        let server_slug = slugify_mcp_component(server.name(), || {
            format!("server_{}", stable_short_suffix(server.id()))
        });
        for (tool_index, tool) in server.tools().iter().enumerate() {
            let tool_slug = slugify_mcp_component(tool.name(), || format!("tool_{tool_index}"));
            let base_name = format!("mcp__{server_slug}__{tool_slug}");
            let openai_name = unique_mcp_tool_name(&base_name, server.id(), &mut used_names);
            let parameters = match convert_mcp_schema_to_openai_parameters(tool.input_schema()) {
                Ok(parameters) => parameters,
                Err(error) => {
                    log::warn!(
                        "Skipping local MCP tool '{}::{}' because its input schema is unsupported: {}",
                        server.name(),
                        tool.name(),
                        truncate_diagnostic(&error)
                    );
                    continue;
                }
            };
            let description = mcp_tool_description(server.name(), tool.description());
            entries.push(LocalMcpToolCatalogEntry {
                openai_name: openai_name.clone(),
                server_id: server.id().to_string(),
                mcp_tool_name: tool.name().to_string(),
                tool: OpenAIChatTool {
                    r#type: "function",
                    function: OpenAIChatToolFunction {
                        name: openai_name,
                        description,
                        parameters,
                    },
                },
            });
        }
    }

    (!entries.is_empty()).then_some(LocalMcpToolCatalog { entries })
}

pub(super) fn slugify_mcp_component(value: &str, fallback: impl FnOnce() -> String) -> String {
    let mut slug = String::new();
    let mut previous_separator = false;

    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            previous_separator = false;
        } else if !previous_separator && !slug.is_empty() {
            slug.push('_');
            previous_separator = true;
        }
    }

    while slug.ends_with('_') {
        slug.pop();
    }

    if slug.is_empty() {
        fallback()
    } else {
        slug
    }
}

fn unique_mcp_tool_name(
    base_name: &str,
    server_id: &str,
    used_names: &mut HashSet<String>,
) -> String {
    if used_names.insert(base_name.to_string()) {
        return base_name.to_string();
    }

    let suffix = stable_short_suffix(server_id);
    let mut candidate = format!("{base_name}__{suffix}");
    let mut counter = 2usize;
    while !used_names.insert(candidate.clone()) {
        candidate = format!("{base_name}__{suffix}_{counter}");
        counter += 1;
    }
    candidate
}

fn mcp_tool_description(server_name: &str, tool_description: Option<&str>) -> String {
    let server_name = sanitize_provider_visible_text(server_name);
    let mut description = format!("MCP tool from server `{server_name}`.");
    if let Some(tool_description) = tool_description {
        let tool_description = sanitize_provider_visible_text(tool_description.trim());
        if !tool_description.is_empty() {
            description.push(' ');
            description.push_str(&tool_description);
        }
    }
    truncate_chars(&description, MAX_MCP_TOOL_DESCRIPTION_CHARS)
}

pub(super) fn convert_mcp_schema_to_openai_parameters(
    schema: &Map<String, Value>,
) -> Result<Value, String> {
    if schema.is_empty() {
        return Ok(json!({ "type": "object" }));
    }

    let value = Value::Object(schema.clone());
    let converted = convert_schema_value(&value, true)?;
    match converted {
        Value::Object(mut object) => {
            object.insert("type".to_string(), Value::String("object".to_string()));
            Ok(Value::Object(object))
        }
        _ => Err("root input schema must be a JSON object schema".to_string()),
    }
}

fn convert_schema_value(value: &Value, root: bool) -> Result<Value, String> {
    let Value::Object(object) = value else {
        return Err("schema node must be an object".to_string());
    };

    reject_unsupported_schema_keywords(object)?;

    if root && !is_object_schema(object) {
        return Err("root input schema type must be object".to_string());
    }

    let mut converted = Map::new();
    for key in [
        "type",
        "description",
        "enum",
        "required",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
    ] {
        if let Some(value) = object.get(key) {
            converted.insert(key.to_string(), value.clone());
        }
    }

    if let Some(properties) = object.get("properties") {
        let Value::Object(properties) = properties else {
            return Err("properties must be an object".to_string());
        };
        let mut converted_properties = Map::new();
        for (name, property_schema) in properties {
            converted_properties
                .insert(name.clone(), convert_schema_value(property_schema, false)?);
        }
        converted.insert(
            "properties".to_string(),
            Value::Object(converted_properties),
        );
    }

    if let Some(items) = object.get("items") {
        converted.insert("items".to_string(), convert_schema_value(items, false)?);
    }

    if let Some(additional_properties) = object.get("additionalProperties") {
        let converted_additional_properties = match additional_properties {
            Value::Bool(_) => additional_properties.clone(),
            Value::Object(_) => convert_schema_value(additional_properties, false)?,
            _ => {
                return Err("additionalProperties must be a boolean or schema object".to_string());
            }
        };
        converted.insert(
            "additionalProperties".to_string(),
            converted_additional_properties,
        );
    }

    validate_schema_type(&converted, root)?;
    Ok(Value::Object(converted))
}

fn is_object_schema(object: &Map<String, Value>) -> bool {
    match object.get("type") {
        Some(Value::String(schema_type)) => schema_type == "object",
        None => object.is_empty() || object.contains_key("properties"),
        _ => false,
    }
}

fn validate_schema_type(object: &Map<String, Value>, root: bool) -> Result<(), String> {
    let Some(schema_type) = object.get("type") else {
        return Ok(());
    };
    let Value::String(schema_type) = schema_type else {
        return Err("schema type must be a string".to_string());
    };
    if root && schema_type != "object" {
        return Err("root input schema type must be object".to_string());
    }
    if matches!(
        schema_type.as_str(),
        "object" | "string" | "number" | "integer" | "boolean" | "array" | "null"
    ) {
        Ok(())
    } else {
        Err(format!("unsupported schema type `{schema_type}`"))
    }
}

fn reject_unsupported_schema_keywords(object: &Map<String, Value>) -> Result<(), String> {
    for key in [
        "$ref",
        "oneOf",
        "anyOf",
        "allOf",
        "not",
        "if",
        "then",
        "else",
        "patternProperties",
        "dependentSchemas",
    ] {
        if object.contains_key(key) {
            return Err(format!("unsupported schema keyword `{key}`"));
        }
    }
    Ok(())
}

fn stable_short_suffix(input: &str) -> String {
    let mut hash = 0x811c9dc5u32;
    for byte in input.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x01000193);
    }
    format!("{hash:08x}")
}

fn sanitize_provider_visible_text(text: &str) -> String {
    text.split_whitespace()
        .map(redact_secret_token)
        .collect::<Vec<_>>()
        .join(" ")
}

fn redact_secret_token(token: &str) -> String {
    let lower = token.to_ascii_lowercase();
    let sensitive = [
        "api_key",
        "apikey",
        "token",
        "secret",
        "password",
        "credential",
        "authorization",
        "bearer",
    ]
    .iter()
    .any(|needle| lower.contains(needle));
    if !sensitive {
        return token.to_string();
    }

    for separator in ['=', ':'] {
        if let Some(index) = token.find(separator) {
            return format!("{}{}<redacted>", &token[..index], separator);
        }
    }

    if lower.starts_with("bearer") {
        return "Bearer <redacted>".to_string();
    }

    token.to_string()
}

fn truncate_diagnostic(message: &str) -> String {
    truncate_chars(message, MAX_MCP_SCHEMA_DIAGNOSTIC_CHARS)
}

fn truncate_chars(message: &str, limit: usize) -> String {
    if message.chars().count() <= limit {
        return message.to_string();
    }
    let mut truncated = message.chars().take(limit).collect::<String>();
    truncated.push('…');
    truncated
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Map, Value};

    use super::{convert_mcp_schema_to_openai_parameters, mcp_tool_catalog, slugify_mcp_component};
    use crate::ai::agent::{
        local::{LocalMcpContext, OpenAIChatTool, OpenAIChatToolFunction},
        MCPContext, MCPServer,
    };

    fn schema(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    fn mcp_tool(
        name: &str,
        description: &str,
        input_schema: Map<String, Value>,
    ) -> rmcp::model::Tool {
        rmcp::model::Tool::new(name.to_string(), description.to_string(), input_schema)
    }

    fn mcp_context(servers: Vec<MCPServer>) -> LocalMcpContext {
        #[allow(deprecated)]
        LocalMcpContext::from_mcp_context(MCPContext {
            resources: Vec::new(),
            tools: Vec::new(),
            servers,
        })
        .unwrap()
    }

    fn server(id: &str, name: &str, description: &str, tools: Vec<rmcp::model::Tool>) -> MCPServer {
        MCPServer {
            id: id.to_string(),
            name: name.to_string(),
            description: description.to_string(),
            resources: Vec::new(),
            tools,
        }
    }

    fn local_tool(name: &'static str) -> OpenAIChatTool {
        OpenAIChatTool {
            r#type: "function",
            function: OpenAIChatToolFunction {
                name: name.to_string(),
                description: "local".to_string(),
                parameters: json!({ "type": "object" }),
            },
        }
    }

    #[test]
    fn slugify_mcp_component_uses_lowercase_ascii_and_fallback() {
        assert_eq!(
            slugify_mcp_component("My Filesystem.Server!", || "fallback".to_string()),
            "my_filesystem_server"
        );
        assert_eq!(
            slugify_mcp_component("🚀", || "server_deadbeef".to_string()),
            "server_deadbeef"
        );
    }

    #[test]
    fn mcp_tool_catalog_namespaces_and_disambiguates_duplicates() {
        let context = mcp_context(vec![
            server(
                "server-a",
                "GitHub",
                "",
                vec![mcp_tool(
                    "Search Issues",
                    "Search issues",
                    schema(json!({ "type": "object" })),
                )],
            ),
            server(
                "server-b",
                "GitHub",
                "",
                vec![mcp_tool(
                    "Search Issues",
                    "Search issues",
                    schema(json!({ "type": "object" })),
                )],
            ),
        ]);

        let catalog = mcp_tool_catalog(&context, &[]).unwrap();
        let names = catalog
            .entries()
            .iter()
            .map(|entry| entry.openai_name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(names[0], "mcp__github__search_issues");
        assert!(names[1].starts_with("mcp__github__search_issues__"));
        assert_ne!(names[0], names[1]);
        assert_eq!(catalog.entries()[0].server_id, "server-a");
        assert_eq!(catalog.entries()[0].mcp_tool_name, "Search Issues");
    }

    #[test]
    fn mcp_tool_catalog_prevents_local_tool_name_collision() {
        let context = mcp_context(vec![server(
            "server-a",
            "Local",
            "",
            vec![mcp_tool(
                "Read File",
                "Read files",
                schema(json!({ "type": "object" })),
            )],
        )]);

        let catalog = mcp_tool_catalog(&context, &[local_tool("mcp__local__read_file")]).unwrap();

        assert!(catalog.entries()[0]
            .openai_name
            .starts_with("mcp__local__read_file__"));
    }

    #[test]
    fn convert_mcp_schema_preserves_supported_openai_schema_fields() {
        let converted = convert_mcp_schema_to_openai_parameters(&schema(json!({
            "type": "object",
            "description": "Inputs",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search text",
                    "minLength": 1,
                    "maxLength": 100,
                    "enum": ["a", "b"]
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "minItems": 1,
                    "maxItems": 3
                }
            },
            "required": ["query"],
            "additionalProperties": false
        })))
        .unwrap();

        assert_eq!(converted["type"], "object");
        assert_eq!(converted["description"], "Inputs");
        assert_eq!(converted["required"], json!(["query"]));
        assert_eq!(converted["additionalProperties"], false);
        assert_eq!(converted["properties"]["query"]["minLength"], 1);
        assert_eq!(converted["properties"]["limit"]["maximum"], 10);
        assert_eq!(converted["properties"]["tags"]["items"]["type"], "string");
    }

    #[test]
    fn convert_mcp_schema_treats_empty_schema_as_object() {
        let converted = convert_mcp_schema_to_openai_parameters(&Map::new()).unwrap();

        assert_eq!(converted, json!({ "type": "object" }));
    }

    #[test]
    fn mcp_tool_catalog_skips_unsupported_schema_without_panicking() {
        let context = mcp_context(vec![server(
            "server-a",
            "GitHub",
            "",
            vec![mcp_tool(
                "Search",
                "Search",
                schema(json!({
                    "type": "object",
                    "properties": {
                        "query": { "oneOf": [{ "type": "string" }, { "type": "integer" }] }
                    }
                })),
            )],
        )]);

        assert!(mcp_tool_catalog(&context, &[]).is_none());
    }

    #[test]
    fn mcp_tool_description_omits_server_config_and_redacts_tool_secrets() {
        let context = mcp_context(vec![server(
            "server-a",
            "Payments",
            "raw command: run --api-key=server-secret",
            vec![mcp_tool(
                "Charge",
                "Use api_key=tool-secret and bearer token to charge",
                schema(json!({ "type": "object" })),
            )],
        )]);

        let catalog = mcp_tool_catalog(&context, &[]).unwrap();
        let description = &catalog.entries()[0].tool.function.description;

        assert!(description.contains("Payments"));
        assert!(description.contains("api_key=<redacted>"));
        assert!(!description.contains("server-secret"));
        assert!(!description.contains("tool-secret"));
    }
}
