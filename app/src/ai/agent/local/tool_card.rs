use std::collections::BTreeMap;

use chrono::Local;
use serde_json::Value;
use warp_multi_agent_api as api;

use super::{
    apply_file_diff::ApplyFileDiffSummary, mcp_tools::LocalMcpToolCatalogEntry, OpenAIChatToolCall,
};

pub(super) fn structured_tool_card_events(
    task_id: &str,
    request_id: &str,
    tool_call: &OpenAIChatToolCall,
    result: &str,
    apply_file_diff_summary: Option<&ApplyFileDiffSummary>,
) -> Option<Vec<api::ResponseEvent>> {
    let tool = build_tool_call(&tool_call.function.name, &tool_call.function.arguments)?;
    let result = build_tool_call_result(&tool_call.function.name, result, apply_file_diff_summary)?;

    Some(vec![
        add_tool_call_message(task_id, request_id, tool_call, tool),
        add_tool_call_result_message(task_id, request_id, &tool_call.id, result),
    ])
}

pub(super) fn structured_tool_call_event(
    task_id: &str,
    request_id: &str,
    tool_call: &OpenAIChatToolCall,
) -> Option<api::ResponseEvent> {
    let tool = build_tool_call(&tool_call.function.name, &tool_call.function.arguments)?;
    Some(add_tool_call_message(task_id, request_id, tool_call, tool))
}

pub(super) fn structured_mcp_tool_call_event(
    task_id: &str,
    request_id: &str,
    tool_call: &OpenAIChatToolCall,
    mcp_entry: &LocalMcpToolCatalogEntry,
) -> Option<api::ResponseEvent> {
    let tool = build_mcp_tool_call(&tool_call.function.arguments, mcp_entry)?;
    Some(add_tool_call_message(task_id, request_id, tool_call, tool))
}

fn build_tool_call(name: &str, arguments: &str) -> Option<api::message::tool_call::Tool> {
    match name {
        "read_file" => Some(api::message::tool_call::Tool::ReadFiles(
            build_read_files_tool_call(arguments),
        )),
        "grep" => Some(api::message::tool_call::Tool::Grep(build_grep_tool_call(
            arguments,
        ))),
        "glob" => Some(api::message::tool_call::Tool::FileGlobV2(
            build_glob_tool_call(arguments),
        )),
        "list_directory" => Some(api::message::tool_call::Tool::FileGlobV2(
            build_list_directory_tool_call(arguments),
        )),
        "apply_file_diff" => Some(api::message::tool_call::Tool::ApplyFileDiffs(
            build_apply_file_diffs_tool_call(arguments),
        )),
        "write_file" => Some(api::message::tool_call::Tool::ApplyFileDiffs(
            build_write_file_tool_call(arguments),
        )),
        "edit_file" => Some(api::message::tool_call::Tool::ApplyFileDiffs(
            build_edit_file_tool_call(arguments),
        )),
        "run_shell_command" => Some(api::message::tool_call::Tool::RunShellCommand(
            build_run_shell_command_tool_call(arguments),
        )),
        _ => None,
    }
}

fn build_mcp_tool_call(
    arguments: &str,
    mcp_entry: &LocalMcpToolCatalogEntry,
) -> Option<api::message::tool_call::Tool> {
    let args = parse_args(arguments);
    Some(api::message::tool_call::Tool::CallMcpTool(
        api::message::tool_call::CallMcpTool {
            name: mcp_entry.mcp_tool_name.clone(),
            args: Some(serde_json_object_to_prost_struct(args)?),
            server_id: mcp_entry.server_id.clone(),
        },
    ))
}

fn build_tool_call_result(
    name: &str,
    result: &str,
    apply_file_diff_summary: Option<&ApplyFileDiffSummary>,
) -> Option<api::message::tool_call_result::Result> {
    match name {
        "read_file" => Some(api::message::tool_call_result::Result::ReadFiles(
            read_files_result_from_text(result),
        )),
        "grep" => Some(api::message::tool_call_result::Result::Grep(
            grep_result_from_text(result),
        )),
        "glob" | "list_directory" => Some(api::message::tool_call_result::Result::FileGlobV2(
            file_glob_result_from_text(result),
        )),
        "apply_file_diff" => Some(api::message::tool_call_result::Result::ApplyFileDiffs(
            apply_file_diffs_result_from_text(result, apply_file_diff_summary),
        )),
        "write_file" | "edit_file" => Some(api::message::tool_call_result::Result::ApplyFileDiffs(
            apply_file_diffs_result_from_text(result, apply_file_diff_summary),
        )),
        _ => None,
    }
}

fn add_tool_call_message(
    task_id: &str,
    request_id: &str,
    tool_call: &OpenAIChatToolCall,
    tool: api::message::tool_call::Tool,
) -> api::ResponseEvent {
    add_message_to_task(
        task_id,
        api::Message {
            id: format!("{}_call", tool_call.id),
            task_id: task_id.to_string(),
            request_id: request_id.to_string(),
            timestamp: Some(now_timestamp()),
            server_message_data: String::new(),
            citations: vec![],
            message: Some(api::message::Message::ToolCall(api::message::ToolCall {
                tool_call_id: tool_call.id.clone(),
                tool: Some(tool),
            })),
        },
    )
}

fn add_tool_call_result_message(
    task_id: &str,
    request_id: &str,
    tool_call_id: &str,
    result: api::message::tool_call_result::Result,
) -> api::ResponseEvent {
    add_message_to_task(
        task_id,
        api::Message {
            id: format!("{tool_call_id}_result"),
            task_id: task_id.to_string(),
            request_id: request_id.to_string(),
            timestamp: Some(now_timestamp()),
            server_message_data: String::new(),
            citations: vec![],
            message: Some(api::message::Message::ToolCallResult(
                api::message::ToolCallResult {
                    tool_call_id: tool_call_id.to_string(),
                    context: None,
                    result: Some(result),
                },
            )),
        },
    )
}

fn add_message_to_task(task_id: &str, message: api::Message) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::AddMessagesToTask(
                        api::client_action::AddMessagesToTask {
                            task_id: task_id.to_string(),
                            messages: vec![message],
                        },
                    )),
                }],
            },
        )),
    }
}

fn now_timestamp() -> prost_types::Timestamp {
    let now = Local::now();
    prost_types::Timestamp {
        seconds: now.timestamp(),
        nanos: now.timestamp_subsec_nanos() as i32,
    }
}

fn build_read_files_tool_call(arguments: &str) -> api::message::tool_call::ReadFiles {
    let args = parse_args(arguments);
    let name = optional_string_arg(&args, "path").unwrap_or_default();
    let line_ranges = match (
        optional_u32_arg(&args, "start_line"),
        optional_u32_arg(&args, "end_line"),
    ) {
        (Some(start), Some(end)) => vec![api::FileContentLineRange { start, end }],
        _ => Vec::new(),
    };

    api::message::tool_call::ReadFiles {
        files: vec![api::message::tool_call::read_files::File { name, line_ranges }],
    }
}

fn build_grep_tool_call(arguments: &str) -> api::message::tool_call::Grep {
    let args = parse_args(arguments);
    api::message::tool_call::Grep {
        queries: optional_string_arg(&args, "query")
            .map(|query| vec![query])
            .unwrap_or_default(),
        path: optional_string_arg(&args, "path").unwrap_or_else(|| ".".to_string()),
    }
}

fn build_glob_tool_call(arguments: &str) -> api::message::tool_call::FileGlobV2 {
    let args = parse_args(arguments);
    api::message::tool_call::FileGlobV2 {
        patterns: optional_string_arg(&args, "pattern")
            .map(|pattern| vec![pattern])
            .unwrap_or_default(),
        search_dir: optional_string_arg(&args, "path").unwrap_or_else(|| ".".to_string()),
        max_matches: 0,
        max_depth: 0,
        min_depth: 0,
    }
}

fn build_list_directory_tool_call(arguments: &str) -> api::message::tool_call::FileGlobV2 {
    let args = parse_args(arguments);
    api::message::tool_call::FileGlobV2 {
        patterns: vec!["*".to_string()],
        search_dir: optional_string_arg(&args, "path").unwrap_or_else(|| ".".to_string()),
        max_matches: optional_i32_arg(&args, "max_entries")
            .unwrap_or(200)
            .clamp(1, 2000),
        max_depth: optional_i32_arg(&args, "max_depth")
            .unwrap_or(1)
            .clamp(1, 3),
        min_depth: 0,
    }
}

fn build_apply_file_diffs_tool_call(arguments: &str) -> api::message::tool_call::ApplyFileDiffs {
    let args = parse_args(arguments);
    api::message::tool_call::ApplyFileDiffs {
        summary: optional_string_arg(&args, "summary").unwrap_or_default(),
        diffs: Vec::new(),
        new_files: Vec::new(),
        deleted_files: Vec::new(),
        v4a_updates: Vec::new(),
    }
}

fn build_write_file_tool_call(arguments: &str) -> api::message::tool_call::ApplyFileDiffs {
    let args = parse_args(arguments);
    let path = optional_string_arg(&args, "path").unwrap_or_default();
    let content = optional_string_arg(&args, "content").unwrap_or_default();
    api::message::tool_call::ApplyFileDiffs {
        summary: format!("write_file: {path}"),
        diffs: Vec::new(),
        new_files: vec![api::message::tool_call::apply_file_diffs::NewFile {
            file_path: path,
            content,
        }],
        deleted_files: Vec::new(),
        v4a_updates: Vec::new(),
    }
}

fn build_edit_file_tool_call(arguments: &str) -> api::message::tool_call::ApplyFileDiffs {
    let args = parse_args(arguments);
    let path = optional_string_arg(&args, "path").unwrap_or_default();
    api::message::tool_call::ApplyFileDiffs {
        summary: format!("edit_file: {path}"),
        diffs: vec![api::message::tool_call::apply_file_diffs::FileDiff {
            file_path: path,
            search: optional_string_arg(&args, "old_text").unwrap_or_default(),
            replace: optional_string_arg(&args, "new_text").unwrap_or_default(),
        }],
        new_files: Vec::new(),
        deleted_files: Vec::new(),
        v4a_updates: Vec::new(),
    }
}

fn build_run_shell_command_tool_call(arguments: &str) -> api::message::tool_call::RunShellCommand {
    let args = parse_args(arguments);
    let is_read_only = optional_bool_arg(&args, "is_read_only").unwrap_or(false);
    let is_risky = optional_bool_arg(&args, "is_risky").unwrap_or(!is_read_only);
    api::message::tool_call::RunShellCommand {
        command: optional_string_arg(&args, "command").unwrap_or_default(),
        is_read_only,
        uses_pager: optional_bool_arg(&args, "uses_pager").unwrap_or(false),
        citations: Vec::new(),
        is_risky,
        risk_category: if is_read_only {
            api::RiskCategory::ReadOnly as i32
        } else if is_risky {
            api::RiskCategory::Risky as i32
        } else {
            api::RiskCategory::NontrivialLocalChange as i32
        },
        wait_until_complete_value: Some(
            api::message::tool_call::run_shell_command::WaitUntilCompleteValue::WaitUntilComplete(
                true,
            ),
        ),
    }
}

fn read_files_result_from_text(result: &str) -> api::ReadFilesResult {
    if let Some(error) = tool_error(result) {
        return api::ReadFilesResult {
            result: Some(api::read_files_result::Result::Error(
                api::read_files_result::Error { message: error },
            )),
        };
    }

    let (file_path, content) = result
        .strip_prefix("File: ")
        .and_then(|rest| rest.split_once('\n'))
        .map(|(file_path, content)| (file_path.to_string(), content.to_string()))
        .unwrap_or_else(|| (String::new(), result.to_string()));

    api::ReadFilesResult {
        result: Some(api::read_files_result::Result::TextFilesSuccess(
            api::read_files_result::TextFilesSuccess {
                files: vec![api::FileContent {
                    file_path,
                    content,
                    line_range: None,
                }],
            },
        )),
    }
}

fn grep_result_from_text(result: &str) -> api::GrepResult {
    if let Some(error) = tool_error(result) {
        return api::GrepResult {
            result: Some(api::grep_result::Result::Error(api::grep_result::Error {
                message: error,
            })),
        };
    }

    let mut matches_by_file: BTreeMap<
        String,
        Vec<api::grep_result::success::grep_file_match::GrepLineMatch>,
    > = BTreeMap::new();
    if result != "No matches found." {
        for line in result.lines() {
            let Some((file_path, rest)) = line.rsplit_once(':') else {
                continue;
            };
            let Some((file_path, line_number)) = file_path.rsplit_once(':') else {
                continue;
            };
            let Ok(line_number) = line_number.parse::<u32>() else {
                continue;
            };
            let _ = rest;
            matches_by_file
                .entry(file_path.to_string())
                .or_default()
                .push(api::grep_result::success::grep_file_match::GrepLineMatch { line_number });
        }
    }

    api::GrepResult {
        result: Some(api::grep_result::Result::Success(
            api::grep_result::Success {
                matched_files: matches_by_file
                    .into_iter()
                    .map(
                        |(file_path, matched_lines)| api::grep_result::success::GrepFileMatch {
                            file_path,
                            matched_lines,
                        },
                    )
                    .collect(),
            },
        )),
    }
}

fn file_glob_result_from_text(result: &str) -> api::FileGlobV2Result {
    if let Some(error) = tool_error(result) {
        return api::FileGlobV2Result {
            result: Some(api::file_glob_v2_result::Result::Error(
                api::file_glob_v2_result::Error { message: error },
            )),
        };
    }

    let no_results = matches!(result, "No files matched." | "No entries found.");
    api::FileGlobV2Result {
        result: Some(api::file_glob_v2_result::Result::Success(
            api::file_glob_v2_result::Success {
                matched_files: if no_results {
                    Vec::new()
                } else {
                    result
                        .lines()
                        .filter_map(|line| {
                            let file_path = line
                                .strip_prefix("file: ")
                                .or_else(|| line.strip_prefix("dir: "))
                                .unwrap_or(line)
                                .trim();
                            (!file_path.is_empty()).then(|| {
                                api::file_glob_v2_result::success::FileGlobMatch {
                                    file_path: file_path.to_string(),
                                }
                            })
                        })
                        .collect()
                },
                warnings: String::new(),
            },
        )),
    }
}

fn apply_file_diffs_result_from_text(
    result: &str,
    summary: Option<&ApplyFileDiffSummary>,
) -> api::ApplyFileDiffsResult {
    if let Some(error) = tool_error(result) {
        return api::ApplyFileDiffsResult {
            result: Some(api::apply_file_diffs_result::Result::Error(
                api::apply_file_diffs_result::Error { message: error },
            )),
        };
    }

    api::ApplyFileDiffsResult {
        result: Some(api::apply_file_diffs_result::Result::Success({
            #[allow(deprecated)]
            api::apply_file_diffs_result::Success {
                updated_files: Vec::new(),
                updated_files_v2: summary
                    .into_iter()
                    .flat_map(|summary| summary.files.iter())
                    .map(
                        |file| api::apply_file_diffs_result::success::UpdatedFileContent {
                            file: Some(api::FileContent {
                                file_path: file.path.clone(),
                                content: file.content.clone(),
                                line_range: None,
                            }),
                            was_edited_by_user: false,
                        },
                    )
                    .collect(),
                deleted_files: Vec::new(),
            }
        })),
    }
}

fn tool_error(result: &str) -> Option<String> {
    result
        .strip_prefix("Tool error:")
        .map(str::trim)
        .map(str::to_string)
}

fn parse_args(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or(Value::Null)
}

fn serde_json_object_to_prost_struct(value: Value) -> Option<prost_types::Struct> {
    let Value::Object(fields) = value else {
        return None;
    };
    Some(prost_types::Struct {
        fields: fields
            .into_iter()
            .map(|(key, value)| serde_json_to_prost(value).map(|value| (key, value)))
            .collect::<Option<_>>()?,
    })
}

fn serde_json_to_prost(value: Value) -> Option<prost_types::Value> {
    use prost_types::value::Kind;
    Some(prost_types::Value {
        kind: Some(match value {
            Value::Null => Kind::NullValue(0),
            Value::Bool(value) => Kind::BoolValue(value),
            Value::Number(number) => Kind::NumberValue(number.as_f64()?),
            Value::String(value) => Kind::StringValue(value),
            Value::Array(values) => Kind::ListValue(prost_types::ListValue {
                values: values
                    .into_iter()
                    .map(serde_json_to_prost)
                    .collect::<Option<_>>()?,
            }),
            Value::Object(fields) => Kind::StructValue(prost_types::Struct {
                fields: fields
                    .into_iter()
                    .map(|(key, value)| serde_json_to_prost(value).map(|value| (key, value)))
                    .collect::<Option<_>>()?,
            }),
        }),
    })
}

fn optional_string_arg(args: &Value, name: &str) -> Option<String> {
    args.get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn optional_u32_arg(args: &Value, name: &str) -> Option<u32> {
    args.get(name)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn optional_i32_arg(args: &Value, name: &str) -> Option<i32> {
    args.get(name)
        .and_then(Value::as_i64)
        .and_then(|value| i32::try_from(value).ok())
}

fn optional_bool_arg(args: &Value, name: &str) -> Option<bool> {
    args.get(name).and_then(Value::as_bool)
}

#[cfg(test)]
pub(super) fn test_tool_card_events(
    task_id: &str,
    request_id: &str,
    tool_call: &OpenAIChatToolCall,
    result: &str,
) -> Option<Vec<api::ResponseEvent>> {
    structured_tool_card_events(task_id, request_id, tool_call, result, None)
}
