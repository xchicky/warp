use std::collections::BTreeMap;

use chrono::Local;
use serde_json::Value;
use warp_multi_agent_api as api;

use super::OpenAIChatToolCall;

pub(super) fn structured_tool_card_events(
    task_id: &str,
    request_id: &str,
    tool_call: &OpenAIChatToolCall,
    result: &str,
) -> Option<Vec<api::ResponseEvent>> {
    let tool = build_tool_call(&tool_call.function.name, &tool_call.function.arguments)?;
    let result = build_tool_call_result(&tool_call.function.name, result)?;

    Some(vec![
        add_tool_call_message(task_id, request_id, tool_call, tool),
        add_tool_call_result_message(task_id, request_id, &tool_call.id, result),
    ])
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
        _ => None,
    }
}

fn build_tool_call_result(
    name: &str,
    result: &str,
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

fn tool_error(result: &str) -> Option<String> {
    result
        .strip_prefix("Tool error:")
        .map(str::trim)
        .map(str::to_string)
}

fn parse_args(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or(Value::Null)
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

#[cfg(test)]
pub(super) fn test_tool_card_events(
    task_id: &str,
    request_id: &str,
    tool_call: &OpenAIChatToolCall,
    result: &str,
) -> Option<Vec<api::ResponseEvent>> {
    structured_tool_card_events(task_id, request_id, tool_call, result)
}
