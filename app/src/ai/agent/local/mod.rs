use std::{fmt, pin::Pin, sync::Arc, time::Duration};

use anyhow::anyhow;
use async_stream::stream;
use chrono::Local;
use futures_lite::Stream;
use futures_util::StreamExt as _;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use warp_multi_agent_api as api;

use crate::{
    ai::agent::{api::Event, task::helper::TaskExt, AIAgentInput},
    server::server_api::AIApiError,
};

#[derive(Clone, PartialEq, Eq)]
pub struct LocalDirectConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

impl fmt::Debug for LocalDirectConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalDirectConfig")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .finish()
    }
}

#[derive(Debug, Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIChatMessage>,
    stream: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct OpenAIChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamResponse {
    choices: Vec<OpenAIStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    delta: OpenAIStreamDelta,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamDelta {
    content: Option<String>,
}

#[derive(Debug, PartialEq, Eq)]
enum OpenAIStreamEvent {
    Delta(String),
    Done,
}

const LOCAL_DIRECT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_LOCAL_DIRECT_RESPONSE_BYTES: u64 = 10 * 1024 * 1024;
const MAX_LOCAL_DIRECT_HISTORY_MESSAGES: usize = 20;
const MAX_LOCAL_DIRECT_MESSAGE_CHARS: usize = 32 * 1024;

pub type LocalResponseStream = Pin<Box<dyn Stream<Item = Event> + Send + 'static>>;

pub fn generate_openai_compatible_output(
    config: LocalDirectConfig,
    input: Vec<AIAgentInput>,
    tasks: Vec<api::Task>,
) -> LocalResponseStream {
    Box::pin(stream! {
        let request_id = Uuid::new_v4().to_string();
        let conversation_id = Uuid::new_v4().to_string();
        let root_task_id = root_task_id(&tasks);

        yield Ok(stream_init(&request_id, &conversation_id));

        let (task_id, created_task) = match root_task_id {
            Some(task_id) => (task_id, false),
            None => (Uuid::new_v4().to_string(), true),
        };
        let message_id = Uuid::new_v4().to_string();

        if created_task {
            yield Ok(create_task(&task_id));
        }
        yield Ok(add_agent_output_message(
            &task_id,
            &request_id,
            &message_id,
            String::new(),
        ));

        match request_openai_compatible_completion(&config, &input, &tasks).await {
            Ok(response) => {
                let mut body = response.bytes_stream();
                let mut pending = String::new();
                let mut bytes_read = 0u64;
                let mut received_token = false;
                let mut stream_done = false;

                while let Some(chunk) = body.next().await {
                    let events = match sse_events_from_chunk(&mut pending, &mut bytes_read, chunk) {
                        Ok(events) => events,
                        Err(error) => {
                            yield Err(Arc::new(AIApiError::Other(error)));
                            return;
                        }
                    };

                    for event in events {
                        match event {
                            OpenAIStreamEvent::Delta(token) => {
                                received_token = true;
                                yield Ok(append_agent_output_message_content(
                                    &task_id,
                                    &request_id,
                                    &message_id,
                                    token,
                                ));
                            }
                            OpenAIStreamEvent::Done => {
                                stream_done = true;
                                break;
                            }
                        }
                    }

                    if stream_done {
                        break;
                    }
                }

                if !stream_done && !pending.is_empty() {
                    pending.push('\n');
                    let events = match drain_sse_events(&mut pending) {
                        Ok(events) => events,
                        Err(error) => {
                            yield Err(Arc::new(AIApiError::Other(error)));
                            return;
                        }
                    };
                    for event in events {
                        match event {
                            OpenAIStreamEvent::Delta(token) => {
                                received_token = true;
                                yield Ok(append_agent_output_message_content(
                                    &task_id,
                                    &request_id,
                                    &message_id,
                                    token,
                                ));
                            }
                            OpenAIStreamEvent::Done => break,
                        }
                    }
                }

                if !received_token {
                    yield Err(Arc::new(AIApiError::Other(anyhow!(
                        "Local direct provider returned an empty response"
                    ))));
                    return;
                }

                yield Ok(stream_finished_done());
            }
            Err(error) => {
                yield Err(Arc::new(AIApiError::Other(error)));
            }
        }
    })
}

fn root_task(tasks: &[api::Task]) -> Option<&api::Task> {
    tasks.iter().find(|task| task.parent_id().is_none())
}

fn root_task_id(tasks: &[api::Task]) -> Option<String> {
    root_task(tasks).map(|task| task.id.clone())
}

async fn request_openai_compatible_completion(
    config: &LocalDirectConfig,
    input: &[AIAgentInput],
    tasks: &[api::Task],
) -> anyhow::Result<reqwest::Response> {
    let request = OpenAIChatRequest {
        model: config.model.clone(),
        messages: openai_messages_from_inputs_and_tasks(input, tasks)?,
        stream: true,
    };

    let endpoint = chat_completions_url(&config.base_url);
    let response = reqwest::Client::builder()
        .timeout(LOCAL_DIRECT_REQUEST_TIMEOUT)
        .build()
        .map_err(|_| anyhow!("Failed to initialize local direct provider client"))?
        .post(endpoint)
        .bearer_auth(&config.api_key)
        .json(&request)
        .send()
        .await
        .map_err(|_| anyhow!("Failed to send request to local direct provider"))?;

    let status = response.status();
    if !status.is_success() {
        return Err(anyhow!(
            "Local direct provider request failed with status {status}"
        ));
    }

    Ok(response)
}

fn openai_messages_from_inputs_and_tasks(
    input: &[AIAgentInput],
    tasks: &[api::Task],
) -> anyhow::Result<Vec<OpenAIChatMessage>> {
    let mut messages = vec![OpenAIChatMessage {
        role: "system",
        content: "You are a helpful local coding assistant running inside Warp.".to_string(),
    }];

    if let Some(root) = root_task(tasks) {
        let mut history = root
            .messages
            .iter()
            .rev()
            .take(MAX_LOCAL_DIRECT_HISTORY_MESSAGES)
            .collect::<Vec<_>>();
        history.reverse();
        for message in history {
            match &message.message {
                Some(api::message::Message::UserQuery(query)) if !query.query.is_empty() => {
                    messages.push(OpenAIChatMessage {
                        role: "user",
                        content: truncate_message_content(
                            &query.query,
                            MAX_LOCAL_DIRECT_MESSAGE_CHARS,
                        ),
                    });
                }
                Some(api::message::Message::AgentOutput(output)) if !output.text.is_empty() => {
                    messages.push(OpenAIChatMessage {
                        role: "assistant",
                        content: truncate_message_content(
                            &output.text,
                            MAX_LOCAL_DIRECT_MESSAGE_CHARS,
                        ),
                    });
                }
                _ => {}
            }
        }
    }

    let user_query = input
        .iter()
        .filter_map(AIAgentInput::user_query)
        .collect::<Vec<_>>()
        .join("\n\n");
    if user_query.is_empty() {
        return Err(anyhow!(
            "Local direct agent only supports user query inputs"
        ));
    }
    let user_query = truncate_message_content(&user_query, MAX_LOCAL_DIRECT_MESSAGE_CHARS);
    if messages
        .last()
        .is_none_or(|message| message.role != "user" || message.content != user_query)
    {
        messages.push(OpenAIChatMessage {
            role: "user",
            content: user_query,
        });
    }

    truncate_messages_to_char_budget(&mut messages, MAX_LOCAL_DIRECT_MESSAGE_CHARS);

    Ok(messages)
}

fn truncate_message_content(content: &str, max_chars: usize) -> String {
    let char_count = content.chars().count();
    if char_count <= max_chars {
        return content.to_string();
    }

    let start_index = content
        .char_indices()
        .nth(char_count - max_chars)
        .map(|(index, _)| index)
        .unwrap_or(0);
    content[start_index..].to_string()
}

fn truncate_messages_to_char_budget(messages: &mut Vec<OpenAIChatMessage>, max_chars: usize) {
    if messages.len() <= 1 {
        return;
    }

    let non_system_chars = messages[1..]
        .iter()
        .map(|message| message.content.chars().count())
        .sum::<usize>();
    if non_system_chars <= max_chars {
        return;
    }

    let system_message = messages.remove(0);
    let mut retained = Vec::new();
    let mut remaining_chars = max_chars;

    for message in messages.drain(..).rev() {
        if remaining_chars == 0 {
            break;
        }

        let message_chars = message.content.chars().count();
        if message_chars <= remaining_chars {
            remaining_chars -= message_chars;
            retained.push(message);
        } else {
            retained.push(OpenAIChatMessage {
                role: message.role,
                content: truncate_message_content(&message.content, remaining_chars),
            });
            break;
        }
    }

    retained.reverse();
    messages.push(system_message);
    messages.extend(retained);
}

fn sse_events_from_chunk(
    pending: &mut String,
    bytes_read: &mut u64,
    chunk: Result<bytes::Bytes, reqwest::Error>,
) -> anyhow::Result<Vec<OpenAIStreamEvent>> {
    let chunk = chunk.map_err(|_| anyhow!("Failed to read local direct provider response"))?;
    *bytes_read += chunk.len() as u64;
    if *bytes_read > MAX_LOCAL_DIRECT_RESPONSE_BYTES {
        return Err(anyhow!("Local direct provider response is too large"));
    }
    let chunk = std::str::from_utf8(&chunk)
        .map_err(|_| anyhow!("Local direct provider response is not valid UTF-8"))?;
    pending.push_str(chunk);
    drain_sse_events(pending)
}

fn drain_sse_events(pending: &mut String) -> anyhow::Result<Vec<OpenAIStreamEvent>> {
    let mut events = Vec::new();
    while let Some(newline_index) = pending.find('\n') {
        let line = pending[..newline_index].trim().to_string();
        pending.drain(..=newline_index);
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data == "[DONE]" {
            events.push(OpenAIStreamEvent::Done);
            continue;
        }
        let event: OpenAIStreamResponse = serde_json::from_str(data)
            .map_err(|_| anyhow!("Failed to parse local direct provider stream response"))?;
        events.extend(event.choices.into_iter().filter_map(|choice| {
            choice
                .delta
                .content
                .filter(|content| !content.is_empty())
                .map(OpenAIStreamEvent::Delta)
        }));
    }
    Ok(events)
}

fn chat_completions_url(base_url: &str) -> String {
    let base_url = base_url.trim_end_matches('/');
    if base_url.ends_with("/chat/completions") {
        base_url.to_string()
    } else {
        format!("{base_url}/chat/completions")
    }
}

fn stream_init(request_id: &str, conversation_id: &str) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::Init(
            api::response_event::StreamInit {
                conversation_id: conversation_id.to_string(),
                request_id: request_id.to_string(),
                run_id: String::new(),
            },
        )),
    }
}

fn create_task(task_id: &str) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::CreateTask(
                        api::client_action::CreateTask {
                            task: Some(api::Task {
                                id: task_id.to_string(),
                                description: "Local direct response".to_string(),
                                dependencies: None,
                                messages: vec![],
                                summary: String::new(),
                                server_data: String::new(),
                            }),
                        },
                    )),
                }],
            },
        )),
    }
}

fn add_agent_output_message(
    task_id: &str,
    request_id: &str,
    message_id: &str,
    text: String,
) -> api::ResponseEvent {
    let now = Local::now();
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::AddMessagesToTask(
                        api::client_action::AddMessagesToTask {
                            task_id: task_id.to_string(),
                            messages: vec![api::Message {
                                id: message_id.to_string(),
                                task_id: task_id.to_string(),
                                request_id: request_id.to_string(),
                                timestamp: Some(prost_types::Timestamp {
                                    seconds: now.timestamp(),
                                    nanos: now.timestamp_subsec_nanos() as i32,
                                }),
                                server_message_data: String::new(),
                                citations: vec![],
                                message: Some(api::message::Message::AgentOutput(
                                    api::message::AgentOutput { text },
                                )),
                            }],
                        },
                    )),
                }],
            },
        )),
    }
}

fn append_agent_output_message_content(
    task_id: &str,
    request_id: &str,
    message_id: &str,
    text: String,
) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::AppendToMessageContent(
                        api::client_action::AppendToMessageContent {
                            task_id: task_id.to_string(),
                            message: Some(api::Message {
                                id: message_id.to_string(),
                                task_id: task_id.to_string(),
                                request_id: request_id.to_string(),
                                timestamp: None,
                                server_message_data: String::new(),
                                citations: vec![],
                                message: Some(api::message::Message::AgentOutput(
                                    api::message::AgentOutput { text },
                                )),
                            }),
                            mask: Some(prost_types::FieldMask {
                                paths: vec!["agent_output.text".to_string()],
                            }),
                        },
                    )),
                }],
            },
        )),
    }
}

fn stream_finished_done() -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::Finished(
            api::response_event::StreamFinished {
                reason: Some(api::response_event::stream_finished::Reason::Done(
                    api::response_event::stream_finished::Done {},
                )),
                token_usage: vec![],
                should_refresh_model_config: false,
                request_cost: None,
                conversation_usage_metadata: Some(
                    api::response_event::stream_finished::ConversationUsageMetadata {
                        context_window_usage: 0.,
                        summarized: false,
                        credits_spent: 0.,
                        #[allow(deprecated)]
                        token_usage: vec![],
                        tool_usage_metadata: None,
                        warp_token_usage: Default::default(),
                        byok_token_usage: Default::default(),
                    },
                ),
            },
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        chat_completions_url, drain_sse_events, openai_messages_from_inputs_and_tasks,
        root_task_id, OpenAIStreamEvent, MAX_LOCAL_DIRECT_HISTORY_MESSAGES,
        MAX_LOCAL_DIRECT_MESSAGE_CHARS,
    };
    use warp_multi_agent_api as api;

    #[test]
    fn chat_completions_url_appends_path_to_base_url() {
        assert_eq!(
            chat_completions_url("http://localhost:11434/v1"),
            "http://localhost:11434/v1/chat/completions"
        );
    }

    #[test]
    fn chat_completions_url_keeps_full_endpoint() {
        assert_eq!(
            chat_completions_url("https://example.com/v1/chat/completions"),
            "https://example.com/v1/chat/completions"
        );
    }

    #[test]
    fn root_task_id_uses_first_task_without_parent() {
        let tasks = vec![api::Task {
            id: "root".to_string(),
            description: String::new(),
            dependencies: None,
            messages: vec![],
            summary: String::new(),
            server_data: String::new(),
        }];

        assert_eq!(root_task_id(&tasks).as_deref(), Some("root"));
    }

    #[test]
    fn root_task_id_ignores_subtasks() {
        let tasks = vec![api::Task {
            id: "child".to_string(),
            description: String::new(),
            dependencies: Some(api::task::Dependencies {
                parent_task_id: "root".to_string(),
            }),
            messages: vec![],
            summary: String::new(),
            server_data: String::new(),
        }];

        assert_eq!(root_task_id(&tasks), None);
    }
    #[test]
    fn drain_sse_events_extracts_openai_content_deltas() {
        let mut pending = "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: [DONE]\n".to_string();

        let events = drain_sse_events(&mut pending).unwrap();

        assert_eq!(
            events,
            vec![
                OpenAIStreamEvent::Delta("hel".to_string()),
                OpenAIStreamEvent::Delta("lo".to_string()),
                OpenAIStreamEvent::Done,
            ]
        );
        assert!(pending.is_empty());
    }

    #[test]
    fn drain_sse_events_waits_for_partial_lines() {
        let mut pending = "data: {\"choices\":[{\"delta\":{\"content\":\"hel".to_string();

        let events = drain_sse_events(&mut pending).unwrap();

        assert!(events.is_empty());
        assert_eq!(pending, "data: {\"choices\":[{\"delta\":{\"content\":\"hel");

        pending.push_str("lo\"}}]}\r\n");
        let events = drain_sse_events(&mut pending).unwrap();

        assert_eq!(events, vec![OpenAIStreamEvent::Delta("hello".to_string())]);
        assert!(pending.is_empty());
    }

    #[test]
    fn drain_sse_events_rejects_malformed_json() {
        let mut pending = "data: {not-json}\n".to_string();

        let error = drain_sse_events(&mut pending).unwrap_err();

        assert_eq!(
            error.to_string(),
            "Failed to parse local direct provider stream response"
        );
    }

    #[test]
    fn openai_messages_include_history_and_current_query_without_duplicate() {
        let tasks = vec![api::Task {
            id: "root".to_string(),
            description: String::new(),
            dependencies: None,
            messages: vec![
                api::Message {
                    id: "user-1".to_string(),
                    task_id: "root".to_string(),
                    request_id: "request-1".to_string(),
                    timestamp: None,
                    server_message_data: String::new(),
                    citations: vec![],
                    message: Some(api::message::Message::UserQuery(api::message::UserQuery {
                        query: "remember alpha".to_string(),
                        context: None,
                        referenced_attachments: Default::default(),
                        mode: None,
                        intended_agent: 0,
                    })),
                },
                api::Message {
                    id: "assistant-1".to_string(),
                    task_id: "root".to_string(),
                    request_id: "request-1".to_string(),
                    timestamp: None,
                    server_message_data: String::new(),
                    citations: vec![],
                    message: Some(api::message::Message::AgentOutput(
                        api::message::AgentOutput {
                            text: "alpha remembered".to_string(),
                        },
                    )),
                },
                api::Message {
                    id: "user-2".to_string(),
                    task_id: "root".to_string(),
                    request_id: "request-2".to_string(),
                    timestamp: None,
                    server_message_data: String::new(),
                    citations: vec![],
                    message: Some(api::message::Message::UserQuery(api::message::UserQuery {
                        query: "what did I say?".to_string(),
                        context: None,
                        referenced_attachments: Default::default(),
                        mode: None,
                        intended_agent: 0,
                    })),
                },
            ],
            summary: String::new(),
            server_data: String::new(),
        }];
        let input = vec![crate::ai::agent::AIAgentInput::UserQuery {
            query: "what did I say?".to_string(),
            context: std::sync::Arc::from([]),
            static_query_type: None,
            referenced_attachments: Default::default(),
            user_query_mode: crate::ai::agent::UserQueryMode::Normal,
            running_command: None,
            intended_agent: None,
        }];

        let messages = openai_messages_from_inputs_and_tasks(&input, &tasks).unwrap();

        assert_eq!(messages[0].role, "system");
        assert_eq!(
            messages
                .iter()
                .map(|message| (message.role, message.content.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (
                    "system",
                    "You are a helpful local coding assistant running inside Warp."
                ),
                ("user", "remember alpha"),
                ("assistant", "alpha remembered"),
                ("user", "what did I say?"),
            ]
        );
    }

    #[test]
    fn openai_messages_limit_history_to_recent_messages() {
        let messages = (0..MAX_LOCAL_DIRECT_HISTORY_MESSAGES + 5)
            .map(|index| api::Message {
                id: format!("user-{index}"),
                task_id: "root".to_string(),
                request_id: format!("request-{index}"),
                timestamp: None,
                server_message_data: String::new(),
                citations: vec![],
                message: Some(api::message::Message::UserQuery(api::message::UserQuery {
                    query: format!("history {index}"),
                    context: None,
                    referenced_attachments: Default::default(),
                    mode: None,
                    intended_agent: 0,
                })),
            })
            .collect();
        let tasks = vec![api::Task {
            id: "root".to_string(),
            description: String::new(),
            dependencies: None,
            messages,
            summary: String::new(),
            server_data: String::new(),
        }];
        let input = vec![crate::ai::agent::AIAgentInput::UserQuery {
            query: "current".to_string(),
            context: std::sync::Arc::from([]),
            static_query_type: None,
            referenced_attachments: Default::default(),
            user_query_mode: crate::ai::agent::UserQueryMode::Normal,
            running_command: None,
            intended_agent: None,
        }];

        let messages = openai_messages_from_inputs_and_tasks(&input, &tasks).unwrap();

        assert!(messages
            .iter()
            .all(|message| message.content != "history 0"));
        assert!(messages
            .iter()
            .any(|message| message.content == "history 24"));
        assert_eq!(messages.last().unwrap().content, "current");
    }

    #[test]
    fn openai_messages_limit_total_non_system_chars() {
        let tasks = vec![api::Task {
            id: "root".to_string(),
            description: String::new(),
            dependencies: None,
            messages: vec![api::Message {
                id: "assistant-1".to_string(),
                task_id: "root".to_string(),
                request_id: "request-1".to_string(),
                timestamp: None,
                server_message_data: String::new(),
                citations: vec![],
                message: Some(api::message::Message::AgentOutput(
                    api::message::AgentOutput {
                        text: "a".repeat(MAX_LOCAL_DIRECT_MESSAGE_CHARS),
                    },
                )),
            }],
            summary: String::new(),
            server_data: String::new(),
        }];
        let input = vec![crate::ai::agent::AIAgentInput::UserQuery {
            query: "current".to_string(),
            context: std::sync::Arc::from([]),
            static_query_type: None,
            referenced_attachments: Default::default(),
            user_query_mode: crate::ai::agent::UserQueryMode::Normal,
            running_command: None,
            intended_agent: None,
        }];

        let messages = openai_messages_from_inputs_and_tasks(&input, &tasks).unwrap();
        let non_system_chars = messages[1..]
            .iter()
            .map(|message| message.content.chars().count())
            .sum::<usize>();

        assert!(non_system_chars <= MAX_LOCAL_DIRECT_MESSAGE_CHARS);
        assert_eq!(messages.last().unwrap().content, "current");
    }
}
