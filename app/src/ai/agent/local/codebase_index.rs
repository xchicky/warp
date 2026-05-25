use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
    sync::{LazyLock, Mutex},
    time::SystemTime,
};

use anyhow::{anyhow, bail};
use serde::Deserialize;
use serde_json::{json, Value};
use warp_multi_agent_api as api;

const MAX_QUERY_CHARS: usize = 256;
const MAX_RESULTS: usize = 12;
const DEFAULT_RESULTS: usize = 8;
const MAX_FILE_BYTES: u64 = 256 * 1024;
const MAX_INDEX_FILES: usize = 2_000;
const MAX_INDEX_BYTES: u64 = 8 * 1024 * 1024;
const MAX_DEPTH: usize = 12;
const MAX_SNIPPET_CHARS: usize = 800;
const MAX_RESULT_TEXT_CHARS: usize = 24 * 1024;

static CODEBASE_INDEX_CACHE: LazyLock<Mutex<HashMap<PathBuf, LocalCodebaseIndex>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Debug, Deserialize)]
struct SearchCodebaseArgs {
    query: String,
    path: Option<String>,
    scope: Option<String>,
    max_results: Option<usize>,
    include_snippets: Option<bool>,
}

#[derive(Clone, Debug)]
struct LocalCodebaseIndex {
    root: PathBuf,
    scope: PathBuf,
    files: Vec<IndexedFile>,
    manifest: BTreeMap<PathBuf, FileStamp>,
    skipped: IndexSkipped,
}

#[derive(Clone, Debug)]
struct IndexedFile {
    relative_path: String,
    text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileStamp {
    modified: Option<SystemTime>,
    len: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct IndexSkipped {
    large_files: usize,
    binary_files: usize,
    non_utf8_files: usize,
    generated_paths: usize,
    limit_exceeded: bool,
}

#[derive(Clone, Debug)]
struct SearchHit {
    relative_path: String,
    line: usize,
    score: usize,
    snippet: String,
}

pub(super) fn search_codebase_tool_parameters() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": { "type": "string", "description": "Non-empty lexical/symbol query to search in the local codebase index." },
            "path": { "type": "string", "description": "Optional file or directory scope, absolute or relative to current workspace root." },
            "scope": { "type": "string", "description": "Optional alias for path." },
            "max_results": { "type": "integer", "minimum": 1, "maximum": MAX_RESULTS, "description": "Optional maximum number of results." },
            "include_snippets": { "type": "boolean", "description": "Whether to include bounded snippets. Defaults to true." }
        },
        "required": ["query"]
    })
}

pub(super) fn execute_search_codebase_tool(
    arguments: &str,
    cwd: Option<&Path>,
) -> anyhow::Result<String> {
    let args: SearchCodebaseArgs = serde_json::from_str(arguments)
        .map_err(|_| anyhow!("Invalid search_codebase arguments"))?;
    let query = bounded_query(&args.query)?;
    let root = cwd
        .ok_or_else(|| anyhow!("search_codebase requires a current workspace directory"))?
        .canonicalize()
        .map_err(|_| anyhow!("Current workspace directory is unavailable"))?;
    let scope = resolve_scope(&root, args.path.as_deref().or(args.scope.as_deref()))?;
    let max_results = args
        .max_results
        .unwrap_or(DEFAULT_RESULTS)
        .clamp(1, MAX_RESULTS);
    let include_snippets = args.include_snippets.unwrap_or(true);

    let cache_key = scope.clone();
    let mut cache = CODEBASE_INDEX_CACHE
        .lock()
        .map_err(|_| anyhow!("Local codebase index cache is unavailable"))?;
    if let Some(index) = cache.get(&cache_key) {
        let current_manifest = scan_manifest(&root, &scope)?;
        if current_manifest != index.manifest {
            cache.remove(&cache_key);
            return Ok(format!(
                "Status: stale\nScope: {}\nThe local codebase index changed after it was built. Re-run search_codebase to rebuild the local in-memory index before relying on results.",
                relative_scope_for_display(&root, &scope)
            ));
        }
        return Ok(search_index(index, &query, max_results, include_snippets));
    }

    let index = build_index(root, scope.clone())?;
    let result = search_index(&index, &query, max_results, include_snippets);
    cache.insert(cache_key, index);
    Ok(result)
}

pub(super) fn search_codebase_result_from_text(result: &str) -> api::SearchCodebaseResult {
    if result.starts_with("Tool error:") || result.starts_with("Status: stale") {
        return api::SearchCodebaseResult {
            result: Some(api::search_codebase_result::Result::Error(
                api::search_codebase_result::Error {
                    message: truncate_chars(result, MAX_SNIPPET_CHARS),
                },
            )),
        };
    }

    let files = parse_result_files(result);

    api::SearchCodebaseResult {
        result: Some(api::search_codebase_result::Result::Success(
            api::search_codebase_result::Success { files },
        )),
    }
}

pub(super) fn build_search_codebase_tool_call(
    arguments: &str,
) -> api::message::tool_call::SearchCodebase {
    let args = serde_json::from_str::<Value>(arguments).unwrap_or_default();
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let path = args
        .get("path")
        .or_else(|| args.get("scope"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    api::message::tool_call::SearchCodebase {
        query,
        path_filters: (!path.is_empty())
            .then(|| path.clone())
            .into_iter()
            .collect(),
        codebase_path: path,
    }
}

fn bounded_query(query: &str) -> anyhow::Result<String> {
    let query = query.trim();
    if query.is_empty() {
        bail!("search_codebase query must be non-empty");
    }
    Ok(truncate_chars(query, MAX_QUERY_CHARS))
}

fn resolve_scope(root: &Path, scope: Option<&str>) -> anyhow::Result<PathBuf> {
    let candidate = match scope.map(str::trim).filter(|scope| !scope.is_empty()) {
        Some(scope) => {
            let path = PathBuf::from(scope);
            if path.is_absolute() {
                path
            } else {
                root.join(path)
            }
        }
        None => root.to_path_buf(),
    };
    let canonical = candidate
        .canonicalize()
        .map_err(|_| anyhow!("search_codebase scope does not exist"))?;
    if !canonical.starts_with(root) {
        bail!("search_codebase scope escapes the current workspace root");
    }
    Ok(canonical)
}

fn build_index(root: PathBuf, scope: PathBuf) -> anyhow::Result<LocalCodebaseIndex> {
    let mut files = Vec::new();
    let mut manifest = BTreeMap::new();
    let mut skipped = IndexSkipped::default();
    let mut total_bytes = 0u64;
    collect_index_files(
        &root,
        &scope,
        0,
        &mut files,
        &mut manifest,
        &mut skipped,
        &mut total_bytes,
    )?;
    Ok(LocalCodebaseIndex {
        root,
        scope,
        files,
        manifest,
        skipped,
    })
}

fn collect_index_files(
    root: &Path,
    path: &Path,
    depth: usize,
    files: &mut Vec<IndexedFile>,
    manifest: &mut BTreeMap<PathBuf, FileStamp>,
    skipped: &mut IndexSkipped,
    total_bytes: &mut u64,
) -> anyhow::Result<()> {
    if depth > MAX_DEPTH || files.len() >= MAX_INDEX_FILES || *total_bytes >= MAX_INDEX_BYTES {
        skipped.limit_exceeded = true;
        return Ok(());
    }

    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        let canonical = path.canonicalize()?;
        if !canonical.starts_with(root) {
            bail!("search_codebase denied a symlink escape");
        }
    }

    if metadata.is_dir() {
        if is_generated_path(path) {
            skipped.generated_paths += 1;
            return Ok(());
        }
        let mut entries = fs::read_dir(path)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        entries.sort();
        for entry in entries {
            collect_index_files(
                root,
                &entry,
                depth + 1,
                files,
                manifest,
                skipped,
                total_bytes,
            )?;
        }
        return Ok(());
    }

    if !metadata.is_file() {
        return Ok(());
    }

    let canonical = path.canonicalize()?;
    if !canonical.starts_with(root) {
        bail!("search_codebase path escapes the current workspace root");
    }

    manifest.insert(
        canonical.clone(),
        FileStamp {
            modified: metadata.modified().ok(),
            len: metadata.len(),
        },
    );

    if metadata.len() > MAX_FILE_BYTES {
        skipped.large_files += 1;
        return Ok(());
    }
    if *total_bytes + metadata.len() > MAX_INDEX_BYTES {
        skipped.limit_exceeded = true;
        return Ok(());
    }

    let bytes = fs::read(&canonical)?;
    if is_binary(&bytes) {
        skipped.binary_files += 1;
        return Ok(());
    }
    let Ok(text) = String::from_utf8(bytes) else {
        skipped.non_utf8_files += 1;
        return Ok(());
    };
    *total_bytes += metadata.len();
    let relative_path = canonical
        .strip_prefix(root)
        .unwrap_or(canonical.as_path())
        .to_string_lossy()
        .to_string();
    files.push(IndexedFile {
        relative_path,
        text,
    });
    Ok(())
}

fn scan_manifest(root: &Path, scope: &Path) -> anyhow::Result<BTreeMap<PathBuf, FileStamp>> {
    let mut files = Vec::new();
    let mut manifest = BTreeMap::new();
    let mut skipped = IndexSkipped::default();
    let mut total_bytes = 0u64;
    collect_index_files(
        root,
        scope,
        0,
        &mut files,
        &mut manifest,
        &mut skipped,
        &mut total_bytes,
    )?;
    Ok(manifest)
}

fn search_index(
    index: &LocalCodebaseIndex,
    query: &str,
    max_results: usize,
    include_snippets: bool,
) -> String {
    let query_lower = query.to_lowercase();
    let mut hits = Vec::new();
    for file in &index.files {
        let path_score = file
            .relative_path
            .to_lowercase()
            .matches(&query_lower)
            .count();
        let file_score = file.text.to_lowercase().matches(&query_lower).count();
        for (line_index, line) in file.text.lines().enumerate() {
            let line_score = line.to_lowercase().matches(&query_lower).count();
            if line_score + path_score == 0 {
                continue;
            }
            let score = line_score + path_score + file_score;
            let snippet = if include_snippets {
                snippet_for_line(&file.text, line_index)
            } else {
                String::new()
            };
            hits.push(SearchHit {
                relative_path: file.relative_path.clone(),
                line: line_index + 1,
                score,
                snippet,
            });
        }
    }
    hits.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.relative_path.cmp(&right.relative_path))
            .then_with(|| left.line.cmp(&right.line))
    });
    hits.truncate(max_results);

    let mut output = format!(
        "Status: ready\nIndex: local lexical in-memory\nScope: {}\nFiles indexed: {}\nSkipped: large_files={}, binary_files={}, non_utf8_files={}, generated_paths={}, limit_exceeded={}\nResults: {}\n",
        relative_scope_for_display(&index.root, &index.scope),
        index.files.len(),
        index.skipped.large_files,
        index.skipped.binary_files,
        index.skipped.non_utf8_files,
        index.skipped.generated_paths,
        index.skipped.limit_exceeded,
        hits.len()
    );
    for (index, hit) in hits.iter().enumerate() {
        output.push_str(&format!(
            "\n{}. Path: {}\nLine: {}\nScore: {}\n",
            index + 1,
            hit.relative_path,
            hit.line,
            hit.score
        ));
        if include_snippets && !hit.snippet.is_empty() {
            output.push_str("Snippet:\n```text\n");
            output.push_str(&hit.snippet);
            output.push_str("\n```\n");
        }
    }
    truncate_chars(&output, MAX_RESULT_TEXT_CHARS)
}

fn snippet_for_line(text: &str, line_index: usize) -> String {
    let lines = text.lines().collect::<Vec<_>>();
    let start = line_index.saturating_sub(1);
    let end = (line_index + 2).min(lines.len());
    truncate_chars(&lines[start..end].join("\n"), MAX_SNIPPET_CHARS)
}

fn is_generated_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            matches!(
                name,
                ".git" | "node_modules" | "target" | "dist" | "build" | ".next" | ".turbo"
            )
        })
}

fn is_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(4096).any(|byte| *byte == 0)
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut truncated = text.chars().take(max_chars).collect::<String>();
    truncated.push_str("\n[truncated]");
    truncated
}

fn relative_scope_for_display(root: &Path, scope: &Path) -> String {
    scope
        .strip_prefix(root)
        .ok()
        .filter(|relative| !relative.as_os_str().is_empty())
        .map(|relative| relative.to_string_lossy().to_string())
        .unwrap_or_else(|| ".".to_string())
}

fn parse_result_files(result: &str) -> Vec<api::FileContent> {
    let lines = result.lines().collect::<Vec<_>>();
    let mut files = Vec::new();
    let mut index = 0;
    while index < lines.len() && files.len() < MAX_RESULTS {
        let Some(path) = lines[index]
            .strip_prefix("Path: ")
            .or_else(|| lines[index].split_once(". Path: ").map(|(_, path)| path))
        else {
            index += 1;
            continue;
        };
        let line_number = lines
            .get(index + 1)
            .and_then(|line| line.strip_prefix("Line: "))
            .and_then(|line| line.parse::<u32>().ok())
            .unwrap_or(1);
        let snippet = lines[index..]
            .iter()
            .position(|line| *line == "```text")
            .and_then(|snippet_start| {
                let start = index + snippet_start + 1;
                let end = lines[start..]
                    .iter()
                    .position(|line| *line == "```")
                    .map(|offset| start + offset)?;
                Some(lines[start..end].join("\n"))
            })
            .unwrap_or_default();

        files.push(api::FileContent {
            file_path: path.to_string(),
            content: snippet,
            line_range: Some(api::FileContentLineRange {
                start: line_number,
                end: line_number,
            }),
        });
        index += 1;
    }
    files
}

#[cfg(test)]
pub(super) fn clear_codebase_index_cache_for_tests() {
    if let Ok(mut cache) = CODEBASE_INDEX_CACHE.lock() {
        cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_file(root: &Path, relative: &str, content: &[u8]) -> PathBuf {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn path_scope_rejects_parent_and_symlink_escape() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        write_file(outside.path(), "secret.rs", b"secret");

        assert!(resolve_scope(root.path(), Some("../outside")).is_err());

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(outside.path(), root.path().join("link")).unwrap();
            let root = root.path().canonicalize().unwrap();
            assert!(resolve_scope(&root, Some("link")).is_err());
        }
    }

    #[test]
    fn search_results_are_bounded_and_deterministic() {
        clear_codebase_index_cache_for_tests();
        let root = tempfile::tempdir().unwrap();
        write_file(root.path(), "src/a.rs", b"fn target() {}\nfn other() {}\n");
        write_file(root.path(), "src/b.rs", b"target();\ntarget();\n");

        let result = execute_search_codebase_tool(
            r#"{"query":"target","max_results":3}"#,
            Some(root.path()),
        )
        .unwrap();

        assert!(result.contains("Status: ready"));
        assert!(result.contains("Results: 3"));
        assert!(result.find("src/b.rs").unwrap() < result.find("src/a.rs").unwrap());
    }

    #[test]
    fn large_and_binary_files_are_skipped_with_bounded_metadata() {
        clear_codebase_index_cache_for_tests();
        let root = tempfile::tempdir().unwrap();
        write_file(
            root.path(),
            "large.txt",
            &vec![b'a'; (MAX_FILE_BYTES + 1) as usize],
        );
        write_file(root.path(), "binary.bin", b"\0binary");

        let result = execute_search_codebase_tool(
            r#"{"query":"anything","max_results":4}"#,
            Some(root.path()),
        )
        .unwrap();

        assert!(result.contains("large_files=1"));
        assert!(result.contains("binary_files=1"));
    }

    #[test]
    fn non_utf8_files_are_skipped_without_aborting_index() {
        clear_codebase_index_cache_for_tests();
        let root = tempfile::tempdir().unwrap();
        write_file(root.path(), "src/lib.rs", b"fn target() {}\n");
        write_file(
            root.path(),
            "invalid.txt",
            &[0xff, 0xfe, b't', b'e', b'x', b't'],
        );

        let result =
            execute_search_codebase_tool(r#"{"query":"target"}"#, Some(root.path())).unwrap();

        assert!(result.contains("Status: ready"));
        assert!(result.contains("non_utf8_files=1"));
        assert!(result.contains("src/lib.rs"));
    }

    #[test]
    fn total_byte_cap_does_not_make_unchanged_index_immediately_stale() {
        clear_codebase_index_cache_for_tests();
        let root = tempfile::tempdir().unwrap();
        let file_count_over_cap = (MAX_INDEX_BYTES / MAX_FILE_BYTES) + 1;
        for index in 0..file_count_over_cap {
            write_file(
                root.path(),
                &format!("src/{index:03}.txt"),
                &vec![b'a'; MAX_FILE_BYTES as usize],
            );
        }

        let first =
            execute_search_codebase_tool(r#"{"query":"missing"}"#, Some(root.path())).unwrap();
        assert!(first.contains("Status: ready"));
        assert!(first.contains("limit_exceeded=true"));

        let second =
            execute_search_codebase_tool(r#"{"query":"missing"}"#, Some(root.path())).unwrap();
        assert!(second.contains("Status: ready"));
        assert!(!second.contains("Status: stale"));
    }

    #[test]
    fn stale_index_state_is_provider_visible() {
        clear_codebase_index_cache_for_tests();
        let root = tempfile::tempdir().unwrap();
        let path = write_file(root.path(), "src/lib.rs", b"alpha\n");
        let first =
            execute_search_codebase_tool(r#"{"query":"alpha"}"#, Some(root.path())).unwrap();
        assert!(first.contains("Status: ready"));

        fs::write(path, b"beta\n").unwrap();
        let second =
            execute_search_codebase_tool(r#"{"query":"beta"}"#, Some(root.path())).unwrap();

        assert!(second.contains("Status: stale"));
    }

    #[test]
    fn local_index_module_does_not_reference_cloud_store() {
        let source = include_str!("codebase_index.rs");

        let cloud_index_module = ["full_source_code", "emb", "edding"].join("_");
        let cloud_client_type = ["Store", "Client"].concat();
        assert!(!source.contains(&cloud_index_module));
        assert!(!source.contains(&cloud_client_type));
    }
}
