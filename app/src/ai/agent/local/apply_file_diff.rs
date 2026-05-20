use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, bail};

const MAX_LOCAL_DIRECT_PATCH_BYTES: usize = 128 * 1024;
const MAX_LOCAL_DIRECT_PATCH_FILES: usize = 20;
const MAX_LOCAL_DIRECT_PATCH_HUNKS: usize = 200;
const MAX_LOCAL_DIRECT_PATCH_OUTPUT_PATHS: usize = 20;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AppliedFileDiff {
    pub path: String,
    pub additions: usize,
    pub removals: usize,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ApplyFileDiffSummary {
    pub files: Vec<AppliedFileDiff>,
}

impl ApplyFileDiffSummary {
    fn total_additions(&self) -> usize {
        self.files.iter().map(|file| file.additions).sum()
    }

    fn total_removals(&self) -> usize {
        self.files.iter().map(|file| file.removals).sum()
    }
}

#[derive(Debug)]
struct FilePatch {
    old_path: String,
    new_path: String,
    hunks: Vec<Hunk>,
}

#[derive(Debug)]
struct Hunk {
    old_start: usize,
    lines: Vec<HunkLine>,
}

#[derive(Debug)]
enum HunkLine {
    Context(String),
    Remove(String),
    Add(String),
}

#[derive(Debug)]
struct PreparedPatch {
    path: PathBuf,
    display_path: String,
    original_content: String,
    content: String,
    additions: usize,
    removals: usize,
}

pub(super) fn apply_unified_diff(
    patch: &str,
    cwd: Option<&Path>,
    max_file_bytes: u64,
) -> anyhow::Result<ApplyFileDiffSummary> {
    if patch.trim().is_empty() {
        bail!("patch cannot be empty");
    }
    if patch.len() > MAX_LOCAL_DIRECT_PATCH_BYTES {
        bail!("patch is too large");
    }

    let root = writable_root(cwd)?;
    let file_patches = parse_unified_diff(patch)?;
    if file_patches.len() > MAX_LOCAL_DIRECT_PATCH_FILES {
        bail!("patch touches too many files");
    }

    let mut prepared = Vec::new();
    let mut seen_paths = BTreeSet::new();
    for file_patch in file_patches {
        validate_update_only_patch(&file_patch)?;
        let display_path = normalize_diff_path(&file_patch.new_path)?;
        if !seen_paths.insert(display_path.clone()) {
            bail!("duplicate file patch: {display_path}");
        }
        let path = resolve_writable_existing_file(&display_path, &root)?;
        let metadata =
            fs::metadata(&path).map_err(|_| anyhow!("File is not readable: {}", path.display()))?;
        if !metadata.is_file() {
            bail!("Path is not a file: {}", path.display());
        }
        if metadata.len() > max_file_bytes {
            bail!("File is too large to patch: {}", path.display());
        }
        let original = fs::read_to_string(&path)
            .map_err(|_| anyhow!("File is not valid UTF-8: {}", path.display()))?;
        let (content, additions, removals) = apply_file_patch(&original, &file_patch)
            .map_err(|error| anyhow!("{}: {error}", display_path))?;
        if content.len() as u64 > max_file_bytes {
            bail!("Patched file would be too large: {}", path.display());
        }
        prepared.push(PreparedPatch {
            path,
            display_path,
            original_content: original,
            content,
            additions,
            removals,
        });
    }

    write_prepared_patches(&prepared)?;

    Ok(ApplyFileDiffSummary {
        files: prepared
            .into_iter()
            .map(|patch| AppliedFileDiff {
                path: patch.display_path,
                additions: patch.additions,
                removals: patch.removals,
                content: patch.content,
            })
            .collect(),
    })
}

fn write_prepared_patches(prepared: &[PreparedPatch]) -> anyhow::Result<()> {
    let mut written: Vec<&PreparedPatch> = Vec::new();
    for patch in prepared {
        if let Err(error) = fs::write(&patch.path, &patch.content) {
            for written_patch in written.into_iter().rev() {
                let _ = fs::write(&written_patch.path, &written_patch.original_content);
            }
            return Err(anyhow!("Failed to write {}: {error}", patch.path.display()));
        }
        written.push(patch);
    }
    Ok(())
}

pub(super) fn apply_file_diff_result_text(summary: &ApplyFileDiffSummary) -> String {
    let paths = summary
        .files
        .iter()
        .take(MAX_LOCAL_DIRECT_PATCH_OUTPUT_PATHS)
        .map(|file| format!("- {} (+{}, -{})", file.path, file.additions, file.removals))
        .collect::<Vec<_>>()
        .join("\n");
    let extra = summary
        .files
        .len()
        .saturating_sub(MAX_LOCAL_DIRECT_PATCH_OUTPUT_PATHS);
    let extra = if extra > 0 {
        format!("\n- ... and {extra} more")
    } else {
        String::new()
    };

    format!(
        "Applied patch successfully.\nFiles changed: {}\nLines added: {}\nLines removed: {}\nChanged files:\n{}{}",
        summary.files.len(),
        summary.total_additions(),
        summary.total_removals(),
        paths,
        extra,
    )
}

fn parse_unified_diff(patch: &str) -> anyhow::Result<Vec<FilePatch>> {
    let lines = patch.lines().collect::<Vec<_>>();
    let mut index = 0;
    let mut files = Vec::new();
    let mut hunk_count = 0;

    while index < lines.len() {
        if lines[index].starts_with("diff --git ") {
            index += 1;
            continue;
        }
        if !lines[index].starts_with("--- ") {
            index += 1;
            continue;
        }

        let old_path = parse_diff_header_path(lines[index], "--- ")?;
        index += 1;
        if index >= lines.len() || !lines[index].starts_with("+++ ") {
            bail!("malformed diff: missing +++ header");
        }
        let new_path = parse_diff_header_path(lines[index], "+++ ")?;
        index += 1;

        let mut hunks = Vec::new();
        while index < lines.len() {
            let line = lines[index];
            if line.starts_with("diff --git ") || line.starts_with("--- ") {
                break;
            }
            if !line.starts_with("@@ ") {
                index += 1;
                continue;
            }
            let (hunk, next_index) = parse_hunk(&lines, index)?;
            hunk_count += 1;
            if hunk_count > MAX_LOCAL_DIRECT_PATCH_HUNKS {
                bail!("patch has too many hunks");
            }
            hunks.push(hunk);
            index = next_index;
        }
        if hunks.is_empty() {
            bail!("file patch has no hunks: {new_path}");
        }
        files.push(FilePatch {
            old_path,
            new_path,
            hunks,
        });
    }

    if files.is_empty() {
        bail!("malformed diff: no file patches found");
    }
    Ok(files)
}

fn parse_diff_header_path(line: &str, prefix: &str) -> anyhow::Result<String> {
    let path = line
        .strip_prefix(prefix)
        .ok_or_else(|| anyhow!("malformed diff header"))?
        .split('\t')
        .next()
        .unwrap_or_default()
        .trim();
    if path.is_empty() {
        bail!("malformed diff header: missing path");
    }
    Ok(path.to_string())
}

fn parse_hunk(lines: &[&str], start: usize) -> anyhow::Result<(Hunk, usize)> {
    let header = lines[start];
    let old_start = parse_hunk_old_start(header)?;
    let mut hunk_lines = Vec::new();
    let mut index = start + 1;
    while index < lines.len() {
        let line = lines[index];
        if line.starts_with("@@ ") || line.starts_with("diff --git ") || line.starts_with("--- ") {
            break;
        }
        if line == r"\ No newline at end of file" {
            index += 1;
            continue;
        }
        let Some((marker, content)) = line.split_at_checked(1) else {
            bail!("malformed hunk line");
        };
        match marker {
            " " => hunk_lines.push(HunkLine::Context(content.to_string())),
            "-" => hunk_lines.push(HunkLine::Remove(content.to_string())),
            "+" => hunk_lines.push(HunkLine::Add(content.to_string())),
            _ => bail!("malformed hunk line: expected context, addition, or removal"),
        }
        index += 1;
    }
    if hunk_lines.is_empty() {
        bail!("hunk has no lines");
    }
    Ok((
        Hunk {
            old_start,
            lines: hunk_lines,
        },
        index,
    ))
}

fn parse_hunk_old_start(header: &str) -> anyhow::Result<usize> {
    let rest = header
        .strip_prefix("@@ -")
        .ok_or_else(|| anyhow!("malformed hunk header"))?;
    let old_range = rest
        .split_once(' ')
        .map(|(range, _)| range)
        .ok_or_else(|| anyhow!("malformed hunk header"))?;
    let start = old_range
        .split_once(',')
        .map(|(start, _)| start)
        .unwrap_or(old_range)
        .parse::<usize>()
        .map_err(|_| anyhow!("malformed hunk header"))?;
    Ok(start.max(1))
}

fn validate_update_only_patch(file_patch: &FilePatch) -> anyhow::Result<()> {
    if file_patch.old_path == "/dev/null" || file_patch.new_path == "/dev/null" {
        bail!("create and delete patches are not supported");
    }
    let old_path = normalize_diff_path(&file_patch.old_path)?;
    let new_path = normalize_diff_path(&file_patch.new_path)?;
    if old_path != new_path {
        bail!("rename patches are not supported");
    }
    Ok(())
}

fn normalize_diff_path(path: &str) -> anyhow::Result<String> {
    let normalized = path
        .strip_prefix("a/")
        .or_else(|| path.strip_prefix("b/"))
        .unwrap_or(path)
        .trim();
    if normalized.is_empty() {
        bail!("diff path cannot be empty");
    }
    if normalized == "/dev/null" {
        return Ok(normalized.to_string());
    }
    let path = Path::new(normalized);
    if path.components().any(|component| {
        matches!(
            component,
            std::path::Component::ParentDir | std::path::Component::Prefix(_)
        )
    }) {
        bail!("diff path escapes the writable root: {normalized}");
    }
    Ok(normalized.to_string())
}

fn writable_root(cwd: Option<&Path>) -> anyhow::Result<PathBuf> {
    let cwd = cwd.ok_or_else(|| anyhow!("apply_file_diff requires a current working directory"))?;
    let root = cwd
        .canonicalize()
        .map_err(|_| anyhow!("Writable root is not readable: {}", cwd.display()))?;
    if !root.is_dir() {
        bail!("Writable root is not a directory: {}", root.display());
    }
    Ok(root)
}

fn resolve_writable_existing_file(path: &str, root: &Path) -> anyhow::Result<PathBuf> {
    let candidate = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        root.join(path)
    };
    let canonical = candidate.canonicalize().map_err(|_| {
        anyhow!(
            "File does not exist for update-only patch: {}",
            candidate.display()
        )
    })?;
    if !canonical.starts_with(root) {
        bail!("Path is outside writable root: {}", candidate.display());
    }
    Ok(canonical)
}

fn apply_file_patch(original: &str, patch: &FilePatch) -> anyhow::Result<(String, usize, usize)> {
    let mut lines = split_preserving_newlines(original);
    let mut offset: isize = 0;
    let mut additions = 0;
    let mut removals = 0;

    for hunk in &patch.hunks {
        let base = hunk.old_start.saturating_sub(1);
        let index = base
            .checked_add_signed(offset)
            .ok_or_else(|| anyhow!("hunk position is invalid"))?;
        let (replacement, consumed, added, removed) = apply_hunk_at(&lines, index, hunk)?;
        lines.splice(index..index + consumed, replacement);
        offset += added as isize - removed as isize;
        additions += added;
        removals += removed;
    }

    Ok((lines.concat(), additions, removals))
}

fn apply_hunk_at(
    lines: &[String],
    start: usize,
    hunk: &Hunk,
) -> anyhow::Result<(Vec<String>, usize, usize, usize)> {
    let mut cursor = start;
    let mut replacement = Vec::new();
    let mut additions = 0;
    let mut removals = 0;

    for line in &hunk.lines {
        match line {
            HunkLine::Context(content) => {
                let current = lines
                    .get(cursor)
                    .ok_or_else(|| anyhow!("hunk context extends past end of file"))?;
                if line_without_newline(current) != content {
                    bail!("hunk context mismatch");
                }
                replacement.push(current.clone());
                cursor += 1;
            }
            HunkLine::Remove(content) => {
                let current = lines
                    .get(cursor)
                    .ok_or_else(|| anyhow!("hunk removal extends past end of file"))?;
                if line_without_newline(current) != content {
                    bail!("hunk removal mismatch");
                }
                cursor += 1;
                removals += 1;
            }
            HunkLine::Add(content) => {
                replacement.push(format!("{content}\n"));
                additions += 1;
            }
        }
    }

    Ok((replacement, cursor - start, additions, removals))
}

fn split_preserving_newlines(content: &str) -> Vec<String> {
    if content.is_empty() {
        return Vec::new();
    }
    let mut lines = content
        .split_inclusive('\n')
        .map(str::to_string)
        .collect::<Vec<_>>();
    if !content.ends_with('\n') {
        if let Some(last) = lines.last_mut() {
            if last.ends_with('\n') {
                return lines;
            }
        }
    }
    lines
}

fn line_without_newline(line: &str) -> &str {
    line.strip_suffix('\n').unwrap_or(line)
}

#[cfg(test)]
mod tests {
    use super::{apply_file_diff_result_text, apply_unified_diff};

    #[test]
    fn applies_update_patch() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file = temp_dir.path().join("sample.txt");
        std::fs::write(&file, "alpha\nbeta\ngamma\n").unwrap();

        let summary = apply_unified_diff(
            concat!(
                "--- a/sample.txt\n",
                "+++ b/sample.txt\n",
                "@@ -1,3 +1,3 @@\n",
                " alpha\n",
                "-beta\n",
                "+BETTA\n",
                " gamma\n",
            ),
            Some(temp_dir.path()),
            1024,
        )
        .unwrap();

        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "alpha\nBETTA\ngamma\n"
        );
        assert_eq!(summary.files[0].path, "sample.txt");
        assert_eq!(summary.files[0].additions, 1);
        assert_eq!(summary.files[0].removals, 1);
        assert!(apply_file_diff_result_text(&summary).contains("Files changed: 1"));
    }

    #[test]
    fn rejects_malformed_diff() {
        let temp_dir = tempfile::tempdir().unwrap();

        let error = apply_unified_diff("not a diff", Some(temp_dir.path()), 1024).unwrap_err();

        assert!(error.to_string().contains("no file patches"));
    }

    #[test]
    fn rejects_path_escape() {
        let temp_dir = tempfile::tempdir().unwrap();

        let error = apply_unified_diff(
            concat!(
                "--- a/../outside.txt\n",
                "+++ b/../outside.txt\n",
                "@@ -1 +1 @@\n",
                "-old\n",
                "+new\n",
            ),
            Some(temp_dir.path()),
            1024,
        )
        .unwrap_err();

        assert!(error.to_string().contains("escapes"));
    }

    #[test]
    fn hunk_mismatch_is_atomic() {
        let temp_dir = tempfile::tempdir().unwrap();
        let first = temp_dir.path().join("first.txt");
        let second = temp_dir.path().join("second.txt");
        std::fs::write(&first, "one\n").unwrap();
        std::fs::write(&second, "two\n").unwrap();

        let error = apply_unified_diff(
            concat!(
                "--- a/first.txt\n",
                "+++ b/first.txt\n",
                "@@ -1 +1 @@\n",
                "-one\n",
                "+ONE\n",
                "--- a/second.txt\n",
                "+++ b/second.txt\n",
                "@@ -1 +1 @@\n",
                "-not-two\n",
                "+TWO\n",
            ),
            Some(temp_dir.path()),
            1024,
        )
        .unwrap_err();

        assert!(error.to_string().contains("hunk removal mismatch"));
        assert_eq!(std::fs::read_to_string(&first).unwrap(), "one\n");
        assert_eq!(std::fs::read_to_string(&second).unwrap(), "two\n");
    }

    #[test]
    fn rejects_create_delete_and_rename() {
        let temp_dir = tempfile::tempdir().unwrap();

        let create_error = apply_unified_diff(
            concat!(
                "--- /dev/null\n",
                "+++ b/new.txt\n",
                "@@ -0,0 +1 @@\n",
                "+new\n",
            ),
            Some(temp_dir.path()),
            1024,
        )
        .unwrap_err();
        assert!(create_error.to_string().contains("not supported"));

        std::fs::write(temp_dir.path().join("old.txt"), "old\n").unwrap();
        let rename_error = apply_unified_diff(
            concat!(
                "--- a/old.txt\n",
                "+++ b/new.txt\n",
                "@@ -1 +1 @@\n",
                "-old\n",
                "+new\n",
            ),
            Some(temp_dir.path()),
            1024,
        )
        .unwrap_err();
        assert!(rename_error.to_string().contains("rename"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let temp_dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_file = outside.path().join("outside.txt");
        std::fs::write(&outside_file, "old\n").unwrap();
        symlink(&outside_file, temp_dir.path().join("link.txt")).unwrap();

        let error = apply_unified_diff(
            concat!(
                "--- a/link.txt\n",
                "+++ b/link.txt\n",
                "@@ -1 +1 @@\n",
                "-old\n",
                "+new\n",
            ),
            Some(temp_dir.path()),
            1024,
        )
        .unwrap_err();

        assert!(error.to_string().contains("outside writable root"));
        assert_eq!(std::fs::read_to_string(&outside_file).unwrap(), "old\n");
    }
}
