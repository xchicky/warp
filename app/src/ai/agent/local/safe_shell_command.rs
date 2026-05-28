use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};

const DIRECTLY_ALLOWED_COMMANDS: &[&str] = &[
    "pwd", "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "echo", "which", "type", "ps",
    "df", "du", "file", "uname", "date", "whoami", "id", "stat", "tree",
];

const DISALLOWED_TOP_LEVEL_COMMANDS: &[&str] = &[
    "eval", "exec", "source", ".", "sh", "bash", "zsh", "fish", "pwsh", "ssh", "scp", "rsync",
    "curl", "wget", "nc", "netcat", "telnet", "rm", "mv", "cp", "touch", "mkdir", "rmdir", "chmod",
    "chown", "sudo", "tee", "apt", "apt-get", "brew", "cargo", "npm", "pnpm", "yarn", "python",
    "python3", "pip", "pip3", "vim", "vi", "nano", "emacs", "sed",
];

const FIND_DENIED_OPTIONS: &[&str] = &[
    "-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls",
];

const GIT_ALLOWED_SUBCOMMANDS: &[&str] = &[
    "status",
    "diff",
    "log",
    "show",
    "rev-parse",
    "ls-files",
    "grep",
];

const GIT_DENIED_OPTIONS: &[&str] = &[
    "--output",
    "--ext-diff",
    "--external-diff",
    "--exec-path",
    "-c",
    "--config-env",
];

static LOCAL_AUTOEXECUTE_SAFE_TOOL_CALLS: LazyLock<Mutex<HashMap<String, String>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub(super) fn is_local_autoexecute_safe_command(command: &str) -> bool {
    if command.trim().is_empty() || contains_forbidden_shell_syntax(command) {
        return false;
    }
    let Ok(tokens) = shell_words::split(command) else {
        return false;
    };
    classify_tokens(&tokens)
}

pub(crate) fn register_local_autoexecute_safe_tool_call(tool_call_id: &str, command: &str) {
    if let Ok(mut tool_calls) = LOCAL_AUTOEXECUTE_SAFE_TOOL_CALLS.lock() {
        tool_calls.insert(tool_call_id.to_string(), command.to_string());
    }
}

pub(crate) fn is_local_autoexecute_safe_tool_call(tool_call_id: &str, command: &str) -> bool {
    LOCAL_AUTOEXECUTE_SAFE_TOOL_CALLS
        .lock()
        .is_ok_and(|tool_calls| {
            tool_calls
                .get(tool_call_id)
                .is_some_and(|stored| stored == command)
        })
}

fn contains_forbidden_shell_syntax(command: &str) -> bool {
    let mut chars = command.chars().peekable();
    let mut in_single_quote = false;
    while let Some(ch) = chars.next() {
        if ch == '\'' {
            in_single_quote = !in_single_quote;
            continue;
        }
        if in_single_quote {
            continue;
        }
        match ch {
            '\n' | ';' | '`' | '|' | '&' | '>' | '<' => return true,
            '$' if chars.peek() == Some(&'(') => return true,
            _ => {}
        }
    }
    in_single_quote
}

fn classify_tokens(tokens: &[String]) -> bool {
    let Some(command) = tokens.first().map(String::as_str) else {
        return false;
    };
    if DISALLOWED_TOP_LEVEL_COMMANDS.contains(&command) {
        return false;
    }
    match command {
        "command" => matches!(tokens.get(1).map(String::as_str), Some("-v")) && tokens.len() >= 3,
        "git" => classify_git(tokens),
        "top" => classify_top(tokens),
        "find" => !tokens
            .iter()
            .any(|token| FIND_DENIED_OPTIONS.contains(&token.as_str())),
        "grep" | "rg" => classify_grep_like(tokens),
        allowed => DIRECTLY_ALLOWED_COMMANDS.contains(&allowed),
    }
}

fn classify_git(tokens: &[String]) -> bool {
    let Some(subcommand) = tokens.get(1).map(String::as_str) else {
        return false;
    };
    if !GIT_ALLOWED_SUBCOMMANDS.contains(&subcommand) {
        return false;
    }
    !tokens.iter().any(|token| {
        let token = token.as_str();
        GIT_DENIED_OPTIONS.contains(&token)
            || GIT_DENIED_OPTIONS
                .iter()
                .any(|option| token.starts_with(&format!("{option}=")))
    })
}

fn classify_grep_like(tokens: &[String]) -> bool {
    !tokens.iter().any(|token| {
        matches!(token.as_str(), "--output" | "--pre" | "--pre-glob")
            || token.starts_with("--output=")
            || token.starts_with("--pre=")
            || token.starts_with("--pre-glob=")
    })
}

fn classify_top(tokens: &[String]) -> bool {
    let has_macos_bound = tokens
        .windows(2)
        .any(|window| window[0] == "-l" && positive_integer(&window[1]));
    let has_linux_batch = tokens.iter().any(|token| token == "-b")
        && tokens
            .windows(2)
            .any(|window| window[0] == "-n" && positive_integer(&window[1]));
    has_macos_bound || has_linux_batch
}

fn positive_integer(value: &str) -> bool {
    value.parse::<u64>().is_ok_and(|value| value > 0)
}

#[cfg(test)]
mod tests {
    use super::is_local_autoexecute_safe_command;

    #[test]
    fn local_autoexecute_classifier_allows_safe_commands() {
        for command in [
            "pwd",
            "ls -la",
            "cat Cargo.toml",
            "head -n 20 app/src/lib.rs",
            "tail -n 5 README.md",
            "wc -l Cargo.toml",
            "grep LocalAgent app/src/lib.rs",
            "rg LocalAgent",
            "find . -name '*.rs'",
            "echo $HOME",
            "which git",
            "type cargo",
            "command -v git",
            "ps aux",
            "df -h",
            "du -sh .",
            "file Cargo.toml",
            "uname -a",
            "date",
            "whoami",
            "id",
            "stat Cargo.toml",
            "tree -L 2 .",
            "git status --short",
            "git diff -- Cargo.toml",
            "git log --oneline -5",
            "git show HEAD",
            "git rev-parse --show-toplevel",
            "git ls-files",
            "git grep LocalAgent",
            "top -l 1 -stats pid,command",
            "top -b -n 1",
        ] {
            assert!(
                is_local_autoexecute_safe_command(command),
                "expected safe: {command}"
            );
        }
    }

    #[test]
    fn local_autoexecute_classifier_rejects_forbidden_shell_syntax() {
        for command in [
            "ls && rm -rf /",
            "ls || rm -rf /",
            "cat Cargo.toml; rm x",
            "pwd\nrm x",
            "grep foo file | tee out",
            "sleep 1 &",
            "grep foo file > out",
            "grep foo file 2> out",
            "cat < secret",
            "cat <<EOF",
            "cat <<< foo",
            "echo $(rm x)",
            "echo `rm x`",
            "cat <(echo x)",
            "cat >(tee x)",
        ] {
            assert!(
                !is_local_autoexecute_safe_command(command),
                "expected unsafe: {command}"
            );
        }
        assert!(is_local_autoexecute_safe_command("grep ';' Cargo.toml"));
    }

    #[test]
    fn local_autoexecute_classifier_rejects_command_specific_risks() {
        for command in [
            "find . -delete",
            "find . -exec rm {} +",
            "find . -execdir pwd \\;",
            "find . -ok rm {} \\;",
            "find . -okdir rm {} \\;",
            "find . -fprint out",
            "find . -fprintf out '%p'",
            "find . -fls out",
            "rg foo --output out",
            "rg foo --output=out",
            "rg --pre ./filter foo",
            "rg --pre=./filter foo",
            "rg --pre-glob '*.md' foo",
            "grep --output out foo file",
            "git diff --output out",
            "git diff --ext-diff",
            "git show --external-diff",
            "git log --exec-path=/tmp/git",
            "git -c core.pager=cat status",
            "git status -c core.pager=cat",
            "git log --config-env=foo=bar",
        ] {
            assert!(
                !is_local_autoexecute_safe_command(command),
                "expected unsafe: {command}"
            );
        }
    }

    #[test]
    fn local_autoexecute_classifier_rejects_unknown_or_mutating_commands() {
        for command in [
            "top",
            "top -l 0",
            "cargo test",
            "python script.py",
            "npm install",
            "rm file",
            "sed -i 's/a/b/' file",
            "curl https://example.com",
            "bash -lc pwd",
            "command ls",
        ] {
            assert!(
                !is_local_autoexecute_safe_command(command),
                "expected unsafe: {command}"
            );
        }
    }
}
