//! Text splitter for Soprano TTS
//!
//! Splits text into chunks of desired length while keeping sentences intact.
//! Adapted from the Python implementation in soprano/utils/text_splitter.py

use regex::Regex;
use std::sync::OnceLock;

/// Split text into chunks while trying to keep sentences intact
pub fn split_and_recombine_text(
    text: &str,
    desired_length: usize,
    max_length: usize,
) -> Vec<String> {
    // Normalize text
    static NEWLINES_RE: OnceLock<Regex> = OnceLock::new();
    let newlines_re = NEWLINES_RE.get_or_init(|| Regex::new(r"\n\n+").unwrap());
    let text = newlines_re.replace_all(text, "\n");

    static WHITESPACE_RE: OnceLock<Regex> = OnceLock::new();
    let whitespace_re = WHITESPACE_RE.get_or_init(|| Regex::new(r"\s+").unwrap());
    let text = whitespace_re.replace_all(&text, " ");

    static QUOTES_RE: OnceLock<Regex> = OnceLock::new();
    let quotes_re = QUOTES_RE.get_or_init(|| Regex::new("[\u{201C}\u{201D}]").unwrap());
    let text = quotes_re.replace_all(&text, "\"");

    let mut result = Vec::new();
    let mut in_quote = false;
    let mut current = String::new();
    let mut split_pos: Vec<usize> = Vec::new();
    let mut pos: isize = -1;
    let end_pos = text.len() as isize - 1;

    fn seek(
        text: &str,
        pos: &mut isize,
        current: &mut String,
        in_quote: &mut bool,
        delta: isize,
    ) -> char {
        let is_neg = delta < 0;
        for _ in 0..delta.abs() {
            if is_neg {
                *pos -= 1;
                current.pop();
            } else {
                *pos += 1;
                if let Some(c) = text.chars().nth(*pos as usize) {
                    current.push(c);
                }
            }
            if let Some(c) = text.chars().nth(*pos as usize) {
                if c == '"' {
                    *in_quote = !*in_quote;
                }
            }
        }
        text.chars().nth(*pos as usize).unwrap_or(' ')
    }

    fn peek(text: &str, pos: isize, delta: isize) -> Option<char> {
        let p = pos + delta;
        if p >= 0 && p < text.len() as isize {
            text.chars().nth(p as usize)
        } else {
            None
        }
    }

    fn commit(result: &mut Vec<String>, current: &mut String, split_pos: &mut Vec<usize>) {
        result.push(current.trim().to_string());
        current.clear();
        split_pos.clear();
    }

    while pos < end_pos {
        let c = seek(&text, &mut pos, &mut current, &mut in_quote, 1);

        // Check for max length
        if current.len() >= max_length {
            if !split_pos.is_empty() && current.len() > desired_length / 2 {
                // Seek back to last split
                let d = pos - split_pos[split_pos.len() - 1] as isize;
                seek(&text, &mut pos, &mut current, &mut in_quote, -d);
            } else {
                // No full sentences, seek back to word boundary
                while !matches!(c, '!' | '?' | '.' | '\n' | ' ')
                    && pos > 0
                    && current.len() > desired_length
                {
                    seek(&text, &mut pos, &mut current, &mut in_quote, -1);
                }
            }
            commit(&mut result, &mut current, &mut split_pos);
        }
        // Check for sentence boundaries
        else if !in_quote
            && (matches!(c, '!' | '?' | '\n')
                || (c == '.' && matches!(peek(&text, pos, 1), Some('\n') | Some(' '))))
        {
            // Seek forward for consecutive boundary markers
            while pos < text.len() as isize - 1
                && current.len() < max_length
                && matches!(peek(&text, pos, 1), Some('!') | Some('?') | Some('.'))
            {
                seek(&text, &mut pos, &mut current, &mut in_quote, 1);
            }
            split_pos.push(pos as usize);
            if current.len() >= desired_length {
                commit(&mut result, &mut current, &mut split_pos);
            }
        }
        // Handle end of quote
        else if in_quote
            && peek(&text, pos, 1) == Some('"')
            && matches!(peek(&text, pos, 2), Some('\n') | Some(' '))
        {
            seek(&text, &mut pos, &mut current, &mut in_quote, 2);
            split_pos.push(pos as usize);
        }
    }

    if !current.is_empty() {
        result.push(current.trim().to_string());
    }

    // Clean up: remove lines with only whitespace or punctuation
    result
        .into_iter()
        .filter(|s| {
            let trimmed = s.trim();
            !trimmed.is_empty() && !Regex::new(r"^[\s.,;:!?]*$").unwrap().is_match(trimmed)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_split() {
        let text = "Hello world. This is a test. Another sentence here.";
        let chunks = split_and_recombine_text(text, 1, 200);
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].contains("Hello world"));
        assert!(chunks[1].contains("This is a test"));
        assert!(chunks[2].contains("Another sentence"));
    }

    #[test]
    fn test_long_text() {
        let text = "This is a very long text that should be split into multiple chunks. \\
                   Each chunk should be a reasonable length. We want to keep sentences intact. \\
                   This is important for natural speech synthesis.";
        let chunks = split_and_recombine_text(text, 80, 150);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.len() <= 150);
        }
    }

    #[test]
    fn test_with_quotes() {
        let text = r#"He said "Hello world." Then he left."#;
        let chunks = split_and_recombine_text(text, 50, 200);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_empty_text() {
        let text = "";
        let chunks = split_and_recombine_text(text, 50, 200);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_only_punctuation() {
        let text = "...!!!???";
        let chunks = split_and_recombine_text(text, 50, 200);
        assert!(chunks.is_empty());
    }
}
