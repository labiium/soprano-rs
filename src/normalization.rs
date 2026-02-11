//! Text normalization for Soprano TTS
//!
//! Normalizes input text to a format that Soprano recognizes.
//! Adapted from the Python implementation in soprano/utils/text_normalizer.py

use regex::Regex;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

/// Expands abbreviations
fn expand_abbreviations(text: &str) -> String {
    static ABBREVIATIONS: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    let abbrs = ABBREVIATIONS.get_or_init(|| {
        vec![
            (Regex::new(r"(?i)\bmrs\.").unwrap(), "misess"),
            (Regex::new(r"(?i)\bms\.").unwrap(), "miss"),
            (Regex::new(r"(?i)\bmr\.").unwrap(), "mister"),
            (Regex::new(r"(?i)\bdr\.").unwrap(), "doctor"),
            (Regex::new(r"(?i)\bst\.").unwrap(), "saint"),
            (Regex::new(r"(?i)\bco\.").unwrap(), "company"),
            (Regex::new(r"(?i)\bjr\.").unwrap(), "junior"),
            (Regex::new(r"(?i)\bmaj\.").unwrap(), "major"),
            (Regex::new(r"(?i)\bgen\.").unwrap(), "general"),
            (Regex::new(r"(?i)\bdrs\.").unwrap(), "doctors"),
            (Regex::new(r"(?i)\brev\.").unwrap(), "reverend"),
            (Regex::new(r"(?i)\blt\.").unwrap(), "lieutenant"),
            (Regex::new(r"(?i)\bhon\.").unwrap(), "honorable"),
            (Regex::new(r"(?i)\bsgt\.").unwrap(), "sergeant"),
            (Regex::new(r"(?i)\bcapt\.").unwrap(), "captain"),
            (Regex::new(r"(?i)\besq\.").unwrap(), "esquire"),
            (Regex::new(r"(?i)\bltd\.").unwrap(), "limited"),
            (Regex::new(r"(?i)\bcol\.").unwrap(), "colonel"),
            (Regex::new(r"(?i)\bft\.").unwrap(), "fort"),
            // Cased abbreviations
            (Regex::new(r"\bHz\b").unwrap(), "hertz"),
            (Regex::new(r"\bkHz\b").unwrap(), "kilohertz"),
            (Regex::new(r"\bKBs\b").unwrap(), "kilobytes"),
            (Regex::new(r"\bKB\b").unwrap(), "kilobyte"),
            (Regex::new(r"\bMBs\b").unwrap(), "megabytes"),
            (Regex::new(r"\bMB\b").unwrap(), "megabyte"),
            (Regex::new(r"\bGBs\b").unwrap(), "gigabytes"),
            (Regex::new(r"\bGB\b").unwrap(), "gigabyte"),
            (Regex::new(r"\bTBs\b").unwrap(), "terabytes"),
            (Regex::new(r"\bTB\b").unwrap(), "terabyte"),
            (Regex::new(r"\bAPIs\b").unwrap(), "a p i's"),
            (Regex::new(r"\bAPI\b").unwrap(), "a p i"),
            (Regex::new(r"\bCLIs\b").unwrap(), "c l i's"),
            (Regex::new(r"\bCLI\b").unwrap(), "c l i"),
            (Regex::new(r"\bCPUs\b").unwrap(), "c p u's"),
            (Regex::new(r"\bCPU\b").unwrap(), "c p u"),
            (Regex::new(r"\bGPUs\b").unwrap(), "g p u's"),
            (Regex::new(r"\bGPU\b").unwrap(), "g p u"),
            (Regex::new(r"\bAve\b").unwrap(), "avenue"),
            (Regex::new(r"\betc\b").unwrap(), "et cetera"),
            (Regex::new(r"\bMon\b").unwrap(), "monday"),
            (Regex::new(r"\bTues\b").unwrap(), "tuesday"),
            (Regex::new(r"\bWed\b").unwrap(), "wednesday"),
            (Regex::new(r"\bThurs\b").unwrap(), "thursday"),
            (Regex::new(r"\bFri\b").unwrap(), "friday"),
            (Regex::new(r"\bSat\b").unwrap(), "saturday"),
            (Regex::new(r"\bJan\b").unwrap(), "january"),
            (Regex::new(r"\bFeb\b").unwrap(), "february"),
            (Regex::new(r"\bMar\b").unwrap(), "march"),
            (Regex::new(r"\bApr\b").unwrap(), "april"),
            (Regex::new(r"\bAug\b").unwrap(), "august"),
            (Regex::new(r"\bSept\b").unwrap(), "september"),
            (Regex::new(r"\bOct\b").unwrap(), "october"),
            (Regex::new(r"\bNov\b").unwrap(), "november"),
            (Regex::new(r"\bDec\b").unwrap(), "december"),
            (Regex::new(r"and/or").unwrap(), "and or"),
        ]
    });

    let mut result = text.to_string();
    for (regex, replacement) in abbrs {
        result = regex.replace_all(&result, *replacement).to_string();
    }
    result
}

/// Normalize numbers to words
fn normalize_numbers(text: &str) -> String {
    let mut result = text.to_string();

    // Phone numbers: 123-456-7890 or (123) 456-7890
    static PHONE_RE: OnceLock<Regex> = OnceLock::new();
    let phone_re =
        PHONE_RE.get_or_init(|| Regex::new(r"\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})").unwrap());
    result = phone_re
        .replace_all(&result, |caps: &regex::Captures| {
            let part1: Vec<String> = caps[1].chars().map(|c| c.to_string()).collect();
            let part2: Vec<String> = caps[2].chars().map(|c| c.to_string()).collect();
            let part3: Vec<String> = caps[3].chars().map(|c| c.to_string()).collect();
            format!(
                "{} {} {}",
                part1.join(", "),
                part2.join(", "),
                part3.join(", ")
            )
        })
        .to_string();

    // Times: 12:00, 8:30
    static TIME_RE: OnceLock<Regex> = OnceLock::new();
    let time_re = TIME_RE.get_or_init(|| Regex::new(r"(\d{1,2}):(\d{2})(?::(\d{2}))?").unwrap());
    result = time_re
        .replace_all(&result, |caps: &regex::Captures| {
            let hours = &caps[1];
            let minutes = &caps[2];
            if let Some(seconds) = caps.get(3) {
                format!("{} {} {}", hours, minutes, seconds.as_str())
            } else if minutes == "00" {
                if hours == "0" || hours == "00" {
                    "0".to_string()
                } else {
                    format!("{} o'clock", hours)
                }
            } else if minutes.starts_with('0') {
                format!("{} oh {}", hours, minutes.trim_start_matches('0'))
            } else {
                format!("{} {}", hours, minutes)
            }
        })
        .to_string();

    // Currency: $X.XX
    static DOLLARS_RE: OnceLock<Regex> = OnceLock::new();
    let dollars_re = DOLLARS_RE.get_or_init(|| Regex::new(r"\$([\d.,]+)").unwrap());
    result = dollars_re
        .replace_all(&result, |caps: &regex::Captures| {
            let amount = &caps[1];
            if amount.contains('.') {
                let parts: Vec<&str> = amount.split('.').collect();
                let dollars: i64 = parts[0].parse().unwrap_or(0);
                let cents: i64 = parts[1].parse().unwrap_or(0);
                if dollars > 0 && cents > 0 {
                    format!("{} dollars, {} cents", dollars, cents)
                } else if dollars > 0 {
                    format!("{} dollars", dollars)
                } else {
                    format!("{} cents", cents)
                }
            } else {
                format!("{} dollars", amount)
            }
        })
        .to_string();

    // Pounds: £X
    static POUNDS_RE: OnceLock<Regex> = OnceLock::new();
    let pounds_re = POUNDS_RE.get_or_init(|| Regex::new(r"£([\d,]+)").unwrap());
    result = pounds_re
        .replace_all(&result, |caps: &regex::Captures| {
            format!("{} pounds", &caps[1])
        })
        .to_string();

    // Number suffix: 1K, 1M, 1B, 1T
    static NUM_SUFFIX_RE: OnceLock<Regex> = OnceLock::new();
    let num_suffix_re = NUM_SUFFIX_RE.get_or_init(|| Regex::new(r"(?i)\b(\d+)([KMBT])\b").unwrap());
    result = num_suffix_re
        .replace_all(&result, |caps: &regex::Captures| {
            let num = &caps[1];
            let suffix = caps[2].to_uppercase();
            let word = match suffix.as_str() {
                "K" => "thousand",
                "M" => "million",
                "B" => "billion",
                "T" => "trillion",
                _ => "",
            };
            format!("{} {}", num, word)
        })
        .to_string();

    // Remove commas from numbers
    static COMMA_NUM_RE: OnceLock<Regex> = OnceLock::new();
    let comma_num_re = COMMA_NUM_RE.get_or_init(|| Regex::new(r"(\d[\d,]+\d)").unwrap());
    result = comma_num_re
        .replace_all(&result, |caps: &regex::Captures| caps[1].replace(',', ""))
        .to_string();

    // Dates: 1/1/2025 or 1-1-2025
    static DATE_RE: OnceLock<Regex> = OnceLock::new();
    let date_re =
        DATE_RE.get_or_init(|| Regex::new(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})").unwrap());
    result = date_re
        .replace_all(&result, |caps: &regex::Captures| {
            format!("{} dash {} dash {}", &caps[1], &caps[2], &caps[3])
        })
        .to_string();

    // Ordinals: 1st, 2nd, 3rd, 4th
    static ORDINAL_RE: OnceLock<Regex> = OnceLock::new();
    let ordinal_re = ORDINAL_RE.get_or_init(|| Regex::new(r"\b(\d+)(st|nd|rd|th)\b").unwrap());
    result = ordinal_re
        .replace_all(&result, |caps: &regex::Captures| {
            number_to_words(caps[1].parse().unwrap_or(0))
        })
        .to_string();

    // Decimals: 1.17.1.1
    static DECIMAL_RE: OnceLock<Regex> = OnceLock::new();
    let decimal_re = DECIMAL_RE.get_or_init(|| Regex::new(r"(\d+)\.(\d+)").unwrap());
    result = decimal_re
        .replace_all(&result, |caps: &regex::Captures| {
            let int_part = &caps[1];
            let frac_part = &caps[2];
            let digits: String = frac_part.chars().map(|c| format!("{} ", c)).collect();
            format!("{} point {}", int_part, digits.trim())
        })
        .to_string();

    // Fractions: 1/2
    static FRACTION_RE: OnceLock<Regex> = OnceLock::new();
    let fraction_re = FRACTION_RE.get_or_init(|| Regex::new(r"(\d+)/(\d+)").unwrap());
    result = fraction_re
        .replace_all(&result, |caps: &regex::Captures| {
            format!("{} over {}", &caps[1], &caps[2])
        })
        .to_string();

    // Simple numbers
    static NUMBER_RE: OnceLock<Regex> = OnceLock::new();
    let number_re = NUMBER_RE.get_or_init(|| Regex::new(r"\b\d+\b").unwrap());
    result = number_re
        .replace_all(&result, |caps: &regex::Captures| {
            let num: u64 = caps[0].parse().unwrap_or(0);
            number_to_words(num)
        })
        .to_string();

    result
}

/// Convert a number to words
fn number_to_words(n: u64) -> String {
    if n == 0 {
        return "zero".to_string();
    }

    let units = [
        "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ];
    let teens = [
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    let tens = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];

    fn convert_less_than_thousand(n: u64, units: &[&str], teens: &[&str], tens: &[&str]) -> String {
        if n == 0 {
            return String::new();
        }
        if n < 10 {
            return units[n as usize].to_string();
        }
        if n < 20 {
            return teens[(n - 10) as usize].to_string();
        }
        if n < 100 {
            let ten = n / 10;
            let rem = n % 10;
            if rem == 0 {
                return tens[ten as usize].to_string();
            } else {
                return format!("{} {}", tens[ten as usize], units[rem as usize]);
            }
        }
        let hundred = n / 100;
        let rem = n % 100;
        if rem == 0 {
            format!("{} hundred", units[hundred as usize])
        } else {
            format!(
                "{} hundred {}",
                units[hundred as usize],
                convert_less_than_thousand(rem, units, teens, tens)
            )
        }
    }

    if n < 1000 {
        return convert_less_than_thousand(n, &units, &teens, &tens);
    }

    let mut parts = Vec::new();
    let mut remaining = n;

    let billions = remaining / 1_000_000_000;
    remaining %= 1_000_000_000;
    if billions > 0 {
        parts.push(format!(
            "{} billion",
            convert_less_than_thousand(billions, &units, &teens, &tens)
        ));
    }

    let millions = remaining / 1_000_000;
    remaining %= 1_000_000;
    if millions > 0 {
        parts.push(format!(
            "{} million",
            convert_less_than_thousand(millions, &units, &teens, &tens)
        ));
    }

    let thousands = remaining / 1_000;
    remaining %= 1_000;
    if thousands > 0 {
        parts.push(format!(
            "{} thousand",
            convert_less_than_thousand(thousands, &units, &teens, &tens)
        ));
    }

    if remaining > 0 {
        parts.push(convert_less_than_thousand(remaining, &units, &teens, &tens));
    }

    parts.join(" ")
}

/// Normalize special characters
fn normalize_special(text: &str) -> String {
    static SPECIAL_CHARS: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    let specials = SPECIAL_CHARS.get_or_init(|| {
        vec![
            // Links
            (
                Regex::new(r"https?://").unwrap(),
                "h t t p s colon slash slash ",
            ),
            // Dashes
            (Regex::new(r"(.) - (.)").unwrap(), "$1, $2"),
            // Dots between letters
            (Regex::new(r"([A-Z])\.([A-Z])").unwrap(), "$1 dot $2"),
            // Parentheses
            (Regex::new(r"[\(\[\{]").unwrap(), ", "),
            (Regex::new(r"[\)\]\}][^$.!?]").unwrap(), ", "),
            (Regex::new(r"[\)\]\}]").unwrap(), ""),
        ]
    });

    let mut result = text.to_string();
    for (regex, replacement) in specials {
        result = regex.replace_all(&result, *replacement).to_string();
    }

    // Handle special characters
    result = result.replace('@', " at ");
    result = result.replace('&', " and ");
    result = result.replace('%', " percent ");
    result = result.replace(':', ".");
    result = result.replace(';', ",");
    result = result.replace('+', " plus ");
    result = result.replace('\\', " backslash ");
    result = result.replace('~', " about ");
    result = result.replace('<', " less than ");
    result = result.replace('>', " greater than ");
    result = result.replace('=', " equals ");
    result = result.replace('/', " slash ");
    result = result.replace('_', " ");
    result = result.replace('*', " ");

    result
}

/// Expand special characters
fn expand_special_characters(text: &str) -> String {
    // Pre-unicode special characters
    static PREUNICODE_SPECIAL: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    let preunicode = PREUNICODE_SPECIAL.get_or_init(|| vec![(Regex::new(r"—").unwrap(), " - ")]);

    let mut result = text.to_string();
    for (regex, replacement) in preunicode {
        result = regex.replace_all(&result, *replacement).to_string();
    }
    result
}

/// Normalize CamelCase
fn normalize_mixedcase(text: &str) -> String {
    static CAMEL_RE: OnceLock<Regex> = OnceLock::new();
    let camel_re = CAMEL_RE.get_or_init(|| Regex::new(r"\b([A-Z][a-z]*)+\b").unwrap());

    camel_re
        .replace_all(text, |caps: &regex::Captures| {
            let word = &caps[0];
            // Check if all uppercase
            if word.chars().all(|c| !c.is_alphabetic() || c.is_uppercase()) {
                return word.to_string();
            }
            // Check if plural uppercase
            if word.len() > 1
                && word[..word.len() - 1].chars().all(|c| c.is_uppercase())
                && word.ends_with('s')
            {
                return format!("{}'s", &word[..word.len() - 1]);
            }
            // Split CamelCase
            let mut result = String::new();
            for (i, c) in word.chars().enumerate() {
                if i > 0 && c.is_uppercase() {
                    result.push(' ');
                }
                result.push(c);
            }
            result
        })
        .to_string()
}

/// Normalize newlines and add periods
fn normalize_newlines(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    let processed: Vec<String> = lines
        .iter()
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return String::new();
            }
            if !trimmed.ends_with(['.', '!', '?']) {
                return format!("{}.", trimmed);
            }
            trimmed.to_string()
        })
        .filter(|s| !s.is_empty())
        .collect();
    processed.join(" ")
}

/// Remove unknown characters
fn remove_unknown_characters(text: &str) -> String {
    text.chars()
        .filter(|c| {
            c.is_ascii_alphabetic() || c.is_ascii_digit() || " !$%&'*+,-./<>?_".contains(*c)
        })
        .filter(|c| !"<>/_+".contains(*c))
        .collect()
}

/// Collapse whitespace
fn collapse_whitespace(text: &str) -> String {
    static WHITESPACE_RE: OnceLock<Regex> = OnceLock::new();
    let ws_re = WHITESPACE_RE.get_or_init(|| Regex::new(r"\s+").unwrap());

    let mut result = ws_re.replace_all(text, " ").to_string();

    // Remove space before punctuation
    static SPACE_PUNCT_RE: OnceLock<Regex> = OnceLock::new();
    let space_punct_re = SPACE_PUNCT_RE.get_or_init(|| Regex::new(r" ([.?!,])").unwrap());
    result = space_punct_re.replace_all(&result, "$1").to_string();

    result.trim().to_string()
}

/// Deduplicate punctuation
fn dedup_punctuation(text: &str) -> String {
    static ELLIPSIS_RE: OnceLock<Regex> = OnceLock::new();
    let ellipsis_re = ELLIPSIS_RE.get_or_init(|| Regex::new(r"\.{3,}").unwrap());

    let mut result = text.to_string();
    result = ellipsis_re.replace_all(&result, "[ELLIPSIS]").to_string();
    result = result.replace(",,+", ",");

    // Handle multiple periods
    static MULTI_PERIOD_RE: OnceLock<Regex> = OnceLock::new();
    let multi_period_re = MULTI_PERIOD_RE.get_or_init(|| Regex::new(r"[.,]*\.[.,]*").unwrap());
    result = multi_period_re.replace_all(&result, ".").to_string();

    // Handle multiple exclamation marks
    static MULTI_EXCL_RE: OnceLock<Regex> = OnceLock::new();
    let multi_excl_re = MULTI_EXCL_RE.get_or_init(|| Regex::new(r"[.,!]*![.,!]*").unwrap());
    result = multi_excl_re.replace_all(&result, "!").to_string();

    // Handle multiple question marks
    static MULTI_QUEST_RE: OnceLock<Regex> = OnceLock::new();
    let multi_quest_re = MULTI_QUEST_RE.get_or_init(|| Regex::new(r"[.,!?]*\?[.,!?]*").unwrap());
    result = multi_quest_re.replace_all(&result, "?").to_string();

    result = result.replace("[ELLIPSIS]", "...");
    result
}

/// Collapse triple letters
fn collapse_triple_letters(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        result.push(c);

        // Check if we have 3+ of the same character in a row
        if c.is_alphabetic() {
            let mut count = 1;
            while chars.peek() == Some(&c) {
                chars.next();
                count += 1;
            }
            // Keep at most 2 consecutive same letters
            if count >= 2 {
                result.push(c);
            }
        }
    }

    result
}

/// Main clean_text function
pub fn clean_text(text: &str) -> String {
    let mut result = text.to_string();

    // Step 1: Expand pre-unicode special characters
    result = expand_special_characters(&result);

    // Step 2: Convert to ASCII
    result = result.nfc().collect::<String>();
    result = result
        .chars()
        .map(|c| if c.is_ascii() { c } else { ' ' })
        .collect();

    // Step 3: Normalize newlines
    result = normalize_newlines(&result);

    // Step 4: Normalize numbers
    result = normalize_numbers(&result);

    // Step 5: Normalize special characters
    result = normalize_special(&result);

    // Step 6: Expand abbreviations
    result = expand_abbreviations(&result);

    // Step 7: Normalize mixed case
    result = normalize_mixedcase(&result);

    // Step 8: Expand special characters
    result = expand_special_characters(&result);

    // Step 9: Lowercase
    result = result.to_lowercase();

    // Step 10: Remove unknown characters
    result = remove_unknown_characters(&result);

    // Step 11: Collapse whitespace
    result = collapse_whitespace(&result);

    // Step 12: Deduplicate punctuation
    result = dedup_punctuation(&result);

    // Step 13: Collapse triple letters
    result = collapse_triple_letters(&result);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_text_basic() {
        let text = "Hello world.";
        assert_eq!(clean_text(text), "hello world.");
    }

    #[test]
    fn test_expand_abbreviations() {
        assert_eq!(expand_abbreviations("Dr. Smith"), "doctor Smith");
        assert_eq!(expand_abbreviations("Mr. Jones"), "mister Jones");
        assert_eq!(expand_abbreviations("CPU"), "c p u");
        assert_eq!(expand_abbreviations("API"), "a p i");
    }

    #[test]
    fn test_normalize_numbers() {
        assert!(normalize_numbers("123").contains("one"));
        assert!(normalize_numbers("1000").contains("thousand"));
        assert!(normalize_numbers("$10").contains("dollars"));
    }

    #[test]
    fn test_number_to_words() {
        assert_eq!(number_to_words(0), "zero");
        assert_eq!(number_to_words(1), "one");
        assert_eq!(number_to_words(15), "fifteen");
        assert_eq!(number_to_words(42), "forty two");
        assert_eq!(number_to_words(100), "one hundred");
        assert_eq!(
            number_to_words(1234),
            "one thousand two hundred thirty four"
        );
    }

    #[test]
    fn test_collapse_whitespace() {
        assert_eq!(collapse_whitespace("hello   world"), "hello world");
        assert_eq!(collapse_whitespace("hello . world"), "hello. world");
    }

    #[test]
    fn test_dedup_punctuation() {
        assert_eq!(dedup_punctuation("hello...."), "hello...");
        assert_eq!(dedup_punctuation("hello!!!"), "hello!");
    }

    #[test]
    fn test_collapse_triple_letters() {
        assert_eq!(collapse_triple_letters("sooooo"), "soo");
        assert_eq!(collapse_triple_letters("hello"), "hello");
    }

    #[test]
    fn test_full_pipeline() {
        let text = "Hello, Dr. Smith! I have $100 and 2 CPUs.";
        let result = clean_text(text);
        assert!(result.contains("hello"));
        assert!(result.contains("doctor"));
        assert!(result.contains("dollars"));
        assert!(result.contains("c p u"));
    }
}
