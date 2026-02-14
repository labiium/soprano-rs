//! Fancy CLI styling and ASCII art for Soprano TTS

use owo_colors::OwoColorize;
use std::io::Write;

/// Check if terminal supports colors
pub fn supports_color() -> bool {
    supports_color::on(supports_color::Stream::Stdout).is_some()
}

/// Get the Soprano ASCII art banner
pub fn get_banner() -> &'static str {
    r#"
      ██████  ██████  ██████  ██████   █████  ███    ██  ██████
     ██      ██    ██ ██   ██ ██   ██ ██   ██ ████   ██ ██    ██
      █████  ██    ██ ██████  ██████  ███████ ██ ██  ██ ██    ██
          ██ ██    ██ ██      ██   ██ ██   ██ ██  ██ ██ ██    ██
      ██████  ██████  ██      ██   ██ ██   ██ ██   ████  ██████

                 High-Performance Neural Text-to-Speech
"#
}

/// Print the banner with gradient colors
pub fn print_banner() {
    let banner = get_banner();

    if supports_color() {
        // Print with gradient effect
        for (i, line) in banner.lines().enumerate() {
            match i % 6 {
                0 => println!("{}", line.bright_red()),
                1 => println!("{}", line.red()),
                2 => println!("{}", line.yellow()),
                3 => println!("{}", line.bright_yellow()),
                4 => println!("{}", line.green()),
                _ => println!("{}", line.bright_green()),
            }
        }
    } else {
        println!("{}", banner);
    }
}

/// Print a styled box with title
pub fn print_box(title: &str, content: &[(&str, &str)]) {
    let width = 60;
    let title_width = title.len() + 4;
    let padding = (width - title_width) / 2;

    // Top border
    println!("╔{}╗", "═".repeat(width));

    // Title
    print!("║{}{} ║", " ".repeat(padding), title.bright_cyan().bold());
    if (width - title_width) % 2 == 1 {
        print!(" ");
    }
    println!();

    // Separator
    println!("╠{}╣", "═".repeat(width));

    // Content
    for (label, value) in content {
        let line = format!("  {}: {}", label.bright_blue(), value);
        let spaces = width.saturating_sub(line.len() + 2);
        println!("║{}{}║", line, " ".repeat(spaces));
    }

    // Bottom border
    println!("╚{}╝", "═".repeat(width));
}

/// Print a styled box with owned key/value strings
pub fn print_box_kv(title: &str, content: &[(String, String)]) {
    let width = 60;
    let title_width = title.len() + 4;
    let padding = (width - title_width) / 2;

    // Top border
    println!("╔{}╗", "═".repeat(width));

    // Title
    print!("║{}{} ║", " ".repeat(padding), title.bright_cyan().bold());
    if (width - title_width) % 2 == 1 {
        print!(" ");
    }
    println!();

    // Separator
    println!("╠{}╣", "═".repeat(width));

    // Content
    for (label, value) in content {
        let line = if supports_color() {
            format!("  {}: {}", label.bright_blue(), value.bright_white())
        } else {
            format!("  {}: {}", label, value)
        };
        let spaces = width.saturating_sub(line.len() + 2);
        println!("║{}{}║", line, " ".repeat(spaces));
    }

    // Bottom border
    println!("╚{}╝", "═".repeat(width));
}

/// Print a section header
pub fn print_section(title: &str) {
    if supports_color() {
        println!("\n{}", "━".repeat(60).bright_black());
        println!("  {}", title.bright_cyan().bold());
        println!("{}", "━".repeat(60).bright_black());
    } else {
        println!("\n{}", "━".repeat(60));
        println!("  {}", title);
        println!("{}", "━".repeat(60));
    }
}

/// Print a compact single-line configuration summary
pub fn print_compact_config(items: &[(&str, String)]) {
    if supports_color() {
        let parts: Vec<String> = items
            .iter()
            .map(|(label, value)| {
                format!("{} {}", label.bright_blue().bold(), value.bright_white())
            })
            .collect();
        println!("{}", parts.join("  "));
    } else {
        let parts: Vec<String> = items
            .iter()
            .map(|(label, value)| format!("{} {}", label, value))
            .collect();
        println!("{}", parts.join("  "));
    }
}

/// Print a success message
pub fn print_success(message: &str) {
    if supports_color() {
        println!("{} {}", "✓".bright_green().bold(), message.green());
    } else {
        println!("[OK] {}", message);
    }
}

/// Print an error message
pub fn print_error(message: &str) {
    if supports_color() {
        eprintln!("{} {}", "✗".bright_red().bold(), message.red());
    } else {
        eprintln!("[ERR] {}", message);
    }
}

/// Print a warning message
pub fn print_warning(message: &str) {
    if supports_color() {
        println!("{} {}", "⚠".bright_yellow().bold(), message.yellow());
    } else {
        println!("[WARN] {}", message);
    }
}

/// Print an info message
pub fn print_info(message: &str) {
    if supports_color() {
        println!("{} {}", "ℹ".bright_blue().bold(), message.bright_white());
    } else {
        println!("[INFO] {}", message);
    }
}

/// Print device status
pub fn print_device_status(name: &str, available: bool) {
    if available {
        if supports_color() {
            println!(
                "  {} {} {}",
                "✓".bright_green().bold(),
                name.bright_green(),
                "(available)".green()
            );
        } else {
            println!("  [OK] {} (available)", name);
        }
    } else {
        if supports_color() {
            println!(
                "  {} {} {}",
                "✗".bright_red(),
                name.bright_black(),
                "(not available)".bright_black()
            );
        } else {
            println!("  [  ] {} (not available)", name);
        }
    }
}

/// Print the devices info box
pub fn print_devices_box(
    platform: &str,
    devices: &[(&str, bool)],
    recommended: &str,
    selected: &str,
) {
    let width = 58;

    // Top
    println!("\n  ╔{}╗", "═".repeat(width));
    println!(
        "  ║{:^width$}║",
        "🎵  Compute Device Information  🎵".bright_cyan().bold(),
        width = width
    );
    println!("  ╠{}╣", "═".repeat(width));

    // Platform
    println!(
        "  ║  {} {:<48}║",
        "Platform:".bright_blue(),
        platform.bright_white()
    );
    println!("  ╠{}╣", "═".repeat(width));

    // Devices
    for (name, available) in devices {
        if *available {
            println!("  ║  {} {:<49}║", "●".bright_green(), name.bright_green());
        } else {
            println!("  ║  {} {:<49}║", "○".bright_black(), name.bright_black());
        }
    }

    println!("  ╠{}╣", "═".repeat(width));

    // Recommended
    println!(
        "  ║  {} {:<43}║",
        "Recommended:".bright_yellow(),
        recommended.bright_yellow().bold()
    );

    // Selected
    println!(
        "  ║  {} {:<45}║",
        "Selected:".bright_magenta(),
        selected.bright_magenta().bold()
    );

    // Bottom
    println!("  ╚{}╝", "═".repeat(width));
}

/// Print usage examples
pub fn print_usage_examples() {
    println!();
    if supports_color() {
        println!("{}", "📚 Usage Examples".bright_cyan().bold());
        println!("{}", "─────────────────".bright_black());
        println!();
        println!(
            "  {} {}",
            "▶".bright_green(),
            "soprano serve --device auto".bright_white()
        );
        println!(
            "    {} {}",
            "→".bright_black(),
            "Auto-select best device (default)".bright_black()
        );
        println!();
        println!(
            "  {} {}",
            "▶".bright_green(),
            "soprano serve --device cuda".bright_white()
        );
        println!(
            "    {} {}",
            "→".bright_black(),
            "Force NVIDIA CUDA GPU".bright_black()
        );
        println!();
        println!(
            "  {} {}",
            "▶".bright_green(),
            "soprano serve --device metal".bright_white()
        );
        println!(
            "    {} {}",
            "→".bright_black(),
            "Force Apple Metal (macOS)".bright_black()
        );
        println!();
        println!(
            "  {} {}",
            "▶".bright_green(),
            "soprano serve --device cpu".bright_white()
        );
        println!(
            "    {} {}",
            "→".bright_black(),
            "Force CPU only".bright_black()
        );
        println!();
    } else {
        println!("Usage Examples");
        println!("──────────────");
        println!();
        println!("  soprano serve --device auto");
        println!("    Auto-select best device (default)");
        println!();
        println!("  soprano serve --device cuda");
        println!("    Force NVIDIA CUDA GPU");
        println!();
        println!("  soprano serve --device metal");
        println!("    Force Apple Metal (macOS)");
        println!();
        println!("  soprano serve --device cpu");
        println!("    Force CPU only");
        println!();
    }
}

/// Print a progress bar
pub fn print_progress(current: usize, total: usize, message: &str) {
    let width = 40;
    let filled = (current * width) / total.max(1);
    let empty = width - filled;

    let bar = if supports_color() {
        format!(
            "{}{}",
            "█".repeat(filled).bright_green(),
            "░".repeat(empty).bright_black()
        )
    } else {
        format!("{}{}", "█".repeat(filled), "░".repeat(empty))
    };

    let percent = (current * 100) / total.max(1);

    print!("\r  [{}] {:>3}% {}", bar, percent, message);
    std::io::stdout().flush().unwrap();

    if current >= total {
        println!();
    }
}

/// Print startup info for server
pub fn print_server_startup(host: &str, port: u16, device: &str, model: &str) {
    print_banner();

    let port_str = port.to_string();
    let info = vec![
        ("Host", host),
        ("Port", port_str.as_str()),
        ("Device", device),
        ("Model", model),
    ];

    print_box("Server Configuration", &info);

    println!();
    print_success(&format!("Server ready at http://{}:{}/", host, port));
    println!();

    if supports_color() {
        println!("{}", "Press Ctrl+C to stop".bright_black());
    } else {
        println!("Press Ctrl+C to stop");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_banner_not_empty() {
        let banner = get_banner();
        assert!(!banner.is_empty());
        assert!(banner.contains("Text-to-Speech"));
    }

    #[test]
    fn test_print_functions() {
        // These should not panic
        print_success("test");
        print_error("test");
        print_warning("test");
        print_info("test");
        print_device_status("test", true);
        print_device_status("test", false);
    }
}
