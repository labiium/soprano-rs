//! Audio Sample Generator Example
//!
//! NOTE: This functionality is now built into the main `soprano` CLI.
//!
//! This example is kept for backward compatibility but simply delegates
//! to the unified CLI. Use the main binary instead:
//!
//!   soprano generate --text "Hello world"
//!   soprano generate --text "Hello world" --use-model
//!   soprano generate --file samples.txt --output-dir ./samples/

use std::process::{Command, Stdio};

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  DEPRECATED: generate_samples example                      ║");
    println!("║                                                            ║");
    println!("║  This functionality is now built into soprano CLI:     ║");
    println!("║                                                            ║");
    println!("║    soprano generate --text \"Hello world\"               ║");
    println!("║    soprano generate --text \"Hello world\" --use-model   ║");
    println!("║    soprano generate --file samples.txt                 ║");
    println!("║                                                            ║");
    println!("║  Forwarding arguments to: soprano generate ...         ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // Get command line arguments (skip the first which is the binary name)
    let args: Vec<String> = std::env::args().skip(1).collect();

    // Build the command to run soprano generate
    let mut cmd = Command::new("soprano");
    cmd.arg("generate");
    cmd.args(&args);
    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());

    // Run the command
    match cmd.status() {
        Ok(status) => {
            std::process::exit(status.code().unwrap_or(1));
        }
        Err(e) => {
            eprintln!("Failed to run soprano: {}", e);
            eprintln!();
            eprintln!("Please ensure soprano is installed and in your PATH:");
            eprintln!("  cargo install --path .");
            std::process::exit(1);
        }
    }
}
