//! Man pages for shiftpaper and shiftpaperd, made from the same clap
//! definitions as their --help. They're kept in assets/man, along with
//! the shell completions, for packages to install, and each binary's
//! tests check they're current. After changing the help, update them with:
//!
//! ```sh
//! SHIFTPAPER_REGENERATE=1 cargo test generated
//! ```

use clap_mangen::Man;
use clap_mangen::roff::{Roff, roman};
use std::io::{Result, Write};
use std::path::Path;

const DOCS: &str = "https://github.com/CPritch/shiftpaper/tree/main/docs";

/// A man page for `cmd`. Each "Heading:" block in its after_help becomes
/// a section, like EXAMPLES, and it ends by pointing at `see_also` and
/// the docs. The version is left out so the pages don't change with every
/// release.
pub fn page(cmd: &clap::Command, see_also: &[&str]) -> String {
    let man = Man::new(cmd.clone()).source("shiftpaper");
    type Section = fn(&Man, &mut dyn Write) -> Result<()>;
    let mut sections: Vec<Section> = vec![
        Man::render_title,
        Man::render_name_section,
        Man::render_synopsis_section,
        Man::render_description_section,
    ];
    if cmd.get_arguments().any(|a| !a.is_hide_set()) {
        sections.push(Man::render_options_section);
    }
    if cmd.get_subcommands().any(|s| !s.is_hide_set()) {
        sections.push(Man::render_subcommands_section);
    }

    // Each part starts with the same preamble, which only needs to be
    // there once.
    let preamble = Roff::default().render();
    let mut page = preamble.clone();
    for section in sections {
        let mut out = Vec::new();
        section(&man, &mut out).unwrap();
        page += String::from_utf8(out)
            .unwrap()
            .strip_prefix(&preamble)
            .unwrap();
    }
    page += after_help(cmd, see_also)
        .render()
        .strip_prefix(&preamble)
        .unwrap();
    page
}

fn after_help(cmd: &clap::Command, see_also: &[&str]) -> Roff {
    let mut roff = Roff::default();
    let help = cmd
        .get_after_help()
        .map(|h| h.to_string())
        .unwrap_or_default();
    let mut in_section = false;
    for line in help.lines() {
        if let Some(heading) = line.strip_suffix(':').filter(|h| !h.starts_with(' ')) {
            if in_section {
                roff.control("fi", []);
            }
            // Keep the lines as they are, so the columns stay lined up.
            roff.control("SH", [heading.to_uppercase().as_str()])
                .control("nf", []);
            in_section = true;
        } else if let Some(text) = line.strip_prefix("  ") {
            roff.text([roman(text)]);
        }
        // Anything else points at --help, which a man page doesn't need.
    }
    if in_section {
        roff.control("fi", []);
    }
    roff.control("SH", ["SEE ALSO"])
        .text([roman(see_also.join(", "))])
        .control("PP", [])
        .text([roman(format!("The docs: {DOCS}"))]);
    roff
}

/// Check each of `files`, a path under assets/ and its contents, matches
/// what's there. With SHIFTPAPER_REGENERATE set, write them there instead.
pub fn check(files: &[(String, String)]) {
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets");
    let regenerate = std::env::var_os("SHIFTPAPER_REGENERATE").is_some();
    let mut stale = Vec::new();
    for (name, contents) in files {
        let path = assets.join(name);
        if regenerate {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, contents).unwrap();
        } else if std::fs::read_to_string(&path).ok().as_ref() != Some(contents) {
            stale.push(name.as_str());
        }
    }
    assert!(
        stale.is_empty(),
        "out of date in assets/: {stale:?}\nupdate them with: SHIFTPAPER_REGENERATE=1 cargo test generated"
    );
}
