#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------
.PHONY: init
init:
	cargo init 

#------------------------------------------------------------------------------
# Maintenance
#------------------------------------------------------------------------------
.PHONY: update
update:
	rustup update
	cargo upgrade -i allow && cargo update

#------------------------------------------------------------------------------
# Development
#------------------------------------------------------------------------------

.PHONY: check-deps
check-deps:
	cargo machete

.PHONY: lint
lint:
	cargo fmt --all
	cargo clippy --all-targets --all-features

.PHONY: fix
fix:
	cargo fix --allow-staged

.PHONY: build
build:
	cargo build

.PHONY: run
run:
	RUST_LOG=debug RUST_BACKTRACE=1 cargo run -- openai -m gpt-4o-mini

.PHONY: clean
clean:
	cargo clean

#------------------------------------------------------------------------------
# Test
#------------------------------------------------------------------------------

.PHONY: test
test:
	cargo test


#------------------------------------------------------------------------------
# Release
#------------------------------------------------------------------------------

.PHONY: release-build
release-build:
	cargo build --release

.PHONY: install-bin
install-bin:
	sudo cp target/release/git_tui_rust /usr/local/bin/git-tui

.PHONY: uninstall
uninstall:
	sodo rm -rf /usr/local/bin/git-tui
