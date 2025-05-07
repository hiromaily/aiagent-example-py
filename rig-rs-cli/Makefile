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
	cargo update

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
	RUST_LOG=debug RUST_BACKTRACE=1 cargo run -- "Who are you" -m gpt-4.1-nano
	RUST_LOG=debug RUST_BACKTRACE=1 cargo run -- "Who are you" -m gpt-4o-mini
	RUST_LOG=debug RUST_BACKTRACE=1 cargo run -- "Who are you" -t ollama -m llama3.2
	RUST_LOG=debug RUST_BACKTRACE=1 cargo run -- -t ollama -m llama3.2 basic --question "Who are you"
	RUST_LOG=debug RUST_BACKTRACE=1 cargo run -- "Who are you" -t ollama -m llama3.2 prompt

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
	sudo cp target/release/rig_rs_cli /usr/local/bin/rig_cli

.PHONY: uninstall
uninstall:
	sodo rm -rf /usr/local/bin/rig_cli
