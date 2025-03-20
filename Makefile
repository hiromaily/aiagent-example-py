
###############################################################################
# Utilities
###############################################################################

.PHONY: clean-venv
clean-venv:
	find . -name ".venv" -type d -exec rm -rf {} +

# `.venv`を除いたディレクトリ内の`__pycache__`を削除する
.PHONY: clean-cache
clean-cache:
	$(MAKE) -C llamaindex-cli clean-cache	
	$(MAKE) -C openai-agents-cli clean-cache	
	$(MAKE) -C pydantic-cli clean-cache	
