PG_MAJOR = 17
EXTVERSION = 0.8.0

###############################################################################
# Vector Database
# - pgvector
###############################################################################
# .PHONY: docker-build
# docker-build:
# 	PG_MAJOR=$(PG_MAJOR) EXTVERSION=$(EXTVERSION) docker compose build db


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
