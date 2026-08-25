# TTS 評測平台 —— 常用指令
#
# 測試不需要任何遠端服務（引擎與 ASR 都被換成假的），所以 CI 上直接跑得動。

# 有 .venv-console 就用它，沒有就用系統的 python3
PY := $(shell [ -x .venv-console/bin/python ] && echo .venv-console/bin/python || echo python3)
N  ?= 20      # make test-flaky 要連跑幾次
J  ?= 1       # make test-flaky 同時跑幾個

.PHONY: help install run test test-ci test-flaky clean

help:            ## 列出可用指令
	@grep -hE '^[a-z][a-z-]*:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:         ## 建 .venv-console 並裝套件
	python3 -m venv .venv-console
	.venv-console/bin/pip install -r requirements-console.txt

run:             ## 啟動平台（http://127.0.0.1:8080）
	./run.sh

test:            ## 跑一次完整流程測試
	$(PY) scripts/smoke_test.py

test-ci:         ## CI 用：只印失敗，另外產出 JUnit 與 JSON 報告
	$(PY) scripts/smoke_test.py --quiet --no-color \
	  --junit smoke-report.xml --json smoke-report.json

test-flaky:      ## 連跑 N 次找時好時壞的項目： make test-flaky N=50 J=4
	$(PY) scripts/smoke_test.py --repeat $(N) --jobs $(J) --no-color \
	  --junit smoke-flaky.xml --json smoke-flaky.json

clean:           ## 清掉測試報告與 __pycache__
	rm -f smoke-report.xml smoke-report.json smoke-flaky.xml smoke-flaky.json
	find . -name __pycache__ -type d -not -path "./.venv*" -exec rm -rf {} +
