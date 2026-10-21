# phro-gcp-mlflow 설계

날짜: 2026-10-19
상태: 초안(사용자 검토 대기)

## 1. 목적

어떤 ML 프로젝트든 GCP GPU VM 1대와 MLflow 위에서 실험 루프를 돌리는 범용 AutoML 플러그인. 프로젝트별 지식은 도메인 팩에 두고, 엔진(스킬·스크립트·작업반장)은 모든 프로젝트가 같이 쓴다.

입력은 둘 중 하나다.
- Mode 1: 대략의 워크플로우. 어떤 모델을 어떻게 조합할지 imozuru로 원문 근거를 모아 분석하고, 작은 실증으로 고른 뒤 Mode 2로 넘긴다.
- Mode 2: 고정된 모델과 워크플로우 구조. 단계별 probe 계획, sweep 축, 게이트, baseline 표를 제안하고, 런이 끝날 때마다 진단해 다음 sweep을 고친다.

어느 쪽이든 결과는 프로젝트의 도메인 팩(`.phro/`)이다. 그 뒤로는 같은 엔진이 MLflow 기록, 작업반장, 한계 정리, 돌파구 조사(imozuru -> 아이디어 주머니), 돌파 실험을 돌린다.

성공 기준:
1. 새 프로젝트를 온보딩하면 엔진 코드 수정 없이 팩과 서버 설치만으로 루프가 돈다.
2. 첫 도메인 팩(비공개 프로젝트)으로 인수 시험을 통과한다(10장).
3. 엔진은 지정된 VM 1대 밖의 GCP 자원을 건드리지 않는다(8장).

범위 밖(필요해지면 추가): VM 여러 대, Vertex AI·GCS·Cloud SQL, 아이디어 선택 밴딧, 정기 자동 watch.

## 2. 구조

```
phro-gcp-mlflow/                 공개 플러그인
  .claude-plugin/plugin.json
  skills/
    phro-onboard/SKILL.md        Mode 1·2 -> 도메인 팩
    phro-mlflow/SKILL.md         운영: 판정, 비교, 진단, sweep 설계, 돌파구
    phro-mlflow/references/      diagnosis.md, eval.md, sweep.md, insights.md,
                                 legacy-review.md, lessons.md, probe.md,
                                 idea-bag.md(형식만), breakthrough.md, contract.md
    phro-onboard/templates/project.example.json
  bin/phro.py                    PC에서 도는 CLI(표준 라이브러리만, MLflow REST + ssh)
  foreman/foreman.py             Stop 훅 + MCP(foreman_check, foreman_report)
  hooks/guard.py, hooks/hooks.json   PreToolUse: GCP 격리 + 반출 금지어
  .mcp.json                      작업반장 MCP
  server/                        VM에서 도는 것(~/phro/<project>/ 로 복사)
    phro_server.py               setup, queue(빈 GPU 띄우기), manifest (표준 라이브러리)
    phro_sync.py                 표준 결과 폴더 -> MLflow (VM venv의 mlflow)
  domains/                       gitignore. 이관용 참고 사본만

<project>/.phro/                 도메인 팩(프로젝트 안, 플러그인 레포에 없음)
  project.json                   스크립트가 읽는 계약
  domain.md                      에이전트가 읽는 서술
  plan.md                        사용자 승인 대기열
  ledger/insights.md             맨 위 "열린 한계" 표 + 인사이트 항목
  ledger/idea-bag.md
  ledger/lessons.md
  research/<topic>/              imozuru 결과(map.md, sources/, report.html)
```

imozuru는 플러그인에 넣지 않는다. 설치된 imozuru 스킬을 호출하고, 결과 위치만 `.phro/research/`로 지정한다. 없으면 phro-mlflow가 설치를 안내하고 돌파구 단계만 건너뛴다.

## 3. 도메인 팩

### 3.1 project.json (스크립트가 읽는 값만)

```json
{
  "name": "proj",
  "language": "ko",
  "goal": {"kpis": [{"key": "heldout/f1", "target": 0.9, "dir": "max", "source": "requirements.md sec 2"}]},
  "stages": [
    {"name": "embed", "experiment": "1_embed", "depends_on": []},
    {"name": "classify", "experiment": "2_classify", "depends_on": ["embed"]}
  ],
  "metrics": [
    {"key": "dev/f1", "stage": "classify", "dir": "max",
     "resample_unit": "doc_id", "agg": {"type": "f1", "tp": "tp", "fp": "fp", "fn": "fn"},
     "gate": {"min_delta": 0.0}},
    {"key": "heldout/f1", "stage": "classify", "dir": "max", "target": 0.9,
     "resample_unit": "doc_id", "agg": {"type": "f1", "tp": "tp", "fp": "fp", "fn": "fn"},
     "gate": {"min_delta": 0.0}}
  ],
  "eval_sets": {"select": "dev", "heldout": "heldout", "final": "final", "look_budget": {"heldout": 10, "final": 1}},
  "server": {"host": "my-gpu", "allow_hosts": ["my-gpu"], "root": "~/phro/proj",
             "mlflow_port": 5002, "gpus": 4, "venv": "~/phro/proj/venv"},
  "egress": {"forbidden_terms": ["..."], "note": "검색어·서브에이전트 프롬프트에 넣지 않을 것"},
  "stats": {"seeds_min": 4, "bootstrap": 1000, "min_units": 30}
}
```

규칙: 어떤 스크립트도 읽지 않는 값은 넣지 않는다. `stats`를 빼면 코드 기본값(seeds_min 4, bootstrap 1000, min_units 30)을 쓴다. 게이트가 있는 지표는 select 세트 쌍둥이(`dev/f1`)를 함께 둔다. `decide`는 select 세트 지표로 판정한다. 데이터 버전은 팩이 아니라 런 태그 `data_version`이다.

`agg`는 단위 점수를 모으는 방법이다. `{"type": "mean", "field": "x"}`는 단위 값의 평균, `{"type": "f1", "tp", "fp", "fn"}`은 단위별 개수를 더한 뒤 micro F1(단위별 F1의 평균이 아님).

선택 키(기존 sync가 있는 프로젝트용, 없으면 기본값):
- `tag_alias`: 계약 태그 이름 -> 그 프로젝트의 태그 이름(예 `{"data_version": "model_version"}`).
- `derived_params`: `diff`에서 빼는 결과성 파라미터 이름 목록. 기본 `[]`.
- `server.manifest`: 기본 `<root>/runs.json`. `server.queue`: 기본 `<root>/gpu_queue.txt`.
- `server.read_only`: 기본 `false`. `true`면 `setup`, `manifest set`이 거부한다(기존 VM을 읽기만 할 때).
- `server.done_globs`: 기본 `["runs/**/DONE"]`(root 기준). `server.run_name_rules`: 끝남 파일 경로 -> 런 이름 `[[정규식, 치환]]`, 기본 `[["(?:.*/)?([^/]+)/DONE$", "\\1"]]`. 치환이 빈 문자열이면 스크립트 끝 표시로 본다.

### 3.2 domain.md (에이전트만 읽는 값)

목표 사슬 표(단계, 지표 키, 목표, 현재 baseline, 상한, 다음 단계로 넘기는 것), 하드웨어·모델 제약(아이디어 fit 판정에 씀), 데이터·개인정보 규칙, 용어, 보고 형식 예시. 숫자는 적지 않고 `phro.py baseline`으로 새로 읽으라고 적는다.

### 3.3 plan.md

사용자가 승인한 다음 일, 위에서부터. 작업반장은 여기 있는 일만 묻지 않고 띄운다. 그 밖(새 방법, 방향 전환, 삭제)은 모두 묻는다. 별도 권한 설정은 두지 않는다.

## 4. MLflow 계약

엔진(`phro.py`, 작업반장)은 파일이 아니라 MLflow만 읽는다. 계약은 MLflow 수준에서 정한다.

| 항목 | 규칙 |
|---|---|
| experiment | `project.json` stages[].experiment |
| run name | 유일, 결과 폴더 이름과 같음. seed는 `_s<N>` 접미사 |
| tags | `status`(`running`/`done`/`smoke`/`frozen`/`aborted`), `stage`, `sweep`(쉼표 목록), `variation`, `compare`, `axis`, `idea`, `insight`, `review`, `intro`, `result`, `data_version`, `budget`(`short`/`full`, 없으면 full), `accepted`(`yes`, `decide` accept 뒤), `gate_<set>`(`decide`의 지표별 판정), `reopen_if`(frozen일 때) |
| metrics | `<eval set>/<name>`. 같은 키끼리만 비교. 런이 CI를 직접 기록하면 `<set>/ci_lo_<name>`, `<set>/ci_hi_<name>` |
| artifact `units.jsonl` | 줄마다 `{"set", "unit", <agg가 읽는 필드>}`(예 tp, fp, fn). 세트 안에서 unit id 중복 금지. CI와 짝 차이는 엔진이 계산 |
| leave-one-out | sweep 이름 `loo_<...>` |
| 판정 기록 | 로컬 `.phro/ledger/decisions.jsonl`, `decide`가 한 줄씩 덧붙임(read_only 팩에서도 됨) |
| 끝남 | 결과 폴더의 `DONE` 파일(작업반장이 봄). `pgrep` 금지 |

표준 결과 폴더(`phro_sync.py`가 읽음): `run.json`(name, stage, params, tags), `metrics.jsonl`(`{"step", "set", "key", "value"}`, 최종값은 `"step": "final"`), `units.jsonl`, `DONE`. 학습 코드가 쓰는 예시 10줄은 `references/contract.md`에 둔다.

이미 자기 sync가 있는 프로젝트는 그대로 써도 된다. 위 표의 이름과 `units.jsonl`만 맞추면 된다.

런 매니페스트(서론·결과 등 실행 전에 쓰는 값)는 서버 `~/phro/<project>/runs.json`. `phro.py manifest get|set NAME key=value`로 고친다(scp 경합 없음). sync가 태그로 옮긴다.

## 5. CLI bin/phro.py

현재 비공개 프로젝트의 `mlflow_query.py`를 팩 기반으로 일반화한다. PC에서 돈다: MLflow는 REST(`http://127.0.0.1:<mlflow_port>`, ssh 터널), GPU·파일은 `ssh <server.host>`. 표준 라이브러리만 쓰므로 PC에 mlflow를 깔지 않는다. 터널이 없으면 터널 명령을 그대로 출력한다. 출력은 숫자만(원문 데이터 출력 금지). 추가 명령: `check-pack`(팩 검사), `leakcheck <dir>`(git 추적 파일에서 금지어·호스트 찾기), `setup`(server/ 복사 후 원격 설치), `manifest get|set`(원격 phro_server.py 호출).

| 명령 | 내용 |
|---|---|
| `status` | GPU별 런, 진행률, 남은 시간, FREE, 대기 스크립트 |
| `list [--exp]` | 런·sweep 이름 |
| `report --run N` / `--since H` | 판정 줄 + CI + compare 태그 런 |
| `compare A B ... [--set S]` / `--sweep S` | 같은 키 나란히, CI, 게이트. 기본 세트는 select |
| `diff NEW --vs OLD` | 실제 파라미터 차이(선언한 축 밖이면 교란) |
| `ci A B ...` | `units.jsonl`로 런별 95% bootstrap CI + 같은 단위 재표집 짝 차이. 단위 잡음만 보므로 판정 없음 |
| `ci BASE ARM --pairs` | 런 가족(`<가족>_s<N>`)의 seed 짝 차이, t 구간(seed 분산 + 단위 분산), 게이트 판정 |
| `decide BASE ARM` | 단계의 select 지표 전부로 채택 판정(accept, accept-short, reject, undecided), `decisions.jsonl`에 기록. `--reverify`, `--loo` |
| `loo --stage S --data V` | leave-one-out으로 시험 안 된 채택과 변경마다 만들 갈래 |
| `splitcheck RUN` | select·heldout·final 사이 겹치는 unit id 수(0이어야 함) |
| `baseline [--data V]` | 데이터 버전 baseline 표: 단계별 런 가족(seed 접미사 제거), seed 수, 평균 [min, max], 목표까지 거리 |
| `cascade --stage S` | `depends_on` DAG로 S가 바뀌면 다시 학습할 끝난 런 목록 |
| `audit` | 누락 태그(intro/result/review), 끝났는데 sync 안 됨, GPU 놀면서 스크립트 대기, look_budget 초과(데이터 버전마다 런 가족 수), leave-one-out 안 된 채택 3개 이상, 재검증 안 된 단기 예산 통과 |
| `manifest get|set` | 런 매니페스트 |

## 6. 온보딩 phro-onboard

### Mode 1: 워크플로우 -> 모델 조합
1. 워크플로우를 단계로 나눈다. 단계마다 입출력, 예산(크기, 지연, 데이터)을 적는다.
2. 단계마다 imozuru 조사. 요구 주장 고정: 이 크기에서 이 일을 하는 모델 계열, 측정된 수치, 실패 조건, 원전. 검색어는 8장 반출 금지어 검사를 거친다.
3. 조합안 2~3개(cascade, end-to-end, 혼합)를 근거 표(원문 `sources/<file>:줄`)와 함께 낸다.
4. 실증: 후보 모델을 dev 일부에 zero-shot 또는 작은 미세조정으로 돌려 단계별 상한을 잰다. GPU를 쓰므로 사용자 승인 뒤.
5. 하나를 확정하면 그 구조를 Mode 2 입력으로 넘긴다.

### Mode 2: 고정 구조 -> 단계별 sweep 제안
단계마다 probe 계획(데이터 점검, 상한, 정보량, 손실 분해, smoke), 방법 선별 축 목록, HPO 공간(방법이 정해진 뒤에만), 게이트와 지표, baseline 표(데이터 버전 × 단계 × 바리에이션 × seed, 사슬 칸 포함)를 낸다. 결과는 `.phro/` 파일들과 `plan.md` 첫 항목들이다.

마지막에 독립 서브에이전트(세션 맥락 없음)가 팩을 목표와 대조해 검토한다(legacy-review와 같은 방식). 사용자 승인 뒤 팩 확정.

## 7. 운영 phro-mlflow

현재 비공개 스킬의 규율을 그대로 옮기고 도메인 예시를 `domain.md`로 뺀다.
- 결과 질문 -> `report`, 목표 사슬 표에서 어느 칸이 얼마나 움직였는지 한 줄로 시작.
- 끝난 런 -> diagnosis 8단계(막는 기준, ablation 사다리, 배제, 상한, 메커니즘, 측정/추정, 판별 진단, 함의) -> insights -> 런 `result`.
- 새 실험 -> insights -> idea-bag -> probe -> sweep(서론 작성) -> legacy-review -> 승인 -> 실행 후 `diff`.
- 실패·유휴 -> lessons(두 번 반복되면 규칙 제안, 사용자 승인 뒤 반영).

### 7.1 한계 정리
`insights.md` 맨 위 "열린 한계" 표. 에이전트가 진단 뒤 고친다.

| L-ID | 단계 | 막는 지표(값) | 목표까지 | 배제된 것(sweep) | 상한 근거 | IDEA | 상태 |
|---|---|---|---|---|---|---|---|

상태: `open`, `broken`(돌파됨, 런 링크), `bound`(입력·정보 상한으로 확인됨).

### 7.2 돌파구 (references/breakthrough.md)
1. 계기: 같은 한계에 sweep 2개가 이득 없이 닫힘, 그 한계를 겨냥한 아이디어 없음, 또는 사용자 요청.
2. 한계 행을 일반적인 문제 서술로 바꾼다. 프로젝트 데이터·금지어 없이.
3. imozuru 조사. 요구 주장: 메커니즘, 측정된 효과, 실패 조건, 우리 규모에서의 비용. 멈춤은 imozuru 규칙(주장 전부 원문, 포화, 4주).
4. `idea-bag.md`에 IDEA-NNN. 각 주장에 `research/<topic>/sources/<file>:줄`과 출처 종류(논문/본인/정리). fit H/M/L은 `domain.md` 제약으로, 어느 L-ID를 겨냥하는지 적는다. fit L은 "넣지 않음" 한 줄.
5. 사용자가 고르면(또는 이미 plan.md에 있으면) 돌파 실험: probe -> 가장 센 baseline 대비 한 축 -> 9장 통계 규칙.
6. 판정 뒤 아이디어 status/result, 인사이트, 한계 표 상태를 고친다.

### 7.3 작업반장
현재 foreman.py를 팩 기반으로 바꾼다. 원격 스냅샷은 `server.host`, `server.root`, `DONE` 파일, 매니페스트, 대기열만 본다. 지시 우선순위:
1. 끝났는데 판정 안 된 런(서론의 기대 대비)
2. 빈 GPU -> `plan.md`
3. 서론만 있고 결과 없는 런
4. 대기 중인 아이디어
5. 단기 예산 채택의 전체 예산 재검증, leave-one-out
6. 할 일이 없으면 돌파구 조사 후보

`phro_server.py queue`(cron 5분): `<GPU> <script>` 줄을 그 GPU에 compute 프로세스가 없을 때만 띄우고, 띄운 PID를 `~/phro/<project>/pids/`에 적는다.

## 8. 격리와 반출 (단순화 대상 아님)

1. 엔진 어디에서도 `gcloud`를 부르지 않는다. 원격 작업은 모두 `ssh <server.host>`.
2. `hooks/guard.py`(PreToolUse, Bash·PowerShell·Monitor·WebSearch·WebFetch)가 막는 것:
   - `gcloud compute instances ... (stop|delete|reset|start|create)`, `gcloud compute firewall-rules`
   - `server.allow_hosts`에 없는 호스트·IP로의 `ssh`/`scp`
   - `egress.forbidden_terms`가 들어간 검색어·URL, 네트워크로 나가는 셸 명령(`curl`, `wget`, `gh`, `Invoke-WebRequest`, imozuru 수집 스크립트; 루프백 URL은 반출 아님), Monitor의 WebSocket URL
   - 인코딩된 PowerShell(`-EncodedCommand`, `-e`). PowerShell 감싸기(`-Command`, 스크립트 블록, `Start-Process`, `iex`)는 안까지 최선을 다해 검사
   팩이 없으면 gcloud 규칙만 적용한다. 자체 점검 하나(`python guard.py selftest`).
3. VM 위: `phro_server.py queue`는 빈 GPU에만 띄우고, 스스로 적은 PID만 멈춘다. 다른 세션·다른 프로젝트 작업은 죽이지 않는다.
4. `phro_server.py setup`: `~/phro/<project>/` 아래에만 설치, sudo 없음, `systemctl --user`, cron 줄에 `# phro:<project>` 표시를 달고 표시 없는 줄은 건드리지 않음, MLflow는 `127.0.0.1`만, 포트가 쓰이고 있으면 중단.
5. 플러그인 레포: `domains/`, `.phro/`는 gitignore. 공개 push 전에 금지어 grep 한 번.

## 9. 통계 규칙 (sweep.md에 규칙으로)

1. 채택은 `phro.py decide <비교 가족> <갈래 가족>`으로만. 같은 seed 짝(`<가족>_s<N>`, `stats.seeds_min` 이상, 기본 4)과 공통 단위(`stats.min_units` 이상, 기본 30)에서, 단계의 select 세트 지표마다 seed별 차이의 평균에 t 구간을 씌운다(자유도 = 짝 수 - 1, 분산 = seed별 차이의 분산 / 짝 수 + 단위 재표집 분산). 지표가 모두 통과면 accept, 하나라도 미달이면 reject, 나머지는 undecided("결정 안 됨", 실패 아님). 판정은 `.phro/ledger/decisions.jsonl`에 쌓인다. 런 하나짜리 `ci`는 단위 잡음만 보므로 판정에 쓰지 않는다.
2. 단기 예산(`budget=short`) 갈래는 accept-short까지만 간다. 전체 예산으로 새 가족을 돌려 `decide <비교> <새 가족> --reverify <단기 가족>`이 accept여야 채택. `audit`이 재검증(accept 또는 reject) 안 된 것을 알린다. 같은 갈래를 seed를 늘려 다시 판정하면 앞 판정을 대신한다. 끝난 런(`status=done`)만 짝짓는다.
3. 단계·데이터 버전마다 leave-one-out으로 시험 안 된 채택이 3개 이상이면 baseline이라 부르기 전에 leave-one-out. `loo`가 변경마다 갈래(그 변경만 뺀 스택)를 내고, `decide <갈래> <스택> --loo <변경 가족>`으로 하나씩 닫는다. reject면 뺀다. undecided면 seed를 늘리거나 더 단순한 스택을 고른다.
4. 고르기는 select 세트, 판정은 heldout(`look_budget` 안에서, 데이터 버전마다 런 가족 수로 셈), final은 마지막 한 번. `splitcheck`로 세트 사이 unit id 겹침이 0인지 본다.
5. 구조·방법 축은 에이전트 sweep, 방법이 정해진 뒤 수치 HPO는 Optuna TPE + pruner. 구조적으로 실패한 방법은 HPO하지 않는다.

보정(seed 잡음을 넣은 시뮬레이션, 단위 1000개, seed 4쌍, 각 200회): 효과 0에서 잘못된 통과 0/200, 잘못된 미달 5/200. 검정력은 +3점 59%, +5점 96%. 이전 방식(런 하나의 단위 bootstrap 짝 차이)은 같은 조건, 효과 0에서 잘못된 통과 18%, 잘못된 미달 12%였다.

## 10. 시험

- 스크립트마다 `selftest`(stdlib assert): `phro.py ci`(알려진 입력의 CI), `cascade`(DAG), `guard.py`(막을 것·통과할 것), `foreman.py`(기록된 스냅샷 -> 지시), `phro_server.py`(대기열 고르기, cron 병합).
- 인수 시험(첫 비공개 팩, 읽기 전용): 서버 키트를 그 VM에 설치하지 않는다. PC에서 기존 MLflow 터널로 `phro.py baseline`, `ci`, `compare`를 돌려 기존 `mlflow_query.py` 출력과 같은 숫자인지 본다. 작업반장은 기록된 스냅샷에서 기존과 같은 지시를 내는지 본다. 기존 cron, 대기열, 런은 건드리지 않는다.
- 새 VM 설치 시험은 사용자가 지정한 새 인스턴스에서만.

## 11. 이관 순서

1. 엔진 일반화: phro.py(mlflow_query 기반), foreman.py, guard.py, references(도메인 예시 제거).
2. 첫 팩 작성(프로젝트 폴더 `.phro/`, 레포 밖): 현재 스킬의 목표 사슬, 규칙, 인사이트, 아이디어 주머니, 대기열을 옮긴다.
3. 인수 시험(10장).
4. phro-onboard(Mode 1·2), phro_server.py, phro_sync.py.
5. 공개 push 전 금지어 grep, 사용자 확인 뒤 push.
