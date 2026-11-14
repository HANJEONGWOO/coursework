# AI Tool Usage Report

## Tool

- Product: OpenAI Codex coding agent
- Environment: local workspace with shell and file-editing tools
- Date used: 2026-10-27
- Purpose: understand the supplied assignment, implement the starter code, and
  verify correctness and scheduling performance

## User prompts and interpreted task specification

### Verbatim prompts

The following prompts were provided verbatim:

1. 과제 가이드 전체를 읽고, 요구되는 구현 사항, 동작 제약 조건, 테스트 절차, 대회 규칙, 제출 형식, 그리고 AI 사용 여부 공개 요건을 파악하십시오.
2. 코드를 수정하기 전에 제공된 엔진과 테스트를 검토하십시오. 솔루션 코드는 과제 요구사항에 따라 `starter/student.py` 파일 내에서만 작성해야 합니다.
3. `StudentMemory`를 구현하십시오. 여기에는 올림(ceiling) 방식을 이용한 블록 계산, 원자적(atomic) 할당 및 할당 실패 처리, 메모리 축소 금지, 올바른 예외 처리, 토큰 요청이 0개인 경우에 대한 대응, 메모리 해제 로직, 그리고 과거 최대 사용량(peak) 추적 기능이 포함되어야 합니다.
4. 필수적인 '프리필 우선(prefill-first)' 및 '디코드 우선(decode-first)' 플래너를 구현하십시오. 각 단계 내에서 요청 수락 순서를 유지하고, 토큰 및 프리필 제한을 준수하며, 거부된 요청은 건너뛰어야 합니다. 또한, 각 반복(iteration)마다 요청당 최대 하나의 작업(action)만 생성하도록 하십시오.
5. 공개된 온라인 상태(online state)와 빌더 API만을 사용하여 결정론적(deterministic)인 커스텀 플래너를 설계하십시오. 미래 정보를 참조하지 않아야 하며, 모든 요청을 궁극적으로 완료해야 합니다. 또한 디코드 단계의 ITL(Inter-Token Latency)을 보호하고, TTFT(Time To First Token)를 고려하여 프리필 작업을 스케줄링해야 합니다.
6. 공개 단위 테스트, 모든 M1/M2 시뮬레이션, 제출 소스 코드 검사, 10개 케이스로 구성된 전체 연습 아레나(practice arena), 메모리 엣지 케이스 테스트, 그리고 추가로 생성된 시드(seed)를 이용한 평가를 수행하십시오. 특정 워크로드 이름, 시드, 또는 요청 ID에 대해 예외적인 처리를 하기보다는, 전체적인 동작 양상을 바탕으로 커스텀 정책을 개선하십시오.
7. 제출 전 검토를 위해, 사용한 AI 도구, 실제 입력한 프롬프트, 수행한 작업 내용, 그리고 검증 결과를 별도의 공개 보고서(disclosure report)에 기록하십시오.

## How the tool was used

Codex performed the following work:

1. Read all 56 pages of `ML_Systems-PA1.pdf` and extracted the M1, M2,
   competition, testing, submission, and AI-disclosure requirements.
2. Inspected the supplied simulator, public tests, workload generator, arena
   runner, and scoring implementation.
3. Implemented only the required code in `starter/student.py`:
   - block rounding, allocation, growth, and release in `StudentMemory`;
   - the required prefill-first and decode-first planners; and
   - a deterministic custom policy that batches decode work, limits total
   prefill work to protect ITL, prioritizes feasible requests by TTFT deadline,
   and drains infeasible requests by shortest remaining work.
4. Compared multiple custom scheduling heuristics on all 10 public practice
   cases and on additional generated seeds. The final policy was selected for
   consistent behavior rather than using request IDs, workload names, future
   arrivals, or hidden output lengths.
5. Ran the public unit tests, submission-source validation, all basic mission
   simulations, additional memory edge checks, and the complete practice arena.

Codex also used internally generated subtask instructions to independently
check the PDF requirements, audit the repository, establish baseline failures,
review edge cases, and compare scheduler variants. These subtasks did not use
external solution material.

## Result and verification

- Public tests: 9 of 9 passed.
- Basic missions: every demo, M1, and M2 run completed.
- Submission AST restrictions: passed.
- Public practice arena: all 10 cases completed; mean raw score was 129.99/200
  in the final verification run.
- No network access or third-party Python package was used.

The generated implementation and this disclosure should be reviewed by the
student before submission. The assignment submission code remains
`starter/student.py`; this report is the separate AI-use disclosure required by
page 56 of the assignment guide.
