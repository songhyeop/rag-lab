# RAG Lab

검색·청킹·근거 인용을 재현 가능하게 비교하는 작은 RAG 실험실. 영어 데이터와 한국어 UI를 사용합니다.

먼저 [프로젝트 계획](docs/RAG_PROJECT_PLAN.md), [데이터 계약](docs/DATA_CONTRACT.md), [실행 방법](docs/RUNBOOK.md), [현재 검증 상태](docs/IMPLEMENTATION_STATUS.md)를 읽으세요.

Python 3.12, uv, Node 22 이상이 필요합니다. 데이터·모델·계정 설정은 저장소에 포함하지 않습니다. 실제 생성은 Google 인증과 공유 DynamoDB 예산, 검증한 가격 설정을 모두 갖춘 경우에만 실행합니다.
