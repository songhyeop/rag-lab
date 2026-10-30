# 구현·검증 상태

2026-10-30. 최신 사용자 지시에 따라 API 임베딩 + 영구 캐시로 전환하고 **실제 호출 전까지만** 구현·검증한다.

- 이전 CPU 색인 프로세스는 종료했다. 전체 검색 평가 리포트는 만들지 않았으며 미완료다.
- 최신 검증: pytest 56건(DynamoDB Local 포함), Ruff, mypy, 프런트엔드 검사·빌드, SAM lint 통과. 실제 캐시 api_batches=0.
- API 호출은 기본 차단, provider 미선택. OpenAI/Bedrock 어댑터는 모의 응답으로 검증한다.
- SQLite 영구 벡터 캐시, batch checkpoint 재개, namespace 분리, offline FAISS 재생성과 portable snapshot export를 추가했다.
- 실제 embedding API·Bedrock 생성·AWS 배포는 실행하지 않았다. 이전 main 후보 CI는 취소했다.
- 아래의 ONNX 임베딩 smoke는 변경 전 버전의 과거 검증이다. 현재 ONNX는 리랭커에만 사용한다.

현재 계약: [EMBEDDING_CACHE.md](EMBEDDING_CACHE.md). 기존 검증 기록과 남은 항목은 아래에 보존한다.

## 구현 및 로컬 검증

- 공개 저장소: https://github.com/songhyeop/rag-lab , 구현 PR: https://github.com/songhyeop/rag-lab/pull/1
- 계획·데이터 계약을 코드 전에 작성했다. 원본·모델·index·사용자 설정은 Git에서 제외한다.
- SciFact 5,183문서, train 809질의·test 300질의의 고정 manifest를 커밋했다.
- HotpotQA 20,000문서, train/dev 각 200질의를 seed 42로 고정했다. 원본 호스트 DNS 실패 후 게시자 namespace의 고정 Parquet revision을 사용했고 출처·checksum을 manifest에 기록했다.
- Python 3.12·uv lock, BM25/FAISS/RRF/ONNX reranker, 128/256/384 설정, 영어 답변·문서 인용 평가, JSONL/CSV/Markdown 리포트를 구현했다.
- 실제 ONNX 임베딩·리랭커 검색 smoke와 Linux amd64 Lambda 이미지 빌드가 통과했다. Lambda runtime health와 비인증 검색 401 차단도 컨테이너에서 검증했다.
- Ruff, mypy, pytest 41건, 프런트엔드 formatting·TypeScript·production build, SAM lint가 통과했다.
- 병렬 예산 검사는 DynamoDB Local에서 실행했다. Moto의 병렬 dictionary 오류를 피하기 위해 CI에도 DynamoDB Local을 사용한다. 실제 AWS DynamoDB 부하 검증으로 해석하지 않는다.
- PR CI는 AWS 인증·Bedrock 호출 없이 실행한다. main 보호에 backend/frontend 필수 검사와 PR 경로를 설정했다.
- 로컬 AWS 인증의 login provider에 필요한 boto3[crt]를 사용한다. 키·로그인 세션을 GitHub로 복사하지 않았다.
- 공식 Sydney Price List에서 Nova Lite On-demand 입력 $0.063 / 출력 $0.252 per million tokens를 확인해 ignored pricing.local.json에 SKU 증거와 함께 고정했다.
- 실제 CountTokens 요청은 `ValidationException: The provided model doesn't support counting tokens.`으로 거부됐다. 따라서 입력 최대 300,000토큰·출력 512토큰 비용을 먼저 예약하고, 실제 response usage로 정산한다. SDK 생성 재시도는 0회다.

가격 출처: https://aws.amazon.com/bedrock/pricing/ 및 https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/ap-southeast-2/index.json

## 실행 대기·미완료

- SciFact/HotpotQA 전체 ONNX 색인과 실제 검색 비교 리포트: 사용자 지시로 중단했다. API 호출 승인 후 새로운 API 모델로 생성해야 한다.
- 128/256/384 전체 train sweep 및 그 결과를 이용한 설정 선택: 아직 실행하지 않았다. 기본 256 설정은 최종 점수를 보고 변경하지 않는다.
- Google OAuth client ID·초대 Gmail·알림 수신 주소가 아직 제공되지 않았다. 실제 Google 로그인 검증은 대기 중이며 인증 우회는 없다.
- AWS 계정은 시드니, Free 플랜 ACTIVE로 확인했다. 플랜·정책 변경 없음.
- storage change set `review-c03eafd2-f7fd-4c4d-bfcc-d0ebe033fef8`은 S3 비공개 versioned 버킷과 ECR 저장소 추가 2건이다. 계획의 별도 확인 단계에 따라 적용 승인을 기다린다. 실제 리소스는 아직 만들지 않았다.
- 서비스 SAM change set은 CI 이미지·index와 Google·알림 설정이 준비된 후 별도로 검토해야 한다.
- 실제 Bedrock 생성 평가·30건 수동 실패 분석·AWS 서비스 배포·CloudFront 로그인·배포 후 실제 rollback·Lambda warm/cold p95는 아직 검증하지 않았다. 코드나 mock 테스트를 이 완료 조건의 대체 증거로 표시하지 않는다.

## 다음 실행

1. CI가 성공한 main 후보의 run ID로 다운로드·checksum 검증한다.
2. storage change set 적용 확인 후 scripts.stage로 그 CI 이미지와 index를 올린다.
3. Google·초대·알림·단가를 local parameter 파일에 채우고 서비스 change set을 검토/확인 후 적용한다.
4. 유효한 Google ID token으로 scripts.deploy의 후보 smoke와 실제 endpoint 검증을 수행한다.
5. --generate로 공유 예산의 생성 평가와 수동 30건 검토를 끝내고, 별도 릴리스에서 실제 rollback을 검증한다.

정확한 명령과 실패 시 복구 절차는 [RUNBOOK.md](RUNBOOK.md)에 있다.
