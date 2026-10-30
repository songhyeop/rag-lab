# 구현·검증 상태

2026-10-30 작업 중. 실제 실행 증거와 미완료 항목은 최종 검증 후 갱신한다.

- 계획·데이터 계약을 코드 전에 작성했다.
- 공개 저장소: https://github.com/songhyeop/rag-lab
- 로컬 Python 3.12·uv lock·ONNX 모델 revision을 준비했다.
- Google OAuth client ID·초대 이메일은 아직 제공되지 않았다. 인증 우회는 없다.
- AWS 계정은 시드니, Free 플랜 ACTIVE로 확인했다. 플랜·정책 변경 없음.
- storage change set은 S3 버킷과 ECR 저장소 추가 2건이며 적용 확인 대기 중이다. 아직 리소스를 만들지 않았다.
- 실제 배포·Google 로그인·Bedrock 생성 평가·rollback은 아직 실행되지 않았다.
