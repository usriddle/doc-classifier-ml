-- =============================================================================
-- PostgreSQL 준비 (한 번만)
--
-- 실행 방법 (둘 중 하나)
--   - psql :  psql -U postgres -f scripts/setup_pgvector.sql
--   - pgAdmin : [Query Tool] 에 붙여넣고 실행
--
-- 테이블은 만들지 않습니다. 서버 기동 또는 scripts/sync_cases.py 가 자동으로 만듭니다.
-- =============================================================================

-- 1) 민원 시스템용 데이터베이스
CREATE DATABASE minwon ENCODING 'UTF8' TEMPLATE template0;

-- 2) 방금 만든 DB 로 이동 (psql 전용 명령. pgAdmin 이라면 minwon DB 를 골라 Query Tool 을 새로 여세요)
\c minwon

-- 3) pgvector 확장 켜기 - 여기서 오류가 나면 pgvector 가 설치되지 않은 것입니다 (README 1-6)
CREATE EXTENSION IF NOT EXISTS vector;

-- 4) 확인 - extversion 에 0.8.x 가 보이면 정상
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
SELECT '[1,2,3]'::vector <=> '[1,2,4]'::vector AS cosine_distance_test;
