@echo off
cd /d C:\Second_project

REM === 시간 포맷 생성 ===
for /f %%i in ('powershell -command "Get-Date -Format yyyy-MM-dd_HH-mm-ss"') do set TIMESTAMP=%%i

REM === 로그 디렉토리 지정 ===
set LOG_DIR=C:\Second_project\logs
set LOG_FILE=%LOG_DIR%\git_log_%TIMESTAMP%.txt

if not exist %LOG_DIR% (
    mkdir %LOG_DIR%
)

set COMMIT_MSG=자동커밋_%TIMESTAMP%

REM === 로그 시작 ===
echo ▶ 실행 시간: %TIMESTAMP% > %LOG_FILE%

REM 1. master 최신 pull
echo [1] origin/master에서 pull 중... >> %LOG_FILE%
git checkout master >> %LOG_FILE% 2>&1
git pull origin master >> %LOG_FILE% 2>&1

REM 2. jhyun 브랜치 전환 (없으면 생성)
echo [2] jhyun 브랜치로 전환 중... >> %LOG_FILE%
git checkout jhyun >> %LOG_FILE% 2>&1 || git checkout -b jhyun >> %LOG_FILE% 2>&1

REM 3. 변경사항 add + commit
echo [3] 변경사항 커밋 중... >> %LOG_FILE%
git add . >> %LOG_FILE% 2>&1
git diff --cached --quiet || git commit -m "%COMMIT_MSG%" >> %LOG_FILE% 2>&1

REM 4. push
echo [4] jhyun 브랜치에 push 중... >> %LOG_FILE%
git push origin jhyun >> %LOG_FILE% 2>&1

echo [완료] %TIMESTAMP% 에 푸시 완료됨 >> %LOG_FILE%

REM 결과 출력
type %LOG_FILE%
pause
