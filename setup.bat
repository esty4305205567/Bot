@echo off
REM שימוש: setup.bat            (משתמש ב-Python ברירת המחדל של py)
REM        setup.bat 3.12       (משתמש בגרסה מסוימת)
if "%1"=="" (set PYCMD=py) else (set PYCMD=py -%1)

%PYCMD% -m venv .venv || goto :fail
call .venv\Scripts\activate || goto :fail
python -m pip install --upgrade pip || goto :fail
pip install -r requirements.txt || goto :fail
if not exist .env copy .env.example .env

echo.
echo === SETUP SUCCEEDED ===
echo Next: edit .env, then run:  .venv\Scripts\activate
exit /b 0

:fail
echo.
echo === SETUP FAILED - see the error above ===
exit /b 1
