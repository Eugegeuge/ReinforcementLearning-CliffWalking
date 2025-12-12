@echo off
echo --- Configuración Automática del Repositorio Remoto ---
echo.
echo Este script vinculará tu repositorio local con GitHub y subirá los cambios.
echo Asegúrate de haber creado un repositorio VACÍO en GitHub antes de continuar.
echo.
set /p repo_url="Pega aquí la URL de tu repositorio de GitHub (ej: https://github.com/usuario/repo.git): "

if "%repo_url%"=="" goto error

echo.
echo Vinculando con %repo_url%...
git remote remove origin 2>nul
git remote add origin %repo_url%

echo.
echo Renombrando rama principal a 'main'...
git branch -M main

echo.
echo Subiendo archivos...
git push -u origin main

echo.
if %errorlevel% neq 0 (
    echo [ERROR] Hubo un problema al subir los archivos. Verifica la URL y tus credenciales.
) else (
    echo [ÉXITO] Repositorio subido correctamente.
)
pause
goto :eof

:error
echo [ERROR] No has introducido ninguna URL.
pause
