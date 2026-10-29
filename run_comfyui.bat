@echo off
chcp 65001 >nul
title ComfyUI - LTX-Video 2.3
echo ========================================
echo   ComfyUI - LTX-Video 2.3
echo   浏览器: http://127.0.0.1:8188
echo ========================================
echo.

call C:\Users\PPXW\miniconda3\condabin\activate.bat ltx-video
cd /d D:\ai\gen-short-video\comfyui

echo [模型文件]
echo   扩散模型 (GGUF):
echo     - LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf  (快速, 8步)
echo     - LTX-2.3-dev-Q4_K_M.gguf                 (高质量, 20步)
echo   文本编码器 (GGUF):
echo     - gemma-3-12b-it-Q4_K_M.gguf
echo   VAE:
echo     - ltx-video-vae.safetensors
echo.
echo 启动中...
echo.

python main.py --lowvram --force-fp16 --preview-method auto

pause
