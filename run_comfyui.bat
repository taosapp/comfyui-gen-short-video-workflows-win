@echo off
title ComfyUI - LTX-Video 2.3 (this project)
echo ========================================
echo   ComfyUI - LTX-Video 2.3
echo   Browser: http://127.0.0.1:8189
echo ========================================
echo.

call C:\Users\PPXW\miniconda3\condabin\activate.bat ltx-video
cd /d D:\ai\comfyui-gen-short-video-workflows-win\comfyui

echo [models]
echo   diffusion (GGUF):
echo     - LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf  (fast, 8 steps)
echo     - LTX-2.3-dev-Q4_K_M.gguf                (quality, 20 steps)
echo   text encoder (GGUF):
echo     - gemma-3-12b-it-Q4_K_M.gguf
echo   VAE:
echo     - ltx-2.3-video-vae.safetensors
echo.
echo Starting...

python -u main.py --port 8189 --lowvram --force-fp16 --disable-auto-launch

pause
