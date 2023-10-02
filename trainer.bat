@echo on
powershell -ExecutionPolicy Bypass -NoExit -Command "& 'C:\Users\leo31\anaconda3\shell\condabin\conda-hook.ps1'; conda activate MachineLearning310; & 'C:\Users\leo31\anaconda3\envs\MachineLearning310\python.exe' 'G:\Il mio Drive\Colab Notebooks\SnakeRL\agent.py'"
pause
