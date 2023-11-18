REM remember to delete the old tensorboard directory if you want to focus on only the new data
@echo off
REM change this file to your own paths: this works within anaconda prompt with env called tensorflow
REM this file opens tensorboard in your default browser
REM it might be necessary to adjust log dir
REM and obviously change the absolute directories to your own


REM Start TensorBoard
set ANACONDA_PATH=C:\Users\tay\anaconda3
set TENSORBOARD_LOGDIR=C:\Users\tay\Desktop\projekt_grupowy\user-verification-touchscreen\machine_learning\learningAPI\tensorboard
call "%ANACONDA_PATH%\Scripts\activate.bat" tensorflow

start http://127.0.0.1:6006/
call tensorboard --logdir="%TENSORBOARD_LOGDIR%" --host=127.0.0.1


