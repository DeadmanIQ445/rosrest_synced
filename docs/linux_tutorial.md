Полезные команды в линуксе:

- ```ssh aidar@10.100.10.50```
    - Нужен для подключения с удаленному серверу
    - Чтобы не вводить несколько раз, нужно скопировать текст из ```id_rsa.pub```
        - ```cat ~/.ssh/id_rsa.pub```
    - и вставить его в конец ```authorized_keys```
        - ```nano ~/.ssh/authorized_keys```

- ```conda```
    -
    cheatsheet: ```https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf```
    - Там можно почитать как он ставится


- ```sshfs```
    - Нужен для поключения файловой системы (чтоб в проводнике можно было копаться как на своем компе)
    -
    Туториал: [sshfs](https://blog.sedicomm.com/2017/11/10/kak-montirovat-udalennuyu-fajlovuyu-sistemu-ili-katalog-linux-s-pomoshhyu-sshfs-cherez-ssh/)
    - Подключение:
        - ```sudo mkdir /mnt/ailab``` - единоразово
        - ```sudo sshfs -o allow_other,default_permissions,IdentityFile=~/.ssh/id_rsa aidar@10.100.10.50:/home/shamil/ /mnt/ailab```
          - каждый раз когда отвалится маунт


- ```rsync -avz <src> aidar@10.100.10.50:<dst>``` или ```rsync -avz aidar@10.100.10.50:<src> <dst>```
    - Перемещать файлы/папки между хостами/серверами
    - Запускать только с локального терминала


- ```screen -L -Logfile logs.txt bash script.sh```
    - Выполняет bash скрипт, отсоединяет сессию от терминала (можно выключить ноут и скрипт не остановится), пишет логи
      в logs.txt
    - Логи можно чекать через ```cat logs.txt```
    - jupyter / tensorboard можно запускать через ```screeen jupyter```


- ```ssh -N -f -L localhost:<port>:localhost:<port>  aidar@10.100.10.50```
    - маппинг портов удаленной машины и локальной (чтобы можно было запустить удаленно jupyter и приконнеститься
      локально)
    - делается каждый раз, когда меняется wifi / ноут перезагружается


- ```ps aux | grep <program>```
    - посмотреть, раннится ли программа или нет


- ```lsof -i :<port>```
    - посмотреть, какая прога сидит на этом порте


- ```nvidia-smi```
    - посмотреть, занята ли видеокарта и какими процессами


- ```CUDA_VISIBLE_DEVICES=<num> python script.py ...```
    - выполнять питоновский скрипт на num видеокарте

- 