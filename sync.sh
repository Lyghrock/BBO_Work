#!/bin/bash

REMOTE="weijun@222.223.127.248:/home/weijun/assigned_runs/BBO_Work/"
LOCAL="/home/weijun/BBO_Work/"

rsync -avz \
  --delete \
  -e "ssh -p 18822" \
  --exclude-from='.rsyncignore' \
  $LOCAL/ $REMOTE/   
