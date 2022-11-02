dataset=$1
kill -9 $(ps -ef | grep cli.py.*${dataset}| awk '{print $2}')
# ps -ef | grep cli.py.*agnews