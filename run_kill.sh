dataset=$1
kill -9 $(ps -ef | grep cli.py.*${dataset}| awk '{print $2}')
# ps -ef | grep cli.py.*agnews

# kill specific dataset according its pattern_ids
kill -9 $(ps -ef | grep pattern_ids.*0.*--data_dir| awk '{print $2}')

# or refer to them directlt
kill -9 $(ps -ef | grep 'task_name agnews' | awk '{print $2}')