import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2, 3"

seed_numbers = [42, 593, 1774, 65336, 189990]
model_type = 'bert'
absa_type = 'tfm'
tfm_mode = 'finetune'
fix_tfm = 0
task_name = 'rest_total'
warmup_steps = 0
overfit = 0
if task_name == 'laptop14':
    train_batch_size = 32
elif task_name == 'rest_total':
    train_batch_size = 16
else:
    raise Exception("Unsupported dataset %s!!!" % task_name)

for run_id, seed in enumerate(seed_numbers):
    command = "python main.py --model_type %s --absa_type %s --tfm_mode %s --fix_tfm %s " \
              "--model_name_or_path bert-base-uncased --data_dir ./data/%s --task_name %s " \
              "--per_gpu_train_batch_size %s --per_gpu_eval_batch_size 8 --learning_rate 2e-5 " \
              "--max_steps 1500 --warmup_steps %s --do_train --do_eval --do_lower_case " \
              "--seed %s --tagging_schema BIEOS --overfit %s " \
              "--overwrite_output_dir --eval_all_checkpoints --MASTER_ADDR localhost --MASTER_PORT 28512" % (
        model_type, absa_type, tfm_mode, fix_tfm, task_name, task_name, train_batch_size, warmup_steps, seed, overfit)
    output_dir = '%s-%s-%s-%s' % (model_type, absa_type, task_name, tfm_mode)
    if fix_tfm:
        output_dir = '%s-fix' % output_dir
    if overfit:
        output_dir = '%s-overfit' % output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    log_file = '%s/log.txt' % output_dir
    if run_id == 0 and os.path.exists(log_file):
        os.remove(log_file)
    with open(log_file, 'a') as fp:
        fp.write("\nIn run %s/5 (seed %s):\n" % (run_id, seed))
    os.system(command)
    if overfit:
        # only conduct one run
        break
