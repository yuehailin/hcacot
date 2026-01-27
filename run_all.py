import subprocess
import os
import time
tasks = [



    ('main_LGG.py', 0, '--lr 0.0002 --hard_or_soft False  --model_type HCACoT'),
    ('main_LIHC.py', 0, '--lr 0.0002 --hard_or_soft False   --model_type HCACoT'),
    ('main_LUAD.py', 1, '--lr 0.0002 --hard_or_soft False   --model_type HCACoT'),
    ('main_LUSC.py', 1, '--lr 0.0002 --hard_or_soft False  --model_type HCACoT'),
    ('main_STAD.py', 2, '--lr 0.0002 --hard_or_soft False  --model_type HCACoT'),
    ('main_UCEC.py', 2, '--lr 0.0001 --hard_or_soft False  --model_type HCACoT'),
    ('main_ESCA.py', 3, '--lr 0.0002 --hard_or_soft False  --model_type HCACoT'),
    ('main_COAD.py', 3, '--lr 0.0002 --hard_or_soft False  --model_type HCACoT')
   
]

for i, (script, gpu_id,args_str) in enumerate(tasks, start=1):
    log_file = f'{os.path.splitext(script)[0]}.log'
    cmd = f'nohup bash -c "CUDA_VISIBLE_DEVICES={gpu_id} python {script} {args_str}" > {log_file} 2>&1 &'

    print(f"[{i}/{len(tasks)}] 正在启动 {script} (GPU {gpu_id}) ...")
    subprocess.call(cmd, shell=True)
    print(f"✅ 已启动 {script}，日志输出到 {log_file}")

    if i < len(tasks):  
        print("⏳ 等待 20 秒后启动下一个任务...")
        time.sleep(5)

print("🎉 所有脚本已分配 GPU 并用 nohup 启动完毕。")
