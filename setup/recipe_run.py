from nemo.collections import vlm
import nemo_run as run

finetune = vlm.llava15_7b.finetune_recipe(
    name="llava15_7b_finetune",
    dir=f"./checkpoints",
    num_nodes=1,
    num_gpus_per_node=4,
    peft_scheme='lora',  # 'lora', 'none'
)

run.run(finetune, executor=run.LocalExecutor())