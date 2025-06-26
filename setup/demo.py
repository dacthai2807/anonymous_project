import nemo_run as run
from nemo.collections import llm

def configure_recipe(nodes=1, gpus_per_node=1):
    return llm.nemotron3_4b.pretrain_recipe(
        dir="/checkpoints/nemo",
        name="nemotron_pretrain",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
        max_steps=100,
    )

if __name__ == "__main__":
    recipe = configure_recipe()
    executor = run.LocalExecutor()
    run.run(recipe, executor=executor, name="local_nemotron")
