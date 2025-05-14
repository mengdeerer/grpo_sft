# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

import datasets
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, TrlParser, get_peft_config
from open_r1.grpo_trainer_my import GRPOTrainer


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # new change
    dataset = load_dataset("json", data_files=script_args.dataset_name)
    for split in dataset:
        if "output" in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column("output", "solution")

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    # 数据集格式示例
    #   {
    #     "problem": "The operation $\\otimes$ is defined for all nonzero numbers by $a \\otimes b = \\frac{a^{2}}{b}$. Determine $[(1 \\otimes 2) \\otimes 3] - [1 \\otimes (2 \\otimes 3)]$.",
    #     "solution": "1. **Apply the operation $\\otimes$ to the innermost parentheses first:**\n   \\[\n   (1 \\otimes 2) \\otimes 3 = \\left(\\frac{1^2}{2}\\right) \\otimes 3 = \\frac{1}{2} \\otimes 3\n   \\]\n   \\[\n   1 \\otimes (2 \\otimes 3) = 1 \\otimes \\left(\\frac{2^2}{3}\\right) = 1 \\otimes \\frac{4}{3}\n   \\]\n\n2. **Calculate each part using the definition of $\\otimes$:**\n   \\[\n   \\frac{1}{2} \\otimes 3 = \\frac{\\left(\\frac{1}{2}\\right)^2}{3} = \\frac{\\frac{1}{4}}{3} = \\frac{1}{12}\n   \\]\n   \\[\n   1 \\otimes \\frac{4}{3} = \\frac{1^2}{\\frac{4}{3}} = \\frac{1}{\\frac{4}{3}} = \\frac{3}{4}\n   \\]\n\n3. **Subtract the two results:**\n   \\[\n   \\left(\\frac{1}{12}\\right) - \\left(\\frac{3}{4}\\right) = \\frac{1}{12} - \\frac{9}{12} = -\\frac{8}{12} = -\\frac{2}{3}\n   \\]\n\n4. **Conclude with the final answer:**\n   \\[\n   \\boxed{A}\n   \\]",
    #     "answer": "-\\frac{2}{3}",
    #     "difficulty": 1.0
    #   },

    # Format into conversation
    def make_conversation(
        example, prompt_column: str = script_args.dataset_prompt_column
    ):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            raise ValueError(
                f"Dataset Question Field Error: {prompt_column} is not supported."
            )

        prompt.append({"role": "user", "content": example[prompt_column]})

        # 保留solution字段
        return {
            "prompt": prompt,
            "solution": example.get("solution", ""),  # 保留solution字段
        }

    dataset = dataset.map(make_conversation)
    print("*"*100)
    print(dataset.column_names)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    #############################
    # Initialize the GRPO trainer
    #############################
    # new change
    if script_args.dataset_test_split not in dataset:
        # 从训练集中划分出验证集
        train_val_split = dataset[script_args.dataset_train_split].train_test_split(
            test_size=0.1
        )
        train_dataset = train_val_split["train"]
        eval_dataset = train_val_split["test"]
    else:
        train_dataset = dataset[script_args.dataset_train_split]
        eval_dataset = dataset[script_args.dataset_test_split]

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
