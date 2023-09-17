import torch, os
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup


optim_fn = optim.AdamW
def build_optimizer(model, total_steps, args):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim_fn(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * total_steps,
        num_training_steps=total_steps,
    )

    # Check if saved optimizer or scheduler states exist
    # if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
    #     os.path.join(args.model_name_or_path, "scheduler.pt")
    # ):
    #     map_location = args.device
    #     optimizer_path = os.path.join(args.model_name_or_path, "optimizer.pt")
    #     scheduler_path = os.path.join(args.model_name_or_path, "scheduler.pt")
    #     # Load in optimizer and scheduler states
    #     optimizer.load_state_dict(torch.load(optimizer_path, map_location=map_location))
    #     scheduler.load_state_dict(torch.load(scheduler_path, map_location=map_location))
    return optimizer, scheduler