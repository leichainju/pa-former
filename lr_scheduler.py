from torch.optim import lr_scheduler


def build_scheduler(config, optimizer, start_epoch=0):
    scheduler_name = config.lr_scheduler.name.lower()

    if scheduler_name == 'multi_step':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=config.lr_scheduler.steps,
            gamma=config.lr_scheduler.gamma,
            last_epoch=start_epoch - 1
        )
    elif scheduler_name == 'ada':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=config.lr_scheduler.mode,
            patience=config.lr_scheduler.patience,
            factor=config.lr_scheduler.gamma,
            min_lr=config.lr_scheduler.min_lr
        )
        config.defrost()
        config.train.lr_scheduler.need_input = True
        config.freeze()
    elif scheduler_name == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config.lr_scheduler.t_max,
            eta_min=config.lr_scheduler.min_lr,
            last_epoch=start_epoch - 1
        )
    elif scheduler_name == 'cos_warm':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_mult=config.lr_scheduler.t_mult,
            T_0=config.lr_scheduler.t_max,
            eta_min=config.lr_scheduler.min_lr,
            last_epoch=start_epoch - 1
        )
    else:
        raise NotImplementedError(f'Scheduler {scheduler_name} is not supported')

    return scheduler
