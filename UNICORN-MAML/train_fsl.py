import numpy as np
import torch
from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    # with launch_ipdb_on_exception():
    pprint(vars(args))

    set_gpu(args.gpu)
    trainer = FSLTrainer(args)
    trainer.train()
    # trainer.evaluate_test()
    # trainer.final_record()
    trainer.evaluate_test_cross_shot()
    print(args.save_path)
    # print(torch.cuda.is_available())


