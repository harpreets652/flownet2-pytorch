#!/usr/bin/env python

import argparse
import os
import subprocess
from os.path import *

import colorama
import numpy as np
import setproctitle
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import cv2
from sklearn.metrics import auc

import datasets_video
import losses
import models
from utils import flow_utils, tools, plot_dynamic_update as dyn_plot

# from torchsummary import summary

# fp32 copy of parameters for update
global param_copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")
    parser.add_argument('--train_n_batches', type=int, default=-1,
                        help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
    parser.add_argument('--crop_size', type=int, nargs='+', default=(512, 384),
                        help="Spatial dimension to crop training samples for training")
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--schedule_lr_frequency', type=int, default=0,
                        help='in number of iterations (0 for no schedule)')
    parser.add_argument('--schedule_lr_fraction', type=float, default=10)
    parser.add_argument("--rgb_max", type=float, default=255.)

    parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
    parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

    parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
    parser.add_argument('--validation_n_batches', type=int, default=-1)
    parser.add_argument('--render_validation', action='store_true',
                        help='run inference (save flows to file) and every validation_frequency epoch')

    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_size', type=int, nargs='+', default=[-1, -1],
                        help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
    parser.add_argument('--inference_batch_size', type=int, default=1)
    parser.add_argument('--inference_n_batches', type=int, default=-1)
    parser.add_argument('--save_flow', action='store_true', help='save predicted flows to file')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")

    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--skip_validation', action='store_true')

    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024.,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    # parameters for ROC curve generation
    parser.add_argument('--generate_roc', action='store_true')
    parser.add_argument('--anomaly_labels_path', type=str, help="txt file listing anomalous frames for test videos")
    parser.add_argument("--anomaly_patch_size", default=[2, 2], type=int, nargs=2, help="anomaly patch size")
    parser.add_argument("--diff_win_map", required=False, default=None, help="png window map image")
    parser.add_argument("--pad_output", action='store_true')

    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')

    tools.add_arguments_for_module(parser, losses, argument_for_class='loss', default='L1Loss')

    tools.add_arguments_for_module(parser, torch.optim, argument_for_class='optimizer', default='Adam',
                                   skip_params=['params'])

    # note~ when multiple video data sets, update this
    tools.add_arguments_for_module(parser, datasets_video,
                                   argument_for_class='training_dataset',
                                   default='VideoFiles',
                                   skip_params=['is_cropped'],
                                   parameter_defaults={'root': './anomaly-highway/train'})

    tools.add_arguments_for_module(parser, datasets_video,
                                   argument_for_class='validation_dataset',
                                   default='VideoFiles',
                                   skip_params=['is_cropped'],
                                   parameter_defaults={'root': './anomaly-highway/train',
                                                       'replicates': 1})

    tools.add_arguments_for_module(parser, datasets_video,
                                   argument_for_class='inference_dataset',
                                   default='VideoFiles',
                                   skip_params=['is_cropped'],
                                   parameter_defaults={'root': './anomaly-highway/train',
                                                       'replicates': 1})

    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)

    # Parse the official arguments
    with tools.TimerBlock("Parsing Arguments") as block:
        args = parser.parse_args()
        if args.number_gpus < 0:
            args.number_gpus = torch.cuda.device_count()

        # Get argument defaults (hastag #thisisahack)
        parser.add_argument('--IGNORE', action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))

        # Print all arguments, color the non-defaults
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.model_class = tools.module_to_dict(models)[args.model]
        args.optimizer_class = tools.module_to_dict(torch.optim)[args.optimizer]
        args.loss_class = tools.module_to_dict(losses)[args.loss]

        args.training_dataset_class = tools.module_to_dict(datasets_video)[args.training_dataset]
        args.validation_dataset_class = tools.module_to_dict(datasets_video)[args.validation_dataset]
        args.inference_dataset_class = tools.module_to_dict(datasets_video)[args.inference_dataset]

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        args.current_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip()
        args.log_file = join(args.save, 'args.txt')

        # dict to collect activation gradients (for training debug purpose)
        args.grads = {}

        if args.inference or args.generate_roc:
            args.skip_validation = True
            args.skip_training = True
            args.total_epochs = 1
            args.inference_dir = "{}/inference".format(args.save)
            args.roc_dir = "{}/roc".format(args.save)

    print('Source Code')
    print(('  Current Git Hash: {}\n'.format(args.current_hash)))

    # Change the title for `top` and `pkill` commands
    setproctitle.setproctitle(args.save)

    # Dynamically load the dataset class with parameters passed in via "--argument_[param]=[value]" arguments
    with tools.TimerBlock("Initializing Datasets") as block:
        args.effective_batch_size = args.batch_size * args.number_gpus
        args.effective_inference_batch_size = args.inference_batch_size * args.number_gpus
        args.effective_number_workers = args.number_workers * args.number_gpus
        gpuargs = {'num_workers': args.effective_number_workers,
                   'pin_memory': True,
                   'drop_last': True} if args.cuda else {}
        inf_gpuargs = gpuargs.copy()
        inf_gpuargs['num_workers'] = args.number_workers

        if exists(args.training_dataset_root):
            train_dataset = args.training_dataset_class(args, **tools.kwargs_from_args(args, 'training_dataset'))
            block.log('Training Dataset: {}'.format(args.training_dataset))
            block.log('Training Dataset Size: {}'.format(len(train_dataset)))

            # note~ batch size is set to one video at a time
            train_loader = DataLoader(train_dataset, batch_size=1)

        if exists(args.validation_dataset_root):
            validation_dataset = args.validation_dataset_class(args, True,
                                                               **tools.kwargs_from_args(args, 'validation_dataset'))
            block.log('Validation Dataset: {}'.format(args.validation_dataset))
            block.log(
                'Validation Input: {}'.format(' '.join([str([d for d in x.size()]) for x in validation_dataset[0][0]])))
            block.log('Validation Targets: {}'.format(
                ' '.join([str([d for d in x.size()]) for x in validation_dataset[0][1]])))
            validation_loader = DataLoader(validation_dataset, batch_size=args.effective_batch_size, shuffle=False,
                                           **gpuargs)
        if exists(args.inference_dataset_root):
            inference_dataset = args.inference_dataset_class(args, **tools.kwargs_from_args(args, 'inference_dataset'))
            block.log('Inference Dataset: {}'.format(args.inference_dataset))
            block.log('Inference Dataset Size: {}'.format(len(inference_dataset)))

            inference_loader = DataLoader(inference_dataset, batch_size=1)

    # Dynamically load model and loss class with parameters passed in via
    # "--model_[param]=[value]" or "--loss_[param]=[value]" arguments
    with tools.TimerBlock("Building {} model".format(args.model)) as block:
        class ModelAndLoss(nn.Module):
            def __init__(self, args):
                super(ModelAndLoss, self).__init__()
                kwargs = tools.kwargs_from_args(args, 'model')
                self.model = args.model_class(args, **kwargs)
                kwargs = tools.kwargs_from_args(args, 'loss')
                self.loss = args.loss_class(args, **kwargs)

            def forward(self, data, target, inference=False):
                output = self.model(data)

                loss_values = self.loss(output, target)

                if not inference:
                    return loss_values
                else:
                    return loss_values, output


        model_and_loss = ModelAndLoss(args)
        # debug code: to determine model summary
        # summary(model_and_loss.model, (3, 2, 384, 512), batch_size=1, device="cpu")

        block.log('Effective Batch Size: {}'.format(args.effective_batch_size))
        block.log('Number of parameters: {}'.format(
            sum([p.data.nelement() if p.requires_grad else 0 for p in model_and_loss.parameters()])))

        # assing to cuda or wrap with dataparallel, model and loss
        if args.cuda and (args.number_gpus > 0) and args.fp16:
            block.log('Parallelizing')
            model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(args.number_gpus)))

            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda().half()
            torch.cuda.manual_seed(args.seed)
            param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model_and_loss.parameters()]

        elif args.cuda and args.number_gpus > 0:
            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda()
            block.log('Parallelizing')
            model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(args.number_gpus)))
            torch.cuda.manual_seed(args.seed)

        else:
            block.log('CUDA not being used')
            torch.manual_seed(args.seed)

        # Load weights if needed, otherwise randomly initialize
        if args.resume and os.path.isfile(args.resume):
            block.log("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if not args.inference and not args.generate_roc:
                args.start_epoch = checkpoint['epoch']
            best_err = checkpoint['best_EPE']
            model_and_loss.module.model.load_state_dict(checkpoint['state_dict'])
            block.log("Loaded checkpoint '{}' (at epoch {})".format(args.resume, checkpoint['epoch']))

        elif args.resume and args.inference:
            block.log("No checkpoint found at '{}'".format(args.resume))
            quit()
        else:
            block.log("Random initialization")

        block.log("Initializing save directory: {}".format(args.save))
        if not os.path.exists(args.save):
            os.makedirs(args.save)

        train_logger = SummaryWriter(log_dir=os.path.join(args.save, 'train'), comment='training')
        validation_logger = SummaryWriter(log_dir=os.path.join(args.save, 'validation'), comment='validation')

    # Dynamically load the optimizer with parameters passed in via "--optimizer_[param]=[value]" arguments
    with tools.TimerBlock("Initializing {} Optimizer".format(args.optimizer)) as block:
        kwargs = tools.kwargs_from_args(args, 'optimizer')
        if args.fp16:
            optimizer = args.optimizer_class([p for p in param_copy if p.requires_grad], **kwargs)
        else:
            optimizer = args.optimizer_class([p for p in model_and_loss.parameters() if p.requires_grad], **kwargs)
        for param, default in list(kwargs.items()):
            block.log("{} = {} ({})".format(param, default, type(default)))

    # Log all arguments to file
    if not args.generate_roc:
        for argument, value in sorted(vars(args).items()):
            block.log2file(args.log_file, '{}: {}'.format(argument, value))


    # Reusable function for training and validataion
    def train(input_args, train_epoch, start_iteration, files_loader,
              model, model_optimizer, logger, is_validate=False, offset=0):
        statistics = []
        total_loss = 0

        if is_validate:
            model.eval()
            title = 'Validating Epoch {}'.format(train_epoch)
            input_args.validation_n_batches = np.inf if input_args.validation_n_batches < 0 else input_args.validation_n_batches
            file_progress = tqdm(tools.IteratorTimer(files_loader), ncols=100,
                                 total=np.minimum(len(files_loader), input_args.validation_n_batches), leave=True,
                                 position=offset,
                                 desc=title)
        else:
            model.train()
            title = 'Training Epoch {}'.format(train_epoch)
            input_args.train_n_batches = np.inf if input_args.train_n_batches < 0 else input_args.train_n_batches
            file_progress = tqdm(tools.IteratorTimer(files_loader), ncols=120,
                                 total=np.minimum(len(files_loader), input_args.train_n_batches), smoothing=.9,
                                 miniters=1,
                                 leave=True, position=offset, desc=title)

        last_log_time = file_progress._time()
        for batch_idx, (data_file) in enumerate(file_progress):
            video_dataset = datasets_video.VideoFileDataJIT(input_args, data_file[0])
            video_loader = DataLoader(video_dataset, batch_size=args.effective_batch_size, shuffle=True, **gpuargs)

            global_iteration = start_iteration + batch_idx

            # note~ for debugging purposes
            # video_frame_progress = tqdm(tools.IteratorTimer(video_loader), ncols=120,
            #                            total=len(video_loader), smoothing=0.9, miniters=1,
            #                            leave=True, desc=data_file[0])

            for i_batch, (data, target) in enumerate(video_loader):
                data, target = [Variable(d) for d in data], [Variable(t) for t in target]
                if input_args.cuda and input_args.number_gpus == 1:
                    data, target = [d.cuda(async=True) for d in data], [t.cuda(async=True) for t in target]

                model_optimizer.zero_grad() if not is_validate else None
                losses = model(data[0], target[0])
                losses = [torch.mean(loss_value) for loss_value in losses]
                loss_val = losses[0]  # Collect first loss for weight update
                total_loss += loss_val.data
                loss_values = [v.data for v in losses]

                # gather loss_labels, direct return leads to recursion limit error as it looks for variables to gather'
                loss_labels = list(model.module.loss.loss_labels)

                assert not np.isnan(total_loss.cpu().numpy())

                if not is_validate and input_args.fp16:
                    loss_val.backward()
                    if input_args.gradient_clip:
                        torch.nn.utils.clip_grad_norm(model.parameters(), input_args.gradient_clip)

                    params = list(model.parameters())
                    for i in range(len(params)):
                        param_copy[i].grad = params[i].grad.clone().type_as(params[i]).detach()
                        param_copy[i].grad.mul_(1. / input_args.loss_scale)
                    model_optimizer.step()
                    for i in range(len(params)):
                        params[i].data.copy_(param_copy[i].data)
                elif not is_validate:
                    loss_val.backward()
                    if input_args.gradient_clip:
                        torch.nn.utils.clip_grad_norm(model.parameters(), input_args.gradient_clip)
                    model_optimizer.step()

                # Update hyperparameters if needed
                if not is_validate:
                    tools.update_hyperparameter_schedule(input_args, train_epoch, global_iteration, model_optimizer)
                    loss_labels.append('lr')
                    loss_values.append(model_optimizer.param_groups[0]['lr'])

                    loss_labels.append('load')
                    loss_values.append(file_progress.iterable.last_duration)

            # Print out statistics
            statistics.append(loss_values)
            title = '{} Epoch {}'.format('Validating' if is_validate else 'Training', train_epoch)

            file_progress.set_description(title + ' ' +
                                          tools.format_dictionary_of_losses(tools.flatten_list(loss_labels),
                                                                            statistics[-1]))

            if ((((global_iteration + 1) % input_args.log_frequency) == 0 and not is_validate) or
                    (is_validate and batch_idx == input_args.validation_n_batches - 1)):

                global_iteration = global_iteration if not is_validate else start_iteration

                logger.add_scalar('batch logs per second', len(statistics) / (file_progress._time() - last_log_time),
                                  global_iteration)
                last_log_time = file_progress._time()

                all_losses = np.array(statistics)

                for i, key in enumerate(tools.flatten_list(loss_labels)):
                    if isinstance(all_losses[:, i].item(), torch.Tensor):
                        average_batch = all_losses[:, i].item().mean()
                    else:
                        average_batch = all_losses[:, i].item()

                    logger.add_scalar('average batch ' + str(key), average_batch, global_iteration)
                    logger.add_histogram(str(key), all_losses[:, i], global_iteration)

            # Reset Summary
            statistics = []

            if is_validate and (batch_idx == input_args.validation_n_batches):
                break

            if (not is_validate) and (batch_idx == (input_args.train_n_batches)):
                break

        file_progress.close()

        return total_loss / float(batch_idx + 1), (batch_idx + 1)


    # Reusable function for inference
    def inference(input_args, inf_epoch, files_loader, model, offset=0):
        model.eval()

        if input_args.save_flow:
            flow_folder = "{}/inference/{}.epoch-{}-flow-field".format(input_args.save,
                                                                       input_args.name.replace('/', '.'), inf_epoch)
            if not os.path.exists(flow_folder):
                os.makedirs(flow_folder)

        input_args.inference_n_batches = np.inf if input_args.inference_n_batches < 0 else input_args.inference_n_batches

        progress = tqdm(files_loader, ncols=100, total=np.minimum(len(files_loader), input_args.inference_n_batches),
                        desc='Inferencing ', leave=True, position=offset)

        statistics = []
        total_loss = 0
        for batch_idx, (data_file) in enumerate(progress):
            video_dataset = datasets_video.VideoFileDataJIT(input_args, data_file[0])
            video_loader = DataLoader(video_dataset, batch_size=args.effective_batch_size, shuffle=False, **gpuargs)

            error_plot = dyn_plot.DynamicUpdate(title="L2", x_label="Input Index", y_label="Loss")
            frame_losses, frame_index = [], []

            for i_batch, (data, target) in enumerate(video_loader):
                if input_args.cuda:
                    data, target = [d.cuda(async=True) for d in data], [t.cuda(async=True) for t in target]
                data, target = [Variable(d) for d in data], [Variable(t) for t in target]

                # when ground-truth flows are not available for inference_dataset,
                # the targets are set to all zeros. thus, losses are actually L1 or L2 norms of compute optical flows,
                # depending on the type of loss norm passed in
                with torch.no_grad():
                    losses, output = model(data[0], target[0], inference=True)

                losses = [torch.mean(loss_value) for loss_value in losses]
                loss_val = losses[0]  # Collect first loss for weight update
                total_loss += loss_val.data
                loss_values = [v.data for v in losses]

                # gather loss_labels, direct return leads to recursion limit error as it looks for variables to gather'
                loss_labels = list(model.module.loss.loss_labels)

                statistics.append(loss_values)
                # import IPython; IPython.embed()

                # note~ assuming batch size of 1...pytorch doesn't return a loss for each example in batch
                frame_index.append(i_batch)
                frame_losses.append(loss_values[0].item())
                error_plot.on_running(frame_index, frame_losses)

                if input_args.save_flow or input_args.render_validation:
                    for i in range(input_args.inference_batch_size):
                        _inference_flow = output[i].data.cpu().numpy().transpose(1, 2, 0)

                        if input_args.save_flow:
                            out_path = join(flow_folder, '%06d.flo' % (batch_idx * input_args.inference_batch_size + i))
                            flow_utils.write_flow(out_path, _inference_flow)

                        if input_args.render_validation:
                            target_flow = target[0].cpu().numpy()[i].transpose(1, 2, 0)
                            input_frames = data[0].cpu().numpy()[i].transpose(1, 2, 3, 0).astype(np.uint8)
                            flow_utils.display_results(target_flow, input_frames, _inference_flow)

                progress.update(1)

            error_plot.clear()
            if batch_idx == (input_args.inference_n_batches - 1):
                break

        progress.close()

        return


    def compute_roc(input_args, files_loader, model):
        threshold_range = np.arange(0.002, 1.0, step=0.002)
        setattr(input_args, "anomaly_thresholds", threshold_range.tolist())

        added_frame_counts = False

        print(f"[{datetime.datetime.now()}] Starting ROC test")
        confusion_mats = evaluate_model(input_args, model, files_loader, threshold_range)

        roc_results = []
        tp_list = []
        fp_list = []
        for i, threshold in enumerate(threshold_range):
            confusion_mat = confusion_mats[i]

            total_anomalous_frames = np.sum(confusion_mat[0])
            total_normal_frames = np.sum(confusion_mat[1])

            if not added_frame_counts:
                added_frame_counts = True
                setattr(input_args, "number_anomalous_frames", int(total_anomalous_frames))
                setattr(input_args, "number_normal_frames", int(total_normal_frames))

            true_pos_rate = confusion_mat[0][0] / total_anomalous_frames if total_anomalous_frames > 0 else 0
            false_pos_rate = confusion_mat[1][0] / total_normal_frames

            roc_results.append((threshold, true_pos_rate, false_pos_rate))
            tp_list.append(true_pos_rate)
            fp_list.append(false_pos_rate)

            print(f"[{datetime.datetime.now()}] Confusion Matrix @ threshold: {threshold} \n {confusion_mat}")

        print(f"[{datetime.datetime.now()}] ROC test finished.")
        tools.save_text_data(input_args.roc_dir, roc_results, "roc.csv", "Threshold,TP,FP", "%0.3f")
        print(f"Results saved at {input_args.roc_dir}")

        for argument, value in sorted(vars(input_args).items()):
            block.log2file(args.log_file, '{}: {}'.format(argument, value))

        # compute and save AUC
        auc_score = auc(np.array(fp_list), np.array(tp_list))
        tools.save_text_data(input_args.roc_dir, [auc_score], "auc.csv", "auc-score", "%0.6f")
        print(f"auc score: {auc_score}")

        roc_np = np.array(roc_results)
        if input_args.anomaly_labels_path:
            tools.visualize_roc_curve(roc_np[:, 2], roc_np[:, 1], "False Positive Rate", "True Positive Rate", "ROC")
        else:
            tools.visualize_roc_curve(roc_np[:, 0], roc_np[:, 2],
                                      "Threshold", "False Positive Rate", "Normal Data Curve")

        return


    def evaluate_model(input_args, trained_model, files_loader, thresholds, offset=0):
        trained_model.eval()

        videos_progress = tqdm(files_loader, ncols=100, total=len(files_loader),
                               desc='Inferencing ', leave=True, position=offset)

        anomalous_frames_dict = {}
        if input_args.anomaly_labels_path:
            anomalous_frames_dict = tools.parse_anomalous_frame_labels(input_args.anomaly_labels_path)

        win_difference_map = None
        if input_args.diff_win_map:
            win_difference_map = cv2.imread(input_args.diff_win_map, cv2.IMREAD_GRAYSCALE)

        aggregated_confusion_mat = np.zeros((len(thresholds), 2, 2), dtype=np.int)
        for video_idx, (data_file) in enumerate(videos_progress):
            video_dataset = datasets_video.VideoFileDataJIT(input_args, data_file[0])
            video_loader = DataLoader(video_dataset, batch_size=args.effective_batch_size, shuffle=False, **gpuargs)

            base_name = os.path.basename(data_file[0])
            anomalous_frames = anomalous_frames_dict[base_name] if base_name in anomalous_frames_dict else []

            local_confusion_mat = np.zeros((len(thresholds), 2, 2), dtype=np.int)
            for i_batch, (data, target) in enumerate(video_loader):
                if input_args.cuda:
                    data, target = [d.cuda(async=True) for d in data], [t.cuda(async=True) for t in target]
                data, target = [Variable(d) for d in data], [Variable(t) for t in target]

                # when ground-truth flows are not available for inference_dataset,
                # the targets are set to all zeros. thus, losses are actually L1 or L2 norms of compute optical flows,
                # depending on the type of loss norm passed in
                with torch.no_grad():
                    _, output = trained_model(data[0], target[0], inference=True)

                for i in range(input_args.effective_batch_size):
                    _inference_flow = output[i].data.cpu().numpy().transpose(1, 2, 0)
                    _ground_truth_flow = target[0].cpu().numpy()[i].transpose(1, 2, 0)

                    frame_number = i_batch * input_args.effective_batch_size + i + 1
                    predicted_labels, actual_label = get_labels(input_args, _inference_flow, _ground_truth_flow,
                                                                frame_number, anomalous_frames, thresholds,
                                                                win_difference_map)

                    for k in range(len(predicted_labels)):
                        local_confusion_mat[k][actual_label][predicted_labels[k]] += 1

            videos_progress.update(1)

            aggregated_confusion_mat = np.add(aggregated_confusion_mat, local_confusion_mat)

        videos_progress.close()

        return aggregated_confusion_mat


    def get_labels(arguments, model_output, ground_truth, frame_number,
                   anomaly_frames_label_list, anomaly_thresholds, win_diff_map=None):
        # (x-a)/(b-a)*(beta-alpha) + alpha ; alpha=0, beta=1, a=min(x), b=max(x)
        if win_diff_map is not None:
            x_diff, y_diff = flow_utils.flow_difference(ground_truth, model_output, difference_func="absolute",
                                                        difference_filter_map=win_diff_map)
        else:
            x_diff, y_diff = flow_utils.flow_difference(ground_truth, model_output, (5, 5), difference_func="absolute")

        predicted_x_labels = flow_utils.is_anomalous(arguments, x_diff, anomaly_thresholds)
        predicted_y_labels = flow_utils.is_anomalous(arguments, y_diff, anomaly_thresholds)

        predicted_labels = []
        for i in range(predicted_x_labels.shape[0]):
            predicted_labels.append(predicted_x_labels[i] and predicted_y_labels[i])

        # assume normal frame unless specified in anomalous frames list
        actual_label = flow_utils.LABEL_NORMAL
        if frame_number in anomaly_frames_label_list:
            actual_label = flow_utils.LABEL_ANOMALOUS

        return predicted_labels, actual_label


    # Primary epoch loop
    best_err = 1e8
    progress = tqdm(list(range(args.start_epoch, args.total_epochs + 1)), miniters=1, ncols=100,
                    desc='Overall Progress', leave=True, position=0)
    offset = 1
    last_epoch_time = progress._time()
    global_iteration = 0

    for epoch in progress:

        if args.generate_roc:
            compute_roc(args, inference_loader, model_and_loss)

        if (args.inference or (args.render_validation and ((epoch - 1) % args.validation_frequency) == 0)) \
                and not args.generate_roc:
            inference(input_args=args, inf_epoch=epoch - 1,
                      files_loader=inference_loader,
                      model=model_and_loss, offset=offset)
            offset += 1

        if not args.skip_validation and ((epoch - 1) % args.validation_frequency) == 0:
            validation_loss, _ = train(input_args=args, train_epoch=epoch - 1, start_iteration=global_iteration,
                                       files_loader=validation_loader, model=model_and_loss, model_optimizer=optimizer,
                                       logger=validation_logger, is_validate=True, offset=offset)
            offset += 1

            is_best = False
            if validation_loss < best_err:
                best_err = validation_loss
                is_best = True

            checkpoint_progress = tqdm(ncols=100, desc='Saving Checkpoint', position=offset)
            tools.save_checkpoint({'arch': args.model,
                                   'epoch': epoch,
                                   'state_dict': model_and_loss.module.model.state_dict(),
                                   'best_EPE': best_err},
                                  is_best, args.save, args.model)
            checkpoint_progress.update(1)
            checkpoint_progress.close()
            offset += 1

        if not args.skip_training:
            train_loss, iterations = train(input_args=args, train_epoch=epoch, start_iteration=global_iteration,
                                           files_loader=train_loader, model=model_and_loss, model_optimizer=optimizer,
                                           logger=train_logger, offset=offset)
            global_iteration += iterations
            offset += 1

            # save checkpoint after every validation_frequency number of epochs
            if ((epoch - 1) % args.validation_frequency) == 0:
                checkpoint_progress = tqdm(ncols=100, desc='Saving Checkpoint', position=offset)
                tools.save_checkpoint({'arch': args.model,
                                       'epoch': epoch,
                                       'state_dict': model_and_loss.module.model.state_dict(),
                                       'best_EPE': train_loss},
                                      False, args.save, args.model, filename='train-checkpoint.pth.tar')
                checkpoint_progress.update(1)
                checkpoint_progress.close()

        train_logger.add_scalar('seconds per epoch', progress._time() - last_epoch_time, epoch)
        last_epoch_time = progress._time()
    print("\n")
