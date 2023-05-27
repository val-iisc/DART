import collections
import json
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from domainbed.datasets import get_dataset, split_dataset
from domainbed import algorithms
from domainbed.evaluator import Evaluator
from domainbed.lib import misc
from domainbed.lib import swa_utils
from domainbed.lib.query import Q
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import swad as swad_module

# if torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"


def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")


def interpolate_algos(sd1, sd2, sd3, sd4, sd5, sd6):
    return {key: (sd1[key] + sd2[key] + sd3[key] +sd4[key]+ sd5[key] +sd6[key])/6 for key in sd1.keys()}
    
def train(test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env=None):
    logger.info("")
    # n_steps = 1
    #######################################################
    # setup dataset & loader
    #######################################################
    args.real_test_envs = test_envs  # for log
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams, algorithm_class)
    test_splits = []
    # if hparams.indomain_test > 0.0:
    #     logger.info("!!! In-domain test mode On !!!")
    #     assert hparams["val_augment"] is False, (
    #         "indomain_test split the val set into val/test sets. "
    #         "Therefore, the val set should be not augmented."
    #     )
    #     val_splits = []
    #     for env_i, (out_split, _weights) in enumerate(out_splits):
    #         n = len(out_split) // 2
    #         seed = misc.seed_hash(args.trial_seed, env_i)
    #         val_split, test_split = split_dataset(out_split, n, seed=seed)
    #         val_splits.append((val_split, None))
    #         test_splits.append((test_split, None))
    #         logger.info(
    #             "env %d: out (#%d) -> val (#%d) / test (#%d)"
    #             % (env_i, len(out_split), len(val_split), len(test_split))
    #         )
    #     out_splits = val_splits

    if target_env is not None:
        testenv_name = f"te_{dataset.environments[target_env]}"
        logger.info(f"Target env = {target_env}")
    else:
        testenv_properties = [str(dataset.environments[i]) for i in test_envs]
        testenv_name = "te_" + "_".join(testenv_properties)

    logger.info(
        "Testenv name escaping {} -> {}".format(testenv_name, testenv_name.replace(".", ""))
    )
    testenv_name = testenv_name.replace(".", "")
    logger.info(f"Test envs = {test_envs}, name = {testenv_name}")

    n_envs = len(dataset)
    train_envs = sorted(set(range(n_envs)) - set(test_envs))
    iterator = misc.SplitIterator(test_envs)
    batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=np.int)
    batch_sizes50 = np.full([n_envs], int(hparams["batch_size"]*5*0.4), dtype=np.int)
    batch_sizes25 = np.full([n_envs], int(hparams["batch_size"]*5*0.15), dtype=np.int)

    
    batch_sizes[test_envs] = 0
    batch_sizes = batch_sizes.tolist()
    batch_sizes50[test_envs] = 0
    batch_sizes50 = batch_sizes50.tolist()
    batch_sizes25[test_envs] = 0
    batch_sizes25 = batch_sizes25.tolist()

    logger.info(f"Batch sizes for CombERM branch: {batch_sizes} (total={sum(batch_sizes)})")
    logger.info(f"Own domain Batch sizes for each domain: {batch_sizes50} (total={sum(batch_sizes50)})")
    logger.info(f"Other domain Batch sizes for each domain: {batch_sizes25} (total={sum(batch_sizes25)})")

    # calculate steps per epoch
    steps_per_epochs = [
        len(env) / batch_size for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    steps_per_epochs50 = [
        len(env) / batch_size50 for (env, _), batch_size50 in iterator.train(zip(in_splits, batch_sizes50))
    ]
    steps_per_epochs25 = [
        len(env) / batch_size25 for (env, _), batch_size25 in iterator.train(zip(in_splits, batch_sizes25))
    ]
    steps_per_epoch = min(steps_per_epochs)
    steps_per_epoch50 = min(steps_per_epochs50)
    steps_per_epoch25 = min(steps_per_epochs25)
    
    # epoch is computed by steps_per_epoch
    prt_steps = ", ".join([f"{step:.2f}" for step in steps_per_epochs])
    prt_steps50 = ", ".join([f"{step:.2f}" for step in steps_per_epochs50])
    prt_steps25 = ", ".join([f"{step:.2f}" for step in steps_per_epochs25])
    logger.info(f"steps-per-epoch for CombERM : {prt_steps} -> min = {steps_per_epoch:.2f}")
    logger.info(f"steps-per-epoch for own domain: {prt_steps50} -> min = {steps_per_epoch50:.2f}")
    logger.info(f"steps-per-epoch for other domain: {prt_steps25} -> min = {steps_per_epoch25:.2f}")
    
    # setup loaders
    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=batch_size,
            num_workers=dataset.N_WORKERS
        )
        for (env, env_weights), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    train_loaders50 = [
        InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=batch_size50,
            num_workers=dataset.N_WORKERS
        )
        for (env, env_weights), batch_size50 in iterator.train(zip(in_splits, batch_sizes50))
    ]
    train_loaders25 = [
        InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=batch_size25,
            num_workers=dataset.N_WORKERS
        )
        for (env, env_weights), batch_size25 in iterator.train(zip(in_splits, batch_sizes25))
    ]

    # setup eval loaders
    eval_loaders_kwargs = []
    for i, (env, _) in enumerate(in_splits + out_splits + test_splits):
        batchsize = hparams["test_batchsize"]
        loader_kwargs = {"dataset": env, "batch_size": batchsize, "num_workers": dataset.N_WORKERS}
        if args.prebuild_loader:
            loader_kwargs = FastDataLoader(**loader_kwargs)
        eval_loaders_kwargs.append(loader_kwargs)

    eval_weights = [None for _, weights in (in_splits + out_splits + test_splits)]
    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]
    eval_loader_names += ["env{}_inTE".format(i) for i in range(len(test_splits))]
    eval_meta = list(zip(eval_loader_names, eval_loaders_kwargs, eval_weights))

    #######################################################
    # setup algorithm (model)
    #######################################################
    algorithmCE1 = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams,
    )
    algorithmCE2 = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams,
    )
    algorithmCE3 = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams,
    )
    algorithmCE4 = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams,
    )
    algorithmCE5 = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams,
    )
    algorithmCE6 = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams,
    )

    algorithmCE1.cuda()
    algorithmCE2.cuda()
    algorithmCE3.cuda()
    algorithmCE4.cuda()
    algorithmCE5.cuda()
    algorithmCE6.cuda()

    n_params = sum([p.numel() for p in algorithmCE1.parameters()])
    logger.info("# of params = %d" % n_params)

    train_minibatches_iterator = zip(*train_loaders)
    train_minibatches_iterator50 = zip(*train_loaders50)
    train_minibatches_iterator25 = zip(*train_loaders25)
    # train_minibatches_iterator25b = zip(*train_loaders25b)
    
    checkpoint_vals = collections.defaultdict(lambda: [])

    #######################################################
    # start training loop
    #######################################################
    evaluator = Evaluator(
        test_envs,
        eval_meta,
        n_envs,
        logger,
        evalmode=args.evalmode,
        debug=args.debug,
        target_env=target_env,
    )

    # swad = None
    # if hparams["swad"]:
    #     swad_algorithm = swa_utils.AveragedModel(algorithm)
    #     swad_cls = getattr(swad_module, hparams["swad"])
    #     swad = swad_cls(evaluator, **hparams.swad_kwargs)
    
    swad1 = None
    if hparams["swad"]:
        swad_algorithm1 = swa_utils.AveragedModel(algorithmCE1)
        swad_cls1 = getattr(swad_module, "LossValley")
        swad1 = swad_cls1(evaluator, **hparams.swad_kwargs)
    swad2 = None
    if hparams["swad"]:
        swad_algorithm2 = swa_utils.AveragedModel(algorithmCE2)
        swad_cls2 = getattr(swad_module, "LossValley")
        swad2 = swad_cls2(evaluator, **hparams.swad_kwargs)
    swad3 = None
    if hparams["swad"]:
        swad_algorithm3 = swa_utils.AveragedModel(algorithmCE3)
        swad_cls3 = getattr(swad_module, "LossValley")
        swad3 = swad_cls3(evaluator, **hparams.swad_kwargs)
    swad4 = None
    if hparams["swad"]:
        swad_algorithm4 = swa_utils.AveragedModel(algorithmCE4)
        swad_cls4 = getattr(swad_module, "LossValley")
        swad4 = swad_cls4(evaluator, **hparams.swad_kwargs)
    swad5 = None
    if hparams["swad"]:
        swad_algorithm5 = swa_utils.AveragedModel(algorithmCE5)
        swad_cls5 = getattr(swad_module, "LossValley")
        swad5 = swad_cls5(evaluator, **hparams.swad_kwargs)
    swad6 = None
    if hparams["swad"]:
        swad_algorithm6 = swa_utils.AveragedModel(algorithmCE6)
        swad_cls6 = getattr(swad_module,"LossValley")
        swad6 = swad_cls6(evaluator, **hparams.swad_kwargs)

    last_results_keys = None
    records = []
    records_inter = []
    epochs_path = args.out_dir / "results.jsonl"

    for step in range(n_steps):
        step_start_time = time.time()
        
        # batches_dictlist: [ {x:<env1x> ,y:<env1y> }, {x:<env2x> ,y: }, {x: ,y: } ]
        batches_dictlist = next(train_minibatches_iterator)
        batches_dictlist50 = next(train_minibatches_iterator50)
        batches_dictlist25 = next(train_minibatches_iterator25)
        
        # batches: {x: [ <env1x>,<env2x>,<env3x> ] ,y: [ <env1y>,<env2y>,<env3y> ] }
        batchesCE = misc.merge_dictlist(batches_dictlist)
        batches1 = {'x': [ batches_dictlist50[0]['x'],batches_dictlist25[1]['x'],batches_dictlist25[2]['x'],batches_dictlist25[3]['x'],batches_dictlist25[4]['x'] ],
                    'y': [ batches_dictlist50[0]['y'],batches_dictlist25[1]['y'],batches_dictlist25[2]['y'],batches_dictlist25[3]['y'],batches_dictlist25[4]['y'] ] }
        batches2 = {'x': [ batches_dictlist25[0]['x'],batches_dictlist50[1]['x'],batches_dictlist25[2]['x'],batches_dictlist25[3]['x'],batches_dictlist25[4]['x'] ],
                    'y': [ batches_dictlist25[0]['y'],batches_dictlist50[1]['y'],batches_dictlist25[2]['y'],batches_dictlist25[3]['y'],batches_dictlist25[4]['y'] ] }
        batches3 = {'x': [ batches_dictlist25[0]['x'],batches_dictlist25[1]['x'],batches_dictlist50[2]['x'],batches_dictlist25[3]['x'],batches_dictlist25[4]['x'] ],
                    'y': [ batches_dictlist25[0]['y'],batches_dictlist25[1]['y'],batches_dictlist50[2]['y'],batches_dictlist25[3]['y'],batches_dictlist25[4]['y'] ] }
        batches4 = {'x': [ batches_dictlist25[0]['x'],batches_dictlist25[1]['x'],batches_dictlist25[2]['x'],batches_dictlist50[3]['x'],batches_dictlist25[4]['x'] ],
                    'y': [ batches_dictlist25[0]['y'],batches_dictlist25[1]['y'],batches_dictlist25[2]['y'],batches_dictlist50[3]['y'],batches_dictlist25[4]['y'] ] }
        batches5 = {'x': [ batches_dictlist25[0]['x'],batches_dictlist25[1]['x'],batches_dictlist25[2]['x'],batches_dictlist25[3]['x'],batches_dictlist50[4]['x'] ],
                    'y': [ batches_dictlist25[0]['y'],batches_dictlist25[1]['y'],batches_dictlist25[2]['y'],batches_dictlist25[3]['y'],batches_dictlist50[4]['y'] ] }
        
        # to device
        batchesCE = {key: [tensor.to(device) for tensor in tensorlist] for key, tensorlist in batchesCE.items()}
        batches1 = {key: [tensor.to(device) for tensor in tensorlist] for key, tensorlist in batches1.items()}
        batches2 = {key: [tensor.to(device) for tensor in tensorlist] for key, tensorlist in batches2.items()}
        batches3 = {key: [tensor.to(device) for tensor in tensorlist] for key, tensorlist in batches3.items()}
        batches4 = {key: [tensor.to(device) for tensor in tensorlist] for key, tensorlist in batches4.items()}
        batches5 = {key: [tensor.to(device) for tensor in tensorlist] for key, tensorlist in batches5.items()}

        inputsCE = {**batchesCE, "step":step}
        inputs1 = {**batches1, "step": step}
        inputs2 = {**batches2, "step": step}
        inputs3 = {**batches3, "step": step}
        inputs4 = {**batches4, "step": step}
        inputs5 = {**batches5, "step": step}

        step_valsCE1 = algorithmCE1.update(**inputs1)
        step_valsCE2 = algorithmCE2.update(**inputs2)
        step_valsCE3 = algorithmCE3.update(**inputs3)
        step_valsCE4 = algorithmCE4.update(**inputs4)
        step_valsCE5 = algorithmCE5.update(**inputs5)
        step_valsCE6 = algorithmCE6.update(**inputsCE)        
        
        
        for key, val in step_valsCE1.items():
            checkpoint_vals['1_'+key].append(val)
        for key, val in step_valsCE2.items():
            checkpoint_vals['2_'+key].append(val)
        for key, val in step_valsCE3.items():
            checkpoint_vals['3_'+key].append(val)
        for key, val in step_valsCE4.items():
            checkpoint_vals['4_'+key].append(val)
        for key, val in step_valsCE5.items():
            checkpoint_vals['5_'+key].append(val)
        for key, val in step_valsCE6.items():
            checkpoint_vals['6_'+key].append(val)
        checkpoint_vals["step_time"].append(time.time() - step_start_time)

        if swad1:
            # swad_algorithm is segment_swa for swad
            swad_algorithm1.update_parameters(algorithmCE1, step=step)
            swad_algorithm2.update_parameters(algorithmCE2, step=step)
            swad_algorithm3.update_parameters(algorithmCE3, step=step)
            swad_algorithm4.update_parameters(algorithmCE4, step=step)
            swad_algorithm5.update_parameters(algorithmCE5, step=step)
            swad_algorithm6.update_parameters(algorithmCE6, step=step)

        if step % checkpoint_freq == 0:
            results = {
                "step": step,
                "epoch": step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            eval_start_time = time.time()
            summaries1 = evaluator.evaluate(algorithmCE1, suffix='_1')
            summaries2 = evaluator.evaluate(algorithmCE2, suffix='_2')
            summaries3 = evaluator.evaluate(algorithmCE3, suffix='_3')
            summaries4 = evaluator.evaluate(algorithmCE4, suffix='_4')
            summaries5 = evaluator.evaluate(algorithmCE5, suffix='_5')
            summaries6 = evaluator.evaluate(algorithmCE6, suffix='_6')
            results["eval_time"] = time.time() - eval_start_time

            # results = (epochs, loss, step, step_time)
            results_keys = list(summaries1.keys()) + list(summaries2.keys()) + list(summaries3.keys()) + list(summaries4.keys()) + list(summaries5.keys()) + list(summaries6.keys()) + list(results.keys())
            # merge results
            results.update(summaries1)
            results.update(summaries2)
            results.update(summaries3)
            results.update(summaries4)
            results.update(summaries5)
            results.update(summaries6)

            # print
            if results_keys != last_results_keys:
                logger.info(misc.to_row(results_keys))
                last_results_keys = results_keys
            logger.info(misc.to_row([results[key] for key in results_keys]))
            records.append(copy.deepcopy(results))

            # update results to record
            results.update({"hparams": dict(hparams), "args": vars(args)})

            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True, default=json_handler) + "\n")

            checkpoint_vals = collections.defaultdict(lambda: [])

            writer.add_scalars_with_prefix(summaries1, step, f"{testenv_name}/summary1/")
            writer.add_scalars_with_prefix(summaries2, step, f"{testenv_name}/summary2/")
            writer.add_scalars_with_prefix(summaries3, step, f"{testenv_name}/summary3/")
            writer.add_scalars_with_prefix(summaries4, step, f"{testenv_name}/summary4/")
            writer.add_scalars_with_prefix(summaries5, step, f"{testenv_name}/summary5/")
            writer.add_scalars_with_prefix(summaries6, step, f"{testenv_name}/summary6/")
            # writer.add_scalars_with_prefix(accuracies, step, f"{testenv_name}/all/")

            if args.model_save and step >= args.model_save:
                ckpt_dir = args.out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)

                test_env_str = ",".join(map(str, test_envs))
                filename = "TE{}_{}.pth".format(test_env_str, step)
                if len(test_envs) > 1 and target_env is not None:
                    train_env_str = ",".join(map(str, train_envs))
                    filename = f"TE{target_env}_TR{train_env_str}_{step}.pth"
                path = ckpt_dir / filename

                save_dict = {
                    "args": vars(args),
                    "model_hparams": dict(hparams),
                    "test_envs": test_envs,
                    "model_dict1": algorithmCE1.cpu().state_dict(),
                    "model_dict2": algorithmCE2.cpu().state_dict(),
                    "model_dict3": algorithmCE3.cpu().state_dict(),
                    "model_dict4": algorithmCE4.cpu().state_dict(),
                    "model_dict5": algorithmCE5.cpu().state_dict(),
                    "model_dict6": algorithmCE6.cpu().state_dict(),
                }
                algorithmCE1.cuda()
                algorithmCE2.cuda()
                algorithmCE3.cuda()
                algorithmCE4.cuda()
                algorithmCE5.cuda()
                algorithmCE6.cuda()
                if not args.debug:
                    torch.save(save_dict, path)
                else:
                    logger.debug("DEBUG Mode -> no save (org path: %s)" % path)

            # swad
            if swad1:
                def prt_results_fn(results, avgmodel):
                    step_str = f" [{avgmodel.start_step}-{avgmodel.end_step}]"
                    row = misc.to_row([results[key] for key in results_keys if key in results])
                    logger.info(row + step_str)

                swad1.update_and_evaluate(
                    swad_algorithm1, results["comb_val_1"], results["comb_val_loss_1"], prt_results_fn
                )
                swad2.update_and_evaluate(
                    swad_algorithm2, results["comb_val_2"], results["comb_val_loss_2"], prt_results_fn
                )
                swad3.update_and_evaluate(
                    swad_algorithm3, results["comb_val_3"], results["comb_val_loss_3"], prt_results_fn
                )
                swad4.update_and_evaluate(
                    swad_algorithm4, results["comb_val_4"], results["comb_val_loss_4"], prt_results_fn
                )
                swad5.update_and_evaluate(
                    swad_algorithm5, results["comb_val_5"], results["comb_val_loss_5"], prt_results_fn
                )
                swad6.update_and_evaluate(
                    swad_algorithm6, results["comb_val_6"], results["comb_val_loss_6"], prt_results_fn
                )

                # if hasattr(swad, "dead_valley") and swad.dead_valley:
                #     logger.info("SWAD valley is dead -> early stop !")
                #     break
                if hasattr(swad1, "dead_valley") and swad1.dead_valley:
                    logger.info("SWAD valley is dead for 1 -> early stop !")
                if hasattr(swad2, "dead_valley") and swad2.dead_valley:
                    logger.info("SWAD valley is dead for 2 -> early stop !")
                if hasattr(swad3, "dead_valley") and swad3.dead_valley:
                    logger.info("SWAD valley is dead for 3 -> early stop !")
                if hasattr(swad4, "dead_valley") and swad4.dead_valley:
                    logger.info("SWAD valley is dead for 4 -> early stop !")
                if hasattr(swad5, "dead_valley") and swad5.dead_valley:
                    logger.info("SWAD valley is dead for 5 -> early stop !")
                if hasattr(swad6, "dead_valley") and swad6.dead_valley:
                    logger.info("SWAD valley is dead for 6 -> early stop !")
                    
                
                if (hparams["model"]=='clip_vit-b16') and (step % 2000 == 0):
                    swad_algorithm1 = swa_utils.AveragedModel(algorithmCE1)  # reset
                    swad_algorithm2 = swa_utils.AveragedModel(algorithmCE2)
                    swad_algorithm3 = swa_utils.AveragedModel(algorithmCE3)
                    swad_algorithm4 = swa_utils.AveragedModel(algorithmCE4)
                    swad_algorithm5 = swa_utils.AveragedModel(algorithmCE5)
                    swad_algorithm6 = swa_utils.AveragedModel(algorithmCE6)

        if step % args.tb_freq == 0:
            # add step values only for tb log
            writer.add_scalars_with_prefix(step_valsCE1, step, f"{testenv_name}/summary1/")
            writer.add_scalars_with_prefix(step_valsCE2, step, f"{testenv_name}/summary2/")
            writer.add_scalars_with_prefix(step_valsCE3, step, f"{testenv_name}/summary3/")
            writer.add_scalars_with_prefix(step_valsCE4, step, f"{testenv_name}/summary4/")
            writer.add_scalars_with_prefix(step_valsCE5, step, f"{testenv_name}/summary5/")
            writer.add_scalars_with_prefix(step_valsCE6, step, f"{testenv_name}/summary6/")
        
        if step%args.inter_freq==0 and step!=0:
            if args.algorithm in ['DANN', 'CDANN']:
                inter_state_dict = interpolate_algos(algorithmCE1.featurizer.state_dict(), algorithCE2.featurizer.state_dict(), algorithmCE3.featurizer.state_dict(), algorithmCE4.featurizer.state_dict(), algorithmCE5.featurizer.state_dict(), algorithCE6.featurizer.state_dict())
                algorithmCE1.featurizer.load_state_dict(inter_state_dict)
                algorithmCE2.featurizer.load_state_dict(inter_state_dict)
                algorithmCE3.featurizer.load_state_dict(inter_state_dict)
                algorithmCE4.featurizer.load_state_dict(inter_state_dict)
                algorithmCE5.featurizer.load_state_dict(inter_state_dict)
                algorithmCE6.featurizer.load_state_dict(inter_state_dict)
                inter_state_dict2 = interpolate_algos(algorithmCE1.classifier.state_dict(), algorithCE2.classifier.state_dict(), algorithmCE3.classifier.state_dict(), algorithmCE4.classifier.state_dict(), algorithmCE5.classifier.state_dict(), algorithCE6.classifier.state_dict())
                algorithmCE1.classifier.load_state_dict(inter_state_dict)
                algorithmCE2.classifier.load_state_dict(inter_state_dict)
                algorithmCE3.classifier.load_state_dict(inter_state_dict)
                algorithmCE4.classifier.load_state_dict(inter_state_dict)
                algorithmCE5.classifier.load_state_dict(inter_state_dict)
                algorithmCE6.classifier.load_state_dict(inter_state_dict)
                inter_state_dict3 = interpolate_algos(algorithmCE1.discriminator.state_dict(), algorithCE2.discriminator.state_dict(), algorithmCE3.discriminator.state_dict(), algorithmCE4.discriminator.state_dict(), algorithmCE5.discriminator.state_dict(), algorithCE6.discriminator.state_dict())
                algorithmCE1.discriminator.load_state_dict(inter_state_dict)
                algorithmCE2.discriminator.load_state_dict(inter_state_dict)
                algorithmCE3.discriminator.load_state_dict(inter_state_dict)
                algorithmCE4.discriminator.load_state_dict(inter_state_dict)
                algorithmCE5.discriminator.load_state_dict(inter_state_dict)
                algorithmCE6.discriminator.load_state_dict(inter_state_dict)
            elif args.algorithm in ['SagNet']:
                inter_state_dict = interpolate_algos(algorithmCE1.network_f.state_dict(), algorithmCE2.network_f.state_dict(), algorithmCE3.network_f.state_dict(), algorithmCE4.network_f.state_dict(), algorithmCE5.network_f.state_dict(), algorithmCE6.network_f.state_dict())
                algorithmCE1.network_f.load_state_dict(inter_state_dict)
                algorithmCE2.network_f.load_state_dict(inter_state_dict)
                algorithmCE3.network_f.load_state_dict(inter_state_dict)
                algorithmCE4.network_f.load_state_dict(inter_state_dict)
                algorithmCE5.network_f.load_state_dict(inter_state_dict)
                algorithmCE6.network_f.load_state_dict(inter_state_dict)
                inter_state_dict2 = interpolate_algos(algorithmCE1.network_c.state_dict(), algorithmCE2.network_c.state_dict(), algorithmCE3.network_c.state_dict(), algorithmCE4.network_c.state_dict(), algorithmCE5.network_c.state_dict(), algorithmCE6.network_c.state_dict())
                algorithmCE1.network_c.load_state_dict(inter_state_dict)
                algorithmCE2.network_c.load_state_dict(inter_state_dict)
                algorithmCE3.network_c.load_state_dict(inter_state_dict)
                algorithmCE4.network_c.load_state_dict(inter_state_dict)
                algorithmCE5.network_c.load_state_dict(inter_state_dict)
                algorithmCE6.network_c.load_state_dict(inter_state_dict)
                inter_state_dict3 = interpolate_algos(algorithmCE1.network_s.state_dict(), algorithmCE2.network_s.state_dict(), algorithmCE3.network_s.state_dict(), algorithmCE4.network_s.state_dict(), algorithmCE5.network_s.state_dict(), algorithmCE6.network_s.state_dict())
                algorithmCE1.network_s.load_state_dict(inter_state_dict)
                algorithmCE2.network_s.load_state_dict(inter_state_dict)
                algorithmCE3.network_s.load_state_dict(inter_state_dict)
                algorithmCE4.network_s.load_state_dict(inter_state_dict)
                algorithmCE5.network_s.load_state_dict(inter_state_dict)
                algorithmCE6.network_s.load_state_dict(inter_state_dict)
            else:
                inter_state_dict = interpolate_algos(algorithmCE1.network.state_dict(), algorithmCE2.network.state_dict(), algorithmCE3.network.state_dict(), algorithmCE4.network.state_dict(), algorithmCE5.network.state_dict(), algorithmCE6.network.state_dict())
                algorithmCE1.network.load_state_dict(inter_state_dict)
                algorithmCE2.network.load_state_dict(inter_state_dict)
                algorithmCE3.network.load_state_dict(inter_state_dict)
                algorithmCE4.network.load_state_dict(inter_state_dict)
                algorithmCE5.network.load_state_dict(inter_state_dict)
                algorithmCE6.network.load_state_dict(inter_state_dict)
            
            logger.info(f"Evaluating interpolated model at {step} step")
            summaries_inter = evaluator.evaluate(algorithmCE1, suffix='_from_inter')
            inter_results = {"inter_step": step, "inter_epoch": step / steps_per_epoch}
            inter_results_keys = list(summaries_inter.keys()) + list(inter_results.keys())
            inter_results.update(summaries_inter)
            logger.info(misc.to_row([inter_results[key] for key in inter_results_keys]))
            records_inter.append(copy.deepcopy(inter_results))
            writer.add_scalars_with_prefix(summaries_inter, step, f"{testenv_name}/summary_inter/")

    # find best
    logger.info("---")
    # print(records)
    records = Q(records)
    records_inter = Q(records_inter)
    
    # print(len(records))
    # print(records)
    
    # 1
    oracle_best1 = records.argmax("test_out_1")["test_in_1"]
    iid_best1 = records.argmax("comb_val_1")["test_in_1"]
    inDom1 = records.argmax("comb_val_1")["comb_val_1"]
    # own_best1 = records.argmax("own_val_from_first")["test_in_from_first"]
    last1 = records[-1]["test_in_1"]
    # 2
    oracle_best2 = records.argmax("test_out_2")["test_in_2"]
    iid_best2 = records.argmax("comb_val_2")["test_in_2"]
    inDom2 = records.argmax("comb_val_2")["comb_val_2"]
    # own_best2 = records.argmax("own_val_from_second")["test_in_from_second"]
    last2 = records[-1]["test_in_2"]
    # 3
    oracle_best3 = records.argmax("test_out_3")["test_in_3"]
    iid_best3 = records.argmax("comb_val_3")["test_in_3"]
    inDom3 = records.argmax("comb_val_3")["comb_val_3"]
    # own_best3 = records.argmax("own_val_from_third")["test_in_from_third"]
    last3 = records[-1]["test_in_3"]
    # CE
    oracle_best4 = records.argmax("test_out_4")["test_in_4"]
    iid_best4 = records.argmax("comb_val_4")["test_in_4"]
    inDom4 = records.argmax("comb_val_4")["comb_val_4"]
    last4 = records[-1]["test_in_4"]
    
    oracle_best5 = records.argmax("test_out_5")["test_in_5"]
    iid_best5 = records.argmax("comb_val_5")["test_in_5"]
    inDom5 = records.argmax("comb_val_5")["comb_val_5"]
    last5 = records[-1]["test_in_5"]
    
    oracle_best6 = records.argmax("test_out_6")["test_in_6"]
    iid_best6 = records.argmax("comb_val_6")["test_in_6"]
    inDom6 = records.argmax("comb_val_6")["comb_val_6"]
    last6 = records[-1]["test_in_6"]
    # inter
    oracle_best_inter = records_inter.argmax("test_out_from_inter")["test_in_from_inter"]
    iid_best_inter = records_inter.argmax("comb_val_from_inter")["test_in_from_inter"]
    inDom_inter = records_inter.argmax("comb_val_from_inter")["comb_val_from_inter"]

    # if hparams.indomain_test:
    #     # if test set exist, use test set for indomain results
    #     in_key = "train_inTE"
    # else:
    #     in_key = "train_out"

    # iid_best_indomain = records.argmax("train_out")[in_key]
    # last_indomain = records[-1][in_key]

    ret = {
        "oracle_1": oracle_best1,
        "iid_1": iid_best1,
        # "own_1": own_best1,
        "inDom1": inDom1,
        "last_1": last1,
        "oracle_2": oracle_best2,
        "iid_2": iid_best2,
        # "own_2": own_best2,
        "inDom2":inDom2,
        "last_2": last2,
        "oracle_3": oracle_best3,
        "iid_3": iid_best3,
        # "own_3": own_best3,
        "inDom3":inDom3,
        "last_3": last3,
        "oracle_4": oracle_best4,
        "iid_4": iid_best4,
        "inDom4": inDom4,
        "last_4": last4,
        
        "oracle_5": oracle_best5,
        "iid_5": iid_best5,
        "inDom5": inDom5,
        "last_5": last5,
        
        "oracle_6": oracle_best6,
        "iid_6": iid_best6,
        "inDom6": inDom6,
        "last_6": last6,
        # "last (inD)": last_indomain,
        # "iid (inD)": iid_best_indomain,
        "oracle_inter": oracle_best_inter,
        "iid_inter": iid_best_inter,
        "inDom_inter":inDom_inter,
    }

    # Evaluate SWAD
    if swad1:
        swad_algorithm1 = swad1.get_final_model()
        swad_algorithm2 = swad2.get_final_model()
        swad_algorithm3 = swad3.get_final_model()
        swad_algorithm4 = swad4.get_final_model()
        swad_algorithm5 = swad5.get_final_model()
        swad_algorithm6 = swad6.get_final_model()
        if hparams["freeze_bn"] is False:
            n_steps = 500 if not args.debug else 10
            logger.warning(f"Update SWAD BN statistics for {n_steps} steps ...")
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm1, n_steps)
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm2, n_steps)
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm3, n_steps)
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm4, n_steps)
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm5, n_steps)
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm6, n_steps)

        logger.warning("Evaluate SWAD ...")
        summaries_swad1 = evaluator.evaluate(swad_algorithm1, suffix='_s1')
        summaries_swad2 = evaluator.evaluate(swad_algorithm2, suffix='_s2')
        summaries_swad3 = evaluator.evaluate(swad_algorithm3, suffix='_s3')
        summaries_swad4 = evaluator.evaluate(swad_algorithm4, suffix='_s4')
        summaries_swad5 = evaluator.evaluate(swad_algorithm5, suffix='_s5')
        summaries_swad6 = evaluator.evaluate(swad_algorithm6, suffix='_s6')
        # accuracies, summaries = evaluator.evaluate(swad_algorithm)
        
        # results = {**summaries, **accuracies}
        # start = swad_algorithm.start_step
        # end = swad_algorithm.end_step
        # step_str = f" [{start}-{end}]  (N={swad_algorithm.n_averaged})"
        # row = misc.to_row([results[key] for key in results_keys if key in results]) + step_str
        # logger.info(row)
        
        swad_results = {**summaries_swad1, **summaries_swad2, **summaries_swad3, **summaries_swad4, **summaries_swad5, **summaries_swad6}
        step_str = f" [{swad_algorithm1.start_step}-{swad_algorithm1.end_step}]  (N={swad_algorithm1.n_averaged})  ||  [{swad_algorithm2.start_step}-{swad_algorithm2.end_step}]  (N={swad_algorithm2.n_averaged}) ||  [{swad_algorithm3.start_step}-{swad_algorithm3.end_step}]  (N={swad_algorithm3.n_averaged})  ||  [{swad_algorithm4.start_step}-{swad_algorithm4.end_step}]  (N={swad_algorithm4.n_averaged}) || [{swad_algorithm5.start_step}-{swad_algorithm5.end_step}]  (N={swad_algorithm5.n_averaged}) || [{swad_algorithm6.start_step}-{swad_algorithm6.end_step}]  (N={swad_algorithm6.n_averaged})"
        row = misc.to_row([swad_results[key] for key in list(swad_results.keys())]) + step_str
        logger.info(row)

        ret["SWAD 1"] = swad_results["test_in_s1"]
        ret["SWAD 1 (inDom)"] = swad_results["comb_val_s1"]
        ret["SWAD 2"] = swad_results["test_in_s2"]
        ret["SWAD 2 (inDom)"] = swad_results["comb_val_s2"]
        ret["SWAD 3"] = swad_results["test_in_s3"]  
        ret["SWAD 3 (inDom)"] = swad_results["comb_val_s3"]
        ret["SWAD 4"] = swad_results["test_in_s4"]  
        ret["SWAD 4 (inDom)"] = swad_results["comb_val_s4"]
        ret["SWAD 5"] = swad_results["test_in_s5"]  
        ret["SWAD 5 (inDom)"] = swad_results["comb_val_s5"]
        ret["SWAD 6"] = swad_results["test_in_s6"]  
        ret["SWAD 6 (inDom)"] = swad_results["comb_val_s6"]

        save_dict = {
            "args": vars(args),
            "model_hparams": dict(hparams),
            "test_envs": test_envs,
            "SWAD_1": swad_algorithm1.network.state_dict(),
            "SWAD_2": swad_algorithm2.network.state_dict(),
            "SWAD_3": swad_algorithm3.network.state_dict(),
            "SWAD_4": swad_algorithm4.network.state_dict(),
            "SWAD_5": swad_algorithm5.network.state_dict(),
            "SWAD_6": swad_algorithm6.network.state_dict(),
        }
        
        if args.algorithm in ['DANN', 'CDANN']:
            inter_state_dict = interpolate_algos(swad_algorithm1.module.featurizer.state_dict(), swad_algorithm2.module.featurizer.state_dict(), swad_algorithm3.module.featurizer.state_dict(), swad_algorithm4.module.featurizer.state_dict(), swad_algorithm5.module.featurizer.state_dict(), swad_algorithm6.module.featurizer.state_dict())
            swad_algorithm1.module.featurizer.load_state_dict(inter_state_dict)
            inter_state_dict2 = interpolate_algos(swad_algorithm1.module.classifier.state_dict(), swad_algorithm2.module.classifier.state_dict(), swad_algorithm3.module.classifier.state_dict(), swad_algorithm4.module.classifier.state_dict(), swad_algorithm5.module.classifier.state_dict(), swad_algorithm6.module.classifier.state_dict())
            swad_algorithm1.module.classifier.load_state_dict(inter_state_dict2)
            inter_state_dict3 = interpolate_algos(swad_algorithm1.module.discriminator.state_dict(), swad_algorithm2.module.discriminator.state_dict(), swad_algorithm3.module.discriminator.state_dict(), swad_algorithm4.module.discriminator.state_dict(), swad_algorithm5.module.discriminator.state_dict(), swad_algorithm6.module.discriminator.state_dict())
            swad_algorithm1.module.discriminator.load_state_dict(inter_state_dict3)
            
        elif args.algorithm in ['SagNet']:
            inter_state_dict = interpolate_algos(swad_algorithm1.module.network_f.state_dict(), swad_algorithm2.module.network_f.state_dict(), swad_algorithm3.module.network_f.state_dict(), swad_algorithm4.module.network_f.state_dict(), swad_algorithm5.module.network_f.state_dict(), swad_algorithm6.module.network_f.state_dict())
            swad_algorithm1.module.network_f.load_state_dict(inter_state_dict)
            inter_state_dict2 = interpolate_algos(swad_algorithm1.module.network_c.state_dict(), swad_algorithm2.module.network_c.state_dict(), swad_algorithm3.module.network_c.state_dict(), swad_algorithm4.module.network_c.state_dict(), swad_algorithm5.module.network_c.state_dict(), swad_algorithm6.module.network_c.state_dict())
            swad_algorithm1.module.network_c.load_state_dict(inter_state_dict2)
            inter_state_dict3 = interpolate_algos(swad_algorithm1.module.network_s.state_dict(), swad_algorithm2.module.network_s.state_dict(), swad_algorithm3.module.network_s.state_dict(), swad_algorithm4.module.network_s.state_dict(), swad_algorithm5.module.network_s.state_dict(), swad_algorithm6.module.network_s.state_dict())
            swad_algorithm1.module.network_s.load_state_dict(inter_state_dict3)
        else:
            inter_state_dict = interpolate_algos(swad_algorithm1.network.state_dict(), swad_algorithm2.network.state_dict(), swad_algorithm3.network.state_dict(), swad_algorithm4.network.state_dict(), swad_algorithm5.network.state_dict(), swad_algorithm6.network.state_dict())
            swad_algorithm1.network.load_state_dict(inter_state_dict)

        logger.info(f"Evaluating interpolated model of SWAD models")
        summaries_swadinter = evaluator.evaluate(swad_algorithm1, suffix='_from_swadinter')
        swadinter_results = {**summaries_swadinter}
        logger.info(misc.to_row([swadinter_results[key] for key in list(swadinter_results.keys())]))
        ret["SWAD INTER"] = swadinter_results["test_in_from_swadinter"]
        ret["SWAD INTER (inDom)"] = swadinter_results["comb_val_from_swadinter"]
        save_dict["SWAD_INTER"] = inter_state_dict
        
    
    ckpt_dir = args.out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    test_env_str = ",".join(map(str, test_envs))
    filename = f"TE{test_env_str}.pth"
    if len(test_envs) > 1 and target_env is not None:
        train_env_str = ",".join(map(str, train_envs))
        filename = f"TE{target_env}_TR{train_env_str}.pth"
    path = ckpt_dir / filename
    if not args.debug:
        torch.save(save_dict, path)
    
    
    for k, acc in ret.items():
        logger.info(f"{k} = {acc:.3%}")

    return ret, records

