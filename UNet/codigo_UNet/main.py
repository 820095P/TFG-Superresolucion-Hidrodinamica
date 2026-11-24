import os
import argparse
import torch
import torch.distributed as dist
from mpi4py import MPI
from unet_model import *
from data_preparation import *
from data_postprocessing import *


def main(global_rank, local_rank, world_size, args):
    try:
        # Each rank loads its own config to avoid broadcasting issues
        config = load_info_model(args.config)
        cfg = ModelConfig.from_dict(config)
        evaluation_mask = load_evaluation_domain_mask(
            cfg.EVALUATION_DOMAIN_DEM_PATH,
            cfg.Z_FINE_PATH,
            cfg.EVALUATION_DOMAIN_NODATA,
        )
        if evaluation_mask is None:
            if global_rank == 0:
                print("Dominio de evaluación: rectangular")
        elif global_rank == 0:
            effective_mask_path = (
                cfg.EVALUATION_DOMAIN_DEM_PATH
                if cfg.EVALUATION_DOMAIN_DEM_PATH is not None
                else cfg.Z_FINE_PATH
            )
            print(
                f"Dominio de evaluación: máscara geográfica | "
                f"ruta DEM usada: {effective_mask_path}"
            )
        cellsize_ratio_train = (
            cfg.CELLSIZE_RATIO_TRAIN
            if cfg.CELLSIZE_RATIO_TRAIN is not None
            else cfg.CELLSIZE_RATIO
        )
        cellsize_ratio_real_case = (
            cfg.CELLSIZE_RATIO_REAL_CASE
            if cfg.CELLSIZE_RATIO_REAL_CASE is not None
            else cfg.CELLSIZE_RATIO
        )

        if global_rank == 0:
            print("\nCONFIGURACIÓN DEL MODELO:\n" + "-" * 40 + "\n" + "\n".join(f"{k}: {v}" for k, v in config.items()) + "\n" + "-" * 40)
            print(f"Effective CELLSIZE_RATIO train/test: {cellsize_ratio_train}")
            print(f"Effective CELLSIZE_RATIO real case: {cellsize_ratio_real_case}")

        if cfg.REPRODUCIBILITY:
            set_seed(cfg.SEED)

        device = torch.device(f"cuda:{local_rank}")
        print(f"Device for rank {global_rank}: {device}")

        # We only load the data paths, not the data itself (to avoid overloading the gpus). Only rank 0 does this and then broadcasts the splits to other ranks
        if global_rank == 0:
            splits, paths = load_and_split_all_data_paths(
                cfg.H_COARSE_DATA_PATH,
                cfg.H_FINE_DATA_PATH,
                cfg.TOPO_DATA_PATHS,
                cfg.Z_FINE_PATH,
                random_state=cfg.SEED
                )
        else:
            splits, paths = None, None

        splits = comm.bcast(splits, root=0)
        paths = comm.bcast(paths, root=0)

        # Each rank loads its own part of the data
        X_train, y_train, z_fine_train = load_split(splits['train'], paths, world_size, global_rank, cfg.PATCH_BASED_MAPPING)
        X_val, y_val, z_fine_val = load_split(splits['val'], paths, world_size, global_rank, cfg.PATCH_BASED_MAPPING)
        X_test, y_test, z_fine_test = load_split(splits['test'], paths, world_size, global_rank, cfg.PATCH_BASED_MAPPING)

        print_memory_usage("After creating per-rank datasets", global_rank, device)

        # Each rank normalizes its own data
        print(f"Rank {global_rank}: Normalizing data...")
        saved_norm_params = None
        if not cfg.DO_TRAINING:
            if global_rank == 0:
                print(
                    f"Loading saved normalization parameters for evaluation: "
                    f"{cfg.NORM_PARAMS_SAVE_PATH}"
                )
            saved_norm_params = torch.load(
                cfg.NORM_PARAMS_SAVE_PATH,
                map_location='cpu',
            )
        (X_train_coarse_norm, X_train_topo_norm, y_train_norm, z_fine_train_norm,
            X_val_coarse_norm, X_val_topo_norm, y_val_norm, z_fine_val_norm,
            X_test_coarse_norm, X_test_topo_norm, y_test_norm, z_fine_test_norm,
            norm_params) = normalize_all_datasets_minmax(
            X_train[0], X_train[1], y_train, z_fine_train,
            X_val[0], X_val[1], y_val, z_fine_val,
            X_test[0], X_test[1], y_test, z_fine_test,
            log1p_exp1m_transform=cfg.LOG1P_EXP1M_TRANSFORM,
            device=device,
            normalization_parameters=saved_norm_params,
        )

        print_memory_usage("After normalization", global_rank, device)

        if global_rank == 0 and cfg.DO_TRAINING:
            torch.save(norm_params, cfg.NORM_PARAMS_SAVE_PATH) # save for later use in real case
        
        y_min, y_max = norm_params[4], norm_params[5]

        # Data stats (differs for each rank)
        diagnose_all_data((X_train_coarse_norm, X_train_topo_norm), y_train_norm,
                              (X_val_coarse_norm, X_val_topo_norm), y_val_norm,
                              (X_test_coarse_norm, X_test_topo_norm), y_test_norm)
        
        # Move normalized data to device without artificial padding
        X_train_normalized = (
            X_train_coarse_norm.to(device),
            X_train_topo_norm.to(device)
        )
        y_train_norm = y_train_norm.to(device)
        z_fine_train_norm = z_fine_train_norm.to(device)

        X_val_normalized = (
            X_val_coarse_norm.to(device),
            X_val_topo_norm.to(device)
        )
        y_val_norm = y_val_norm.to(device)
        z_fine_val_norm = z_fine_val_norm.to(device)

        X_test_normalized = (
            X_test_coarse_norm.to(device),
            X_test_topo_norm.to(device)
        )
        y_test_norm = y_test_norm.to(device)
        z_fine_test_norm = z_fine_test_norm.to(device)

        torch.cuda.synchronize()

        print_memory_usage("After padding and moving to device", global_rank, device)
                
        # Building model, infering shapes from data
        if global_rank == 0:
            print("\nBuilding model...")

        coarse_input_channels = X_train_coarse_norm.shape[3] if X_train_coarse_norm.ndim == 4 else 1
        output_channels = y_train_norm.shape[3] if y_train_norm.ndim == 4 else 1
        fine_grid_topographic_features = X_train_topo_norm.shape[3] if X_train_topo_norm.ndim == 4 else 1
        
        if cfg.FIRST_LAYER == 'SG':
            if global_rank == 0:
                print("Using SubGrid interpolation of coarse input as first layer input.")
            model = UNetWithTopographicFeatures3_SG(
                cfg.LAST_ACTIVATION_FUNCTION,
                coarse_input_channels=coarse_input_channels,
                output_channels=output_channels,
                fine_grid_topographic_features=fine_grid_topographic_features,
                group_size_rows=cfg.GROUPING_PARAMETER_FOR_SUBGRID_ROWS,
                group_size_cols=cfg.GROUPING_PARAMETER_FOR_SUBGRID_COLS
            ).to(device)
        elif cfg.FIRST_LAYER == 'BI':
            if global_rank == 0:
                print("Using Bicubic Interpolation of coarse input as first layer input.")
            model = UNetWithTopographicFeatures3(
                cfg.LAST_ACTIVATION_FUNCTION,
                coarse_input_channels=coarse_input_channels,
                output_channels=output_channels,
                fine_grid_topographic_features=fine_grid_topographic_features
            ).to(device)
        else:
            raise ValueError(f"Unknown FIRST_LAYER option: {cfg.FIRST_LAYER}")

        # Wrap model with DistributedDataParallel
        if dist.is_initialized():
            model = torch.nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=False
            )

        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.INITIAL_LEARNING_RATE)
        if cfg.WHICH_LOSS == 'MAESoftCSILoss':
            criterion = MAESoftCSILoss(alpha_softcsi=cfg.ALPHA_CSI, threshold=cfg.THRESHOLD, smooth_factor=1e-4)
        else:
            raise ValueError(f"Unknown loss function: {cfg.WHICH_LOSS}")

        if cfg.HOTSTART and os.path.exists(cfg.MODEL_SAVE_PATH):
            if global_rank == 0:
                print(f"Loading model from checkpoint (HOTSTART): {cfg.MODEL_SAVE_PATH}")
            checkpoint = torch.load(cfg.MODEL_SAVE_PATH, map_location=device)
            raw_model = model.module if hasattr(model, "module") else model # get the underlying model if wrapped in DDP (multi-GPU)
            raw_model.load_state_dict(checkpoint)

        if global_rank == 0:
            print(f"Model created and loaded successfully on {world_size} GPUs")
            print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train Model
        if cfg.DO_TRAINING == True:
            history = train_model(
                model, optimizer, criterion,
                X_train_normalized, y_train_norm,
                X_val_normalized, y_val_norm,
                z_fine_train_norm, z_fine_val_norm,
                total_pixels_to_crop_on_each_dim=cfg.TOTAL_PIXELS_TO_CROP_ON_EACH_DIM,
                error_threshold=cfg.THRESHOLD,
                clip_value_min=cfg.CLIP_VALUE_MIN,
                clip_value_max=cfg.CLIP_VALUE_MAX,
                log1p_exp1m_transform=cfg.LOG1P_EXP1M_TRANSFORM,
                epochs=cfg.EPOCHS,
                batch_size=cfg.BATCH_SIZE,
                early_stopping_patience=cfg.EARLY_STOPPING_PATIENCE,
                model_save_path=cfg.MODEL_SAVE_PATH,
                device=device,
                y_min=y_min,
                y_max=y_max,
                MASK_FINE_MAP_TO_THRESHOLD=cfg.MASK_FINE_MAP_TO_THRESHOLD,
                world_size=world_size,
                rank=global_rank,
                SEED=cfg.SEED
            )

        # Load best model
        if global_rank == 0:
            print("\nLoading best model for evaluation...")
            
        checkpoint = torch.load(cfg.MODEL_SAVE_PATH, map_location=device)
        model.module.load_state_dict(checkpoint)

        # Evaluate model
        model_metrics_tuple, subgrid_metrics_tuple, bicubic_metrics_tuple = evaluate_model(
            model, criterion,
            X_test_normalized, y_test_norm, z_fine_test_norm,
            log1p_exp1m_transform=cfg.LOG1P_EXP1M_TRANSFORM,
            normalization_parameters=norm_params,
            error_threshold=cfg.THRESHOLD,
            device=device,
            clip_value_min=cfg.CLIP_VALUE_MIN,
            clip_value_max=cfg.CLIP_VALUE_MAX,
            grouping_parameter_for_subgrid_rows=cfg.GROUPING_PARAMETER_FOR_SUBGRID_ROWS,
            grouping_parameter_for_subgrid_cols=cfg.GROUPING_PARAMETER_FOR_SUBGRID_COLS,
            save_txt_with_metrics=False,
            save_predictions=cfg.SAVE_PREDICTIONS,
            predictions_save_path=cfg.VISUALIZATION_SAVE_PATH,
            PATCH_BASED_MAPPING=cfg.PATCH_BASED_MAPPING,
            NUMBER_OF_SUBDOMAINS=cfg.NUMBER_OF_SUBDOMAINS,
            evaluation_mask=evaluation_mask,
            evaluation_mask_path=(
                cfg.EVALUATION_DOMAIN_DEM_PATH
                if cfg.EVALUATION_DOMAIN_DEM_PATH is not None
                else cfg.Z_FINE_PATH
            )
        )

        metrics = {
                "UNet": torch.tensor(model_metrics_tuple, device=device),
                "Subgrid": torch.tensor(subgrid_metrics_tuple, device=device),
                "Bicubic": torch.tensor(bicubic_metrics_tuple, device=device),
        }
        if dist.is_initialized():
            for t in metrics.values():
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        # Metrics processing and visualization and evaluation of real case (only on rank 0)
        if global_rank == 0:
            n_batches = len(y_test_norm) / (32 * world_size) if len(y_test_norm) > 0 else 1.0
            summary = {
                "UNet": compute_metrics(metrics["UNet"], has_loss=True,  n_batches=n_batches, threshold=cfg.THRESHOLD),
                "Subgrid":    compute_metrics(metrics["Subgrid"], has_loss=False, n_batches=n_batches, threshold=cfg.THRESHOLD),
                "Bicubic":    compute_metrics(metrics["Bicubic"], has_loss=False, n_batches=n_batches, threshold=cfg.THRESHOLD),
            }
            print_comparison_table(summary, cfg.THRESHOLD, title="Global Test Results")

            if cfg.DO_TRAINING == True:
                plot_training_history(history, save_path=cfg.HISTORY_SAVE_PATH)
                print(f"Training history plotted and saved in {cfg.HISTORY_SAVE_PATH}\n")
            else:
                print("Training was not performed, so no training history to plot.\n")
            
            # Visualize some predictions
            if cfg.VISUALIZE_PREDICTIONS:
                predictions_np = visualize_predictions(
                    model, X_test_normalized, y_test_norm, z_fine_test_norm,
                    save_path=cfg.VISUALIZATION_SAVE_PATH,
                    total_pixels_to_crop_on_each_dim=cfg.TOTAL_PIXELS_TO_CROP_ON_EACH_DIM,
                    cell_size_ratio=cellsize_ratio_train,
                    log1p_exp1m_transform=cfg.LOG1P_EXP1M_TRANSFORM,
                    normalization_parameters=norm_params,
                    n_samples=5,
                    device=device,
                    seed=cfg.SEED,
                    clip_value_min=cfg.CLIP_VALUE_MIN,
                    clip_value_max=cfg.CLIP_VALUE_MAX,
                    error_threshold=cfg.THRESHOLD,
                    grouping_parameter_for_subgrid_rows=cfg.GROUPING_PARAMETER_FOR_SUBGRID_ROWS,
                    grouping_parameter_for_subgrid_cols=cfg.GROUPING_PARAMETER_FOR_SUBGRID_COLS,
                    specific_indices=[0, 10, 20, 60, 80, 89],
                    PATCH_BASED_MAPPING=cfg.PATCH_BASED_MAPPING,
                    NUMBER_OF_SUBDOMAINS=cfg.NUMBER_OF_SUBDOMAINS,
                    NUMBER_OF_COLUMNS_IN_SUBDOMAINS=cfg.NUMBER_OF_COLUMNS_IN_SUBDOMAINS,
                    evaluation_mask=evaluation_mask,
                    evaluation_mask_path=(
                        cfg.EVALUATION_DOMAIN_DEM_PATH
                        if cfg.EVALUATION_DOMAIN_DEM_PATH is not None
                        else cfg.Z_FINE_PATH
                    )
                )

            if cfg.PREDICTIONS_OF_REAL_CASE:
                print("\nMaking predictions for real case...\n")
                X_real_case, y_real_case, z_fine_real_case = prepare_real_case_inputs( #! FALTARÍA ESTUDIAR TEMA DEL PADDING
                    cfg.H_COARSE_REAL_CASE_DATA_PATH, cfg.Z_FINE_PATH, cfg.TOPO_DATA_PATHS, norm_params, device,
                    H_FINE_REAL_CASE_DATA_PATH=cfg.H_FINE_REAL_CASE_DATA_PATH,
                    LOG1P_EXP1M_TRANSFORM=cfg.LOG1P_EXP1M_TRANSFORM
                )

                if y_real_case is not None:
                    if global_rank == 0:
                        print(
                            f"Instantes evaluados en el caso real: "
                            f"{len(y_real_case)}"
                        )
                    # Evaluate model on real case
                    model_metrics_tuple, subgrid_metrics_tuple, bicubic_metrics_tuple = evaluate_model(
                        model, criterion, X_real_case, y_real_case, z_fine_real_case,
                        log1p_exp1m_transform=cfg.LOG1P_EXP1M_TRANSFORM,
                        normalization_parameters=norm_params,
                        error_threshold=cfg.THRESHOLD,
                        device=device,
                        clip_value_min=cfg.CLIP_VALUE_MIN,
                        clip_value_max=cfg.CLIP_VALUE_MAX,
                        grouping_parameter_for_subgrid_rows=cfg.GROUPING_PARAMETER_FOR_SUBGRID_ROWS,
                        grouping_parameter_for_subgrid_cols=cfg.GROUPING_PARAMETER_FOR_SUBGRID_COLS,
                        save_txt_with_metrics=True,
                        save_predictions=cfg.SAVE_PREDICTIONS,
                        predictions_save_path=cfg.VISUALIZATION_SAVE_PATH + '_RealCase',
                        PATCH_BASED_MAPPING=cfg.PATCH_BASED_MAPPING,
                        NUMBER_OF_SUBDOMAINS=cfg.NUMBER_OF_SUBDOMAINS,
                        measure_unet_cost=True,
                        evaluation_mask=evaluation_mask,
                        evaluation_mask_path=(
                            cfg.EVALUATION_DOMAIN_DEM_PATH
                            if cfg.EVALUATION_DOMAIN_DEM_PATH is not None
                            else cfg.Z_FINE_PATH
                        )
                    )

                    metrics = {
                            "UNet": torch.tensor(model_metrics_tuple, device=device),
                            "Subgrid": torch.tensor(subgrid_metrics_tuple, device=device),
                            "Bicubic": torch.tensor(bicubic_metrics_tuple, device=device),
                    }
                    if dist.is_initialized():
                        for t in metrics.values():
                            dist.all_reduce(t, op=dist.ReduceOp.SUM)

                    # Metrics processing and visualization for real case
                    n_batches = len(y_real_case) / (32 * world_size) if len(y_real_case) > 0 else 1.0
                    summary = {
                        "UNet": compute_metrics(metrics["UNet"], has_loss=True,  n_batches=n_batches, threshold=cfg.THRESHOLD),
                        "Subgrid":    compute_metrics(metrics["Subgrid"], has_loss=False, n_batches=n_batches, threshold=cfg.THRESHOLD),
                        "Bicubic":    compute_metrics(metrics["Bicubic"], has_loss=False, n_batches=n_batches, threshold=cfg.THRESHOLD),
                    }
                    print_comparison_table(summary, cfg.THRESHOLD, title="Real Case Test Results")

                    # Visualize predictions for real case
                    if cfg.VISUALIZE_PREDICTIONS:
                        predictions_real_case_np = visualize_predictions(
                            model, X_real_case, y_real_case, z_fine_real_case,
                            save_path=cfg.VISUALIZATION_SAVE_PATH + '_RealCase',
                            total_pixels_to_crop_on_each_dim=cfg.TOTAL_PIXELS_TO_CROP_ON_EACH_DIM,
                            cell_size_ratio=cellsize_ratio_real_case,
                            log1p_exp1m_transform=cfg.LOG1P_EXP1M_TRANSFORM,
                            normalization_parameters=norm_params,
                            n_samples=6,
                            device=device,
                            seed=cfg.SEED,
                            clip_value_min=cfg.CLIP_VALUE_MIN,
                            clip_value_max=cfg.CLIP_VALUE_MAX,
                            error_threshold=cfg.THRESHOLD,
                            grouping_parameter_for_subgrid_rows=cfg.GROUPING_PARAMETER_FOR_SUBGRID_ROWS,
                            grouping_parameter_for_subgrid_cols=cfg.GROUPING_PARAMETER_FOR_SUBGRID_COLS,
                            specific_indices=[0, 10, 20, 60, 80, 89],  # Indices of specific time steps to visualize
                            PATCH_BASED_MAPPING=cfg.PATCH_BASED_MAPPING,
                            NUMBER_OF_SUBDOMAINS=cfg.NUMBER_OF_SUBDOMAINS,
                            NUMBER_OF_COLUMNS_IN_SUBDOMAINS=cfg.NUMBER_OF_COLUMNS_IN_SUBDOMAINS,
                            evaluation_mask=evaluation_mask,
                            evaluation_mask_path=(
                                cfg.EVALUATION_DOMAIN_DEM_PATH
                                if cfg.EVALUATION_DOMAIN_DEM_PATH is not None
                                else cfg.Z_FINE_PATH
                            )
                        )

                    print(f"Real case predictions saved to {cfg.VISUALIZATION_SAVE_PATH + '_RealCase'}...")

                    print("\nGenerating additional visualizations for real case...")
                    generate_real_case_extra_visualizations(
                        model=model,
                        X_real_case=X_real_case,
                        y_real_case=y_real_case,
                        device=device,
                        cfg=cfg,
                        norm_params=norm_params,
                        save_base_path=cfg.VISUALIZATION_SAVE_PATH + "_RealCase",
                        batch_size=16,
                        use_interpolation="subgrid",  # o "bicubic" o None
                        evaluation_mask=evaluation_mask,
                    )
                    print("\nScatter plot, error histogram, confusion matrix and wet area vs time plot generated.\n")

    except Exception as e:
        print(f"Rank (global) {global_rank}: Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group() # cleanup


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run U-Net model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to the configuration .txt file')
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="29500")
    args = parser.parse_args()
    
    num_gpus_per_node = torch.cuda.device_count() # in case there are multiple GPUs per node

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    global_rank = rank = comm.Get_rank()
    local_rank = global_rank % max(1, num_gpus_per_node)

    print(f"Global rank: {global_rank}, Local rank: {local_rank}, World size: {world_size}")
    torch.cuda.set_device(local_rank)

    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(global_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['MASTER_ADDR'] = str(args.master_addr)
    os.environ['MASTER_PORT'] = str(args.master_port)
    #! ONLY FOR FRONTIER
    # os.environ['NCCL_SOCKET_IFNAME'] = 'hsn0'
    # os.environ['NCCL_IB_HCA'] = 'mlx5_0,mlx5_1,mlx5_2,mlx5_3'
    # os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ['NCCL_NET_GDR_LEVEL'] = '3'

    # Initialize the process group for distributed training (in case only 1 GPU is available, this will just initialize a single-process group. Not ideal but avoids the need to write separate code for single GPU vs multi GPU)
    dist.init_process_group(backend="nccl", init_method='env://', rank=global_rank, world_size=world_size, device_id=torch.device(f"cuda:{local_rank}"))
    
    # Each process directly calls main
    main(global_rank, local_rank, world_size, args)
