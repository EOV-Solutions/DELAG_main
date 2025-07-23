"""
Enhanced Annual Temperature Cycle (ATC) model using PyTorch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import tempfile  # Added for temporary snapshot storage on disk
from joblib import Parallel, delayed  # Added for parallelization
import config  # Assuming your config.py is in the same directory or accessible
import utils  # Added for the new plotting function


class EnhancedATCModel(nn.Module):
    """
    Enhanced ATC model for a single pixel or a batch of pixels.
    T_ATC(d) = C + A * cos(2*pi/DOY_MAX * (d - phi)) + b * T_ERA5(d)
    Parameters C, A, phi, b are learnable.
    """
    def __init__(self, initial_params: dict = None):
        super().__init__()
        # Parameters: C (mean temp), A (amplitude), phi (phase shift), b (ERA5 coefficient)
        # Initialize with some reasonable defaults or provided values
        # These will be per-pixel, so if processing a batch of pixels, these should be vectors.
        # For a single pixel model instance:
        self.C = nn.Parameter(torch.tensor(initial_params.get('C', 290.0) if initial_params else 290.0)) # Avg temp in Kelvin
        self.A = nn.Parameter(torch.tensor(initial_params.get('A', 10.0) if initial_params else 10.0))   # Amplitude in Kelvin
        self.phi = nn.Parameter(torch.tensor(initial_params.get('phi', 180.0) if initial_params else 180.0)) # Phase shift in days
        self.A_2 = nn.Parameter(torch.tensor(initial_params.get('A_2', 1.0) if initial_params else 1.0))   # Amplitude of second harmonic
        self.phi_2 = nn.Parameter(torch.tensor(initial_params.get('phi_2', 90.0) if initial_params else 90.0)) # Phase shift of second harmonic
        
        self.b = nn.Parameter(torch.tensor(initial_params.get('b', 0.5) if initial_params else 0.5))     # ERA5 coefficient
        
        self.days_in_year = config.DAYS_OF_YEAR # From config file

    def forward(self, doy: torch.Tensor, t_era5_1: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ATC model.

        Args:
            doy (torch.Tensor): Day of year (1 to self.days_in_year).
            t_era5_1 (torch.Tensor): ERA5 band 1 values for the corresponding doy.

        Returns:
            torch.Tensor: Predicted LST (T_ATC).
        """
        term_cos = torch.cos(2 * np.pi / self.days_in_year * (doy - self.phi))
        term_sin = torch.sin(2 * np.pi / self.days_in_year * (doy - self.phi_2))
        t_atc = self.C + self.A * term_cos + self.A_2 * term_sin + self.b * t_era5_1
        # t_atc = self.C + self.A * term_cos
        return t_atc

def train_atc_model_pixelwise(
    pixel_lst_clear: np.ndarray, 
    pixel_doy_clear: np.ndarray, 
    pixel_era5_clear: np.ndarray,
    app_config: 'config',
    pixel_identifier: str = "" # Keep for potential diagnostic messages
) -> tuple[EnhancedATCModel | None, list[dict], dict[str, list[float]]]: # Model can be None, return dict for losses
    """
    Trains the Enhanced ATC model for a single pixel using a two-phase approach.
    Phase 1: Search for optimal initial parameters by running short training trials.
    Phase 2: Train the model fully starting with the best initial parameters found.

    Args:
        pixel_lst_clear (np.ndarray): Clear-sky LST observations for the pixel (1D array).
        pixel_doy_clear (np.ndarray): Corresponding day of year for LST_clear (1D array).
        pixel_era5_clear (np.ndarray): Corresponding ERA5 skin temperature for LST_clear (1D array).
        app_config: Configuration object.
        pixel_identifier (str, optional): Identifier for the pixel for logging.

    Returns:
        tuple[EnhancedATCModel | None, list[dict], dict[str, list[float]]]: 
            - The trained ATC model for the pixel.
            - A list of model state_dict snapshots for ensemble.
            - A dictionary containing lists of mean 'train' and 'val' losses for each logging interval.
    """
    device = torch.device(app_config.ATC_DEVICE if torch.cuda.is_available() else "cpu")
    loss_fn = nn.L1Loss()  # Changed from MSELoss to L1Loss to match paper specification

    # --- Data Split ---
    num_samples = len(pixel_lst_clear)
    # Initialize loss dictionary for return in case of early exit
    loss_logging_interval_setup = getattr(app_config, 'ATC_LOSS_LOGGING_INTERVAL', 100)
    total_epochs_setup = app_config.ATC_EPOCHS
    num_loss_intervals_setup = (total_epochs_setup + loss_logging_interval_setup - 1) // loss_logging_interval_setup
    default_losses_dict = {
        'train': [np.nan] * num_loss_intervals_setup,
        'val': [np.nan] * num_loss_intervals_setup
    }

    if num_samples < 2: # Not enough data to split
        return None, [], default_losses_dict

    indices = np.random.permutation(num_samples)
    split_idx = int(num_samples * 0.9)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    if len(train_indices) == 0 or len(val_indices) == 0:
        # Not enough data for a meaningful train/val split, use all for training.
        train_indices = indices
        val_indices = [] # No validation set

    # Convert all inputs to tensors first
    lst_tensor_full = torch.from_numpy(pixel_lst_clear).float().to(device)
    doy_tensor_full = torch.from_numpy(pixel_doy_clear).float().to(device)
    era5_tensor_full = torch.from_numpy(pixel_era5_clear).float().to(device)  # Single band ERA5

    # Create train and val tensors from indices
    lst_tensor_train = lst_tensor_full[train_indices]
    doy_tensor_train = doy_tensor_full[train_indices]
    era5_tensor_train = era5_tensor_full[train_indices]

    if val_indices.size > 0:
        lst_tensor_val = lst_tensor_full[val_indices]
        doy_tensor_val = doy_tensor_full[val_indices]
        era5_tensor_val = era5_tensor_full[val_indices]
    else:
        lst_tensor_val, doy_tensor_val, era5_tensor_val = None, None, None


    # --- Phase 1: Search for best initial parameters (using training data only) ---
    init_search_trials = getattr(app_config, 'ATC_INIT_SEARCH_TRIALS', 20)
    init_search_epochs = getattr(app_config, 'ATC_INIT_SEARCH_EPOCHS', 40)
    
    best_initial_params = None
    best_loss = float('inf')

    # Base values for randomization from training data
    initial_C_base = np.nanmean(pixel_lst_clear[train_indices]) if len(train_indices) > 0 else 290.0
    initial_A_base = np.nanstd(pixel_lst_clear[train_indices]) if len(train_indices) > 1 else 10.0
    initial_phi_base = 0
    initial_b_base = 0.1
    
    initial_C_base_val = float(initial_C_base) if np.isfinite(initial_C_base) else 290.0
    initial_A_base_val = float(initial_A_base) if np.isfinite(initial_A_base) and initial_A_base > 0 else 10.0

    for _ in range(init_search_trials):
        # Randomize initial parameters as requested: base * random_float(0-1)
        trial_initial_params = {
            'C': initial_C_base_val * (0.5 + np.random.rand()),
            'A': initial_A_base_val * np.random.rand(),
            'phi': initial_phi_base * (0.5 + np.random.rand()),
            'A_2': (initial_A_base_val / 5) * np.random.rand(), # A_2 is typically smaller
            'phi_2': initial_phi_base * np.random.rand(),
            'b': initial_b_base * (0.5 + np.random.rand()),
        }
        
        # Short training for this trial
        atc_model_trial = EnhancedATCModel(initial_params=trial_initial_params).to(device)
        optimizer_trial = optim.Adam(atc_model_trial.parameters(), lr=app_config.ATC_LEARNING_RATE)
        trial_final_loss = float('inf')

        for _ in range(init_search_epochs):
            atc_model_trial.train()
            optimizer_trial.zero_grad()
            
            predictions = atc_model_trial(doy_tensor_train, era5_tensor_train)
            loss = loss_fn(predictions, lst_tensor_train)
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                trial_final_loss = float('inf')
                break # This trial with these initial params failed

            loss.backward()
            optimizer_trial.step()
            trial_final_loss = loss.item()
        
        if not np.isinf(trial_final_loss) and trial_final_loss < best_loss:
            best_loss = trial_final_loss
            best_initial_params = trial_initial_params

    # Fallback to deterministic initialization if search fails to find any valid params
    if best_initial_params is None:
        best_initial_params = {
            'C': initial_C_base_val,
            'A': initial_A_base_val,
            'phi': initial_phi_base,
            'A_2': initial_A_base_val / 5, # A_2 is typically smaller
            'phi_2': initial_phi_base / 2,
            'b': initial_b_base,
        }

    # --- Phase 2: Full training with best initial parameters ---
    atc_model = EnhancedATCModel(initial_params=best_initial_params).to(device)
    optimizer = optim.Adam(atc_model.parameters(), 
                           lr=app_config.ATC_LEARNING_RATE, 
                           weight_decay=getattr(app_config, 'ATC_WEIGHT_DECAY', 0.0))
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=getattr(app_config, 'ATC_LR_SCHEDULER_FACTOR', 0.1),
        patience=getattr(app_config, 'ATC_LR_SCHEDULER_PATIENCE', 10),
        min_lr=getattr(app_config, 'ATC_LR_SCHEDULER_MIN_LR', 1e-6)
    )

    snapshots = []
    
    # Loss logging setup
    loss_logging_interval = getattr(app_config, 'ATC_LOSS_LOGGING_INTERVAL', 100)
    total_epochs = app_config.ATC_EPOCHS
    num_loss_intervals = (total_epochs + loss_logging_interval - 1) // loss_logging_interval
    interval_losses_output = {
        'train': [np.nan] * num_loss_intervals,
        'val': [np.nan] * num_loss_intervals
    }
    
    current_interval_train_losses = []

    phase2_epochs = total_epochs - init_search_epochs
    if phase2_epochs < 0:
        phase2_epochs = 0
        
    last_global_epoch = -1 # To handle the case where phase2_epochs is 0

    for epoch in range(phase2_epochs):
        global_epoch = epoch + init_search_epochs
        last_global_epoch = global_epoch

        atc_model.train()
        optimizer.zero_grad()
        
        predictions = atc_model(doy_tensor_train, era5_tensor_train)
        
        loss = loss_fn(predictions, lst_tensor_train)
        
        loss.backward()
        if torch.isnan(loss).any() or torch.isinf(loss).any(): # Added check for inf loss
            # On failure in phase 2, return current losses (mostly NaNs).
            return None, [], interval_losses_output

        optimizer.step()
        
        current_interval_train_losses.append(loss.item())

        # Step the scheduler based on training loss
        scheduler.step(loss)

        # Log loss at interval
        if (global_epoch + 1) % loss_logging_interval == 0:
            current_interval_idx = global_epoch // loss_logging_interval
            if current_interval_train_losses and current_interval_idx < num_loss_intervals:
                interval_losses_output['train'][current_interval_idx] = np.mean(current_interval_train_losses)
            current_interval_train_losses = [] # Reset for next interval

            # Calculate validation loss for the interval
            if lst_tensor_val is not None:
                atc_model.eval() # Switch to evaluation mode
                with torch.no_grad():
                    val_predictions = atc_model(doy_tensor_val, era5_tensor_val)
                    val_loss = loss_fn(val_predictions, lst_tensor_val)
                    if current_interval_idx < num_loss_intervals:
                        interval_losses_output['val'][current_interval_idx] = val_loss.item()
                atc_model.train() # Switch back to training mode

        # Store snapshots for ensemble
        if (global_epoch >= app_config.ATC_ENSEMBLE_START_EPOCH and 
            (global_epoch - app_config.ATC_ENSEMBLE_START_EPOCH) % app_config.ATC_SNAPSHOT_INTERVAL == 0):
            snapshots.append({k: v.clone().cpu().detach() for k, v in atc_model.state_dict().items()})
            if len(snapshots) >= app_config.ATC_ENSEMBLE_SNAPSHOTS:
                # If we break early, handle the last partial loss interval
                if current_interval_train_losses:
                    current_interval_idx = global_epoch // loss_logging_interval
                    if current_interval_idx < num_loss_intervals:
                        # Log train loss
                        interval_losses_output['train'][current_interval_idx] = np.mean(current_interval_train_losses)
                        # Log val loss one last time if possible
                        if lst_tensor_val is not None:
                            atc_model.eval()
                            with torch.no_grad():
                                val_predictions = atc_model(doy_tensor_val, era5_tensor_val)
                                val_loss = loss_fn(val_predictions, lst_tensor_val)
                                if current_interval_idx < num_loss_intervals:
                                    interval_losses_output['val'][current_interval_idx] = val_loss.item()
                            atc_model.train()
                break 
    
    # Handle final partial interval if training finished all epochs
    if current_interval_train_losses and last_global_epoch != -1:
        current_interval_idx = last_global_epoch // loss_logging_interval
        if current_interval_idx < num_loss_intervals and np.isnan(interval_losses_output['train'][current_interval_idx]):
             interval_losses_output['train'][current_interval_idx] = np.mean(current_interval_train_losses)
             # Note: Validation loss is only calculated at interval boundaries, so no need to calculate it here.

    return atc_model, snapshots, interval_losses_output

# MODIFIED for returning structured interval losses
def _train_pixel_atc_worker(
    r: int,
    c: int,
    pixel_lst_all_times_slice: np.ndarray,
    pixel_era5_all_times_slice: np.ndarray,
    doy_stack_all_days_numpy: np.ndarray,
    app_config: "config",
    num_times_for_output: int,
    temp_snapshot_dir: str,
) -> tuple[int, int, str, dict[str, list[float]]]:
    """
    Worker function to train ATC for a single pixel.

    To reduce RAM usage, the trained snapshot parameters are written to a temporary
    ``.npz`` file on disk (one per pixel). The path to this file is returned so
    the parent process can later merge all snapshots into the final stacks
    without ever holding the full collection in memory.
    Returns:
        tuple: (row, col, path_to_snapshot_file, interval_loss_dict)
    """
    worker_device_str = app_config.ATC_DEVICE
    if app_config.ATC_DEVICE.lower() == "cuda" and getattr(app_config, 'ATC_N_JOBS', -1) != 1:
        worker_device_str = "cpu" 
    
    device = torch.device(worker_device_str if torch.cuda.is_available() and worker_device_str.lower() == "cuda" else "cpu")

    pixel_id_str = f"Pixel ({r},{c})"
    
    # Initialize default interval losses (all NaNs) as a dictionary
    loss_logging_interval = getattr(app_config, 'ATC_LOSS_LOGGING_INTERVAL', 100)
    num_loss_intervals = (app_config.ATC_EPOCHS + loss_logging_interval - 1) // loss_logging_interval
    default_interval_losses = {
        'train': [np.nan] * num_loss_intervals,
        'val': [np.nan] * num_loss_intervals
    }

    if np.isnan(pixel_era5_all_times_slice).all():
        return r, c, "", default_interval_losses # Return empty snapshots and default losses

    clear_sky_indices = np.where(~np.isnan(pixel_lst_all_times_slice))[0]
    pixel_lst_clear = pixel_lst_all_times_slice[clear_sky_indices]
    pixel_doy_clear = doy_stack_all_days_numpy[clear_sky_indices]
    pixel_era5_clear = pixel_era5_all_times_slice[clear_sky_indices]
    
    mask_lst = ~np.isnan(pixel_lst_clear)
    mask_era5 = ~np.isnan(pixel_era5_clear)  # Only one ERA5 band now
    valid_data_mask = mask_lst & mask_era5
    pixel_lst_clear_valid = pixel_lst_clear[valid_data_mask]
    pixel_doy_clear_valid = pixel_doy_clear[valid_data_mask]
    pixel_era5_clear_valid = pixel_era5_clear[valid_data_mask]

    model_snapshots: list[dict] = []
    pixel_interval_losses_final = {k: v[:] for k, v in default_interval_losses.items()} # Deep copy

    if len(pixel_lst_clear_valid) >= app_config.MIN_CLEAR_OBS_ATC:
        if not (np.isnan(pixel_lst_clear_valid).any() or \
                np.isnan(pixel_doy_clear_valid).any() or \
                np.isnan(pixel_era5_clear_valid).any()):
            
            trained_model_pixel, model_snaps_from_train, interval_losses_from_train = train_atc_model_pixelwise(
                pixel_lst_clear_valid, pixel_doy_clear_valid, pixel_era5_clear_valid, app_config,
                pixel_identifier=pixel_id_str
            )
            if trained_model_pixel and model_snaps_from_train:
                model_snapshots = model_snaps_from_train
            # Use interval_losses_from_train regardless of whether model was trained
            pixel_interval_losses_final = interval_losses_from_train 
    
    # ------------------------------
    # Persist snapshots to disk to limit RAM usage
    # ------------------------------
    num_snaps_expected = app_config.ATC_ENSEMBLE_SNAPSHOTS
    C_arr   = np.full(num_snaps_expected, np.nan, dtype=np.float32)
    A_arr   = np.full(num_snaps_expected, np.nan, dtype=np.float32)
    phi_arr = np.full(num_snaps_expected, np.nan, dtype=np.float32)
    A2_arr  = np.full(num_snaps_expected, np.nan, dtype=np.float32)
    phi2_arr= np.full(num_snaps_expected, np.nan, dtype=np.float32)
    b_arr   = np.full(num_snaps_expected, np.nan, dtype=np.float32)

    for idx, state_dict in enumerate(model_snapshots):
        if idx >= num_snaps_expected:
            break
        # Extract floats regardless of tensor/float type
        C_arr[idx]    = float(state_dict.get("C", np.nan)) if not torch.is_tensor(state_dict.get("C", None)) else state_dict["C"].item()
        A_arr[idx]    = float(state_dict.get("A", np.nan)) if not torch.is_tensor(state_dict.get("A", None)) else state_dict["A"].item()
        phi_arr[idx]  = float(state_dict.get("phi", np.nan)) if not torch.is_tensor(state_dict.get("phi", None)) else state_dict["phi"].item()
        A2_arr[idx]   = float(state_dict.get("A_2", np.nan)) if not torch.is_tensor(state_dict.get("A_2", None)) else state_dict["A_2"].item()
        phi2_arr[idx] = float(state_dict.get("phi_2", np.nan)) if not torch.is_tensor(state_dict.get("phi_2", None)) else state_dict["phi_2"].item()
        b_arr[idx]    = float(state_dict.get("b", np.nan)) if not torch.is_tensor(state_dict.get("b", None)) else state_dict["b"].item()

    pixel_snapshot_path = os.path.join(temp_snapshot_dir, f"pixel_{r}_{c}.npz")
    try:
        np.savez_compressed(
            pixel_snapshot_path,
            C=C_arr,
            A=A_arr,
            phi=phi_arr,
            A_2=A2_arr,
            phi_2=phi2_arr,
            b=b_arr,
        )
    except Exception as e:
        print(f"Error saving snapshot for pixel ({r},{c}) to {pixel_snapshot_path}: {e}")
        # Fallback: still return empty path so downstream knows something went wrong
        pixel_snapshot_path = ""

    return r, c, pixel_snapshot_path, pixel_interval_losses_final

# MODIFIED to collect and return dict of loss maps
def train_and_collect_all_atc_snapshots(
    preprocessed_data: dict, app_config: 'config'
) -> tuple[dict[tuple[int, int], str], dict[str, np.ndarray]]:
    """
    Trains ATC models for all pixels, collects snapshots, and interval loss maps.
    Returns:
        tuple[dict[tuple[int, int], list[dict]], dict[str, np.ndarray]]:
            - all_pixel_snapshots: Dict mapping (r,c) to list of state_dict snapshots.
            - interval_loss_maps: Dict with 'train' and 'val' keys, each mapping to a 
                                  NumPy array (num_intervals, height, width) of mean losses.
    """
    lst_stack = preprocessed_data["lst_stack"]
    era5_stack = preprocessed_data["era5_stack"]
    doy_stack_numpy = preprocessed_data["doy_stack"]
    training_pixel_mask = preprocessed_data.get("training_pixel_mask")
    
    num_times_obs, height, width = lst_stack.shape

    # --- START: Plot input data timeseries overview --- ADDED BLOCK ---
    if getattr(app_config, 'PLOT_INPUT_TIMESERIES_OVERVIEW', True): # Check a config flag if you want to make it optional
        print("\nGenerating input data timeseries overview plot...")
        try:
            utils.plot_input_data_timeseries_overview(
                doy_stack_numpy=doy_stack_numpy,
                lst_stack=lst_stack,
                era5_stack=era5_stack, # era5_stack is (time, 2, H, W)
                training_pixel_mask=training_pixel_mask, # Can be None
                output_dir=app_config.OUTPUT_DIR,
                roi_name=preprocessed_data.get('roi_name', 'UnknownROI')
            )
        except Exception as e:
            print(f"Warning: Failed to generate input data timeseries overview plot. Error: {e}")
            import traceback
            traceback.print_exc()
    # --- END: Plot input data timeseries overview ---

    if training_pixel_mask is None:
        training_pixel_mask = np.ones((height, width), dtype=bool)
    elif not isinstance(training_pixel_mask, np.ndarray) or training_pixel_mask.shape != (height, width):
        print(f"Warning: training_pixel_mask has unexpected type/shape ({type(training_pixel_mask)}, {training_pixel_mask.shape if isinstance(training_pixel_mask, np.ndarray) else 'N/A'}). Defaulting to training all pixels.")
        training_pixel_mask = np.ones((height, width), dtype=bool)

    num_pixels_to_train = np.sum(training_pixel_mask)
    # print(f"Preparing arguments for parallel ATC model training (snapshot & loss collection) for {num_pixels_to_train} selected pixels...")

    # Create a temporary directory for per-pixel snapshot files
    temp_snapshot_dir = tempfile.mkdtemp(prefix="atc_pixel_snapshots_")
    print(f"Per-pixel snapshots will be written to temporary directory: {temp_snapshot_dir}")

    tasks_args_list = []
    for r_iter in range(height):
        for c_iter in range(width):
            if training_pixel_mask[r_iter, c_iter]:
                tasks_args_list.append(
                    (r_iter, c_iter,
                     lst_stack[:, r_iter, c_iter].copy(),
                     era5_stack[:, r_iter, c_iter].copy(),  # ERA5 is now 2D: (time, spatial)
                     doy_stack_numpy.copy(),
                     app_config,
                     num_times_obs,
                     temp_snapshot_dir
                    )
                )

    n_jobs = getattr(app_config, 'ATC_N_JOBS', -1)
    # print(f"Starting parallel ATC training (snapshot & loss collection) with n_jobs={n_jobs}...")
    
    delayed_jobs = [delayed(_train_pixel_atc_worker)(*task_args) for task_args in tasks_args_list]
    
    results = Parallel(n_jobs=n_jobs, verbose=0, backend='loky')(
        tqdm(delayed_jobs, desc="Training ATC & Collecting Losses/Snapshots", total=len(delayed_jobs))
    )
    # print(f"Parallel ATC snapshot & loss collection finished. Processed {len(results)} pixel tasks.")

    all_pixel_snapshots: dict[tuple[int, int], str] = {}
    
    # Prepare structure for interval loss maps (train and val)
    loss_logging_interval = getattr(app_config, 'ATC_LOSS_LOGGING_INTERVAL', 100)
    num_loss_intervals = (app_config.ATC_EPOCHS + loss_logging_interval -1) // loss_logging_interval
    interval_loss_maps = {
        'train': np.full((num_loss_intervals, height, width), np.nan, dtype=np.float32),
        'val': np.full((num_loss_intervals, height, width), np.nan, dtype=np.float32)
    }

    print("Collecting snapshots and interval losses from parallel ATC training...")
    for r_res, c_res, snapshot_path_for_pixel, interval_losses_dict in tqdm(results, desc="Organizing Results"):
        all_pixel_snapshots[(r_res, c_res)] = snapshot_path_for_pixel
        
        # Unpack the dictionary of losses and populate the respective maps
        train_losses = interval_losses_dict.get('train', [])
        val_losses = interval_losses_dict.get('val', [])

        if len(train_losses) == num_loss_intervals:
            for interval_idx in range(num_loss_intervals):
                interval_loss_maps['train'][interval_idx, r_res, c_res] = train_losses[interval_idx]
        
        if len(val_losses) == num_loss_intervals:
            for interval_idx in range(num_loss_intervals):
                interval_loss_maps['val'][interval_idx, r_res, c_res] = val_losses[interval_idx]
    
    # For pixels not in training_pixel_mask, their entries in interval_loss_maps will remain NaN.
    
    print("Finished collecting all ATC model snapshots and interval losses.")
    return all_pixel_snapshots, interval_loss_maps

# -----------------------------------------------------------------
#  SAVE SNAPSHOTS – accepts in-memory lists OR on-disk per-pixel files
# -----------------------------------------------------------------
def save_atc_snapshots(
    all_pixel_snapshots: dict,
    filepath: str,
    image_height: int,
    image_width: int,
    num_snapshots_expected: int,
):
    """
    Saves the collected ATC model parameter snapshots to a compressed NPZ file.

    Args:
        all_pixel_snapshots (dict): Dict mapping (r,c) to list of state_dict snapshots.
                                    Each state_dict contains 'C', 'A', 'phi', 'b' as tensors.
        filepath (str): Path to save the .npz file.
        image_height (int): Height of the original image.
        image_width (int): Width of the original image.
        num_snapshots_expected (int): The number of snapshots expected per pixel (e.g., app_config.ATC_ENSEMBLE_SNAPSHOTS).
    """
    # Initialize stacks for each parameter
    # Shape: (num_snapshots, height, width)
    C_stack = np.full((num_snapshots_expected, image_height, image_width), np.nan, dtype=np.float32)
    A_stack = np.full((num_snapshots_expected, image_height, image_width), np.nan, dtype=np.float32)
    phi_stack = np.full((num_snapshots_expected, image_height, image_width), np.nan, dtype=np.float32)
    A_2_stack = np.full((num_snapshots_expected, image_height, image_width), np.nan, dtype=np.float32)
    phi_2_stack = np.full((num_snapshots_expected, image_height, image_width), np.nan, dtype=np.float32)
    b_stack = np.full((num_snapshots_expected, image_height, image_width), np.nan, dtype=np.float32)

    print(f"Structuring snapshots for saving. Expected snapshots per pixel: {num_snapshots_expected}")
    
    for (r, c), snapshot_entry in tqdm(all_pixel_snapshots.items(), desc="Processing pixels for saving"):
        # ------------------------------------------------------------
        # Case 1: snapshot_entry is a path to an on-disk npz file (new).
        # Case 2: snapshot_entry is an in-memory list of state_dicts (legacy).
        # ------------------------------------------------------------

        if isinstance(snapshot_entry, str):
            pixel_file_path = snapshot_entry
            if not pixel_file_path or not os.path.exists(pixel_file_path):
                print(f"Warning: Snapshot file for pixel ({r},{c}) not found. Filling with NaNs.")
                continue

            try:
                with np.load(pixel_file_path) as pdata:
                    C_arr   = pdata.get("C")
                    A_arr   = pdata.get("A")
                    phi_arr = pdata.get("phi")
                    A2_arr  = pdata.get("A_2")
                    phi2_arr = pdata.get("phi_2")
                    b_arr   = pdata.get("b")

                if C_arr is None:
                    print(f"Warning: Snapshot file {pixel_file_path} missing parameters. Skipping.")
                    continue

                # Ensure correct length by padding/truncating
                for idx in range(min(num_snapshots_expected, len(C_arr))):
                    C_stack[idx, r, c]   = C_arr[idx]
                    A_stack[idx, r, c]   = A_arr[idx]
                    phi_stack[idx, r, c] = phi_arr[idx]
                    A_2_stack[idx, r, c] = A2_arr[idx]
                    phi_2_stack[idx, r, c] = phi2_arr[idx]
                    b_stack[idx, r, c]   = b_arr[idx]

            except Exception as e:
                print(f"Error reading snapshot file {pixel_file_path}: {e}")

            continue  # Done with on-disk path entry

        # ------------------------------------------------------------
        # Legacy path – snapshot_entry is list of state_dicts held in RAM
        # ------------------------------------------------------------
        snapshots_list = snapshot_entry
        if not snapshots_list:
            print(f"Warning: No snapshots found for pixel ({r},{c}). Will be NaNs in saved file.")
            continue

        if len(snapshots_list) != num_snapshots_expected:
            print(
                f"Warning: Pixel ({r},{c}) has {len(snapshots_list)} snapshots, expected {num_snapshots_expected}."
            )

        for idx, state_dict in enumerate(snapshots_list):
            if idx >= num_snapshots_expected:
                break
            C_stack[idx, r, c]   = float(state_dict.get("C", np.nan)) if not torch.is_tensor(state_dict.get("C", None)) else state_dict["C"].item()
            A_stack[idx, r, c]   = float(state_dict.get("A", np.nan)) if not torch.is_tensor(state_dict.get("A", None)) else state_dict["A"].item()
            phi_stack[idx, r, c] = float(state_dict.get("phi", np.nan)) if not torch.is_tensor(state_dict.get("phi", None)) else state_dict["phi"].item()
            A_2_stack[idx, r, c] = float(state_dict.get("A_2", np.nan)) if not torch.is_tensor(state_dict.get("A_2", None)) else state_dict["A_2"].item()
            phi_2_stack[idx, r, c] = float(state_dict.get("phi_2", np.nan)) if not torch.is_tensor(state_dict.get("phi_2", None)) else state_dict["phi_2"].item()
            b_stack[idx, r, c]   = float(state_dict.get("b", np.nan)) if not torch.is_tensor(state_dict.get("b", None)) else state_dict["b"].item()

    print(f"Saving snapshot stacks to {filepath}...")
    np.savez_compressed(filepath, 
                        C_snapshots=C_stack, 
                        A_snapshots=A_stack, 
                        phi_snapshots=phi_stack,
                        A_2_snapshots=A_2_stack,
                        phi_2_snapshots=phi_2_stack,
                        b_snapshots=b_stack)
    print(f"Snapshots saved successfully.")

def load_atc_snapshots(filepath: str) -> dict:
    """
    Loads ATC model parameter snapshots from an NPZ file.

    Args:
        filepath (str): Path to the .npz file.

    Returns:
        dict: A dictionary with keys 'C_snapshots', 'A_snapshots', 'phi_snapshots', 'b_snapshots',
              each mapping to a NumPy array of shape (num_snapshots, height, width).
    """
    print(f"Loading snapshots from {filepath}...")
    data = np.load(filepath)
    print("Snapshots loaded.")
    # For now, assume new files won't have 'd_snapshots'. Prediction code needs to handle this.
    # Also, this structure is for ATC model. MLP snapshots would be raw state_dicts.
    loaded_data = {
        "C_snapshots": data.get('C_snapshots'), # Use .get() in case file is from older version or MLP (where this won't be)
        "A_snapshots": data.get('A_snapshots'),
        "phi_snapshots": data.get('phi_snapshots'),
        "A_2_snapshots": data.get('A_2_snapshots'),
        "phi_2_snapshots": data.get('phi_2_snapshots'),
        "b_snapshots": data.get('b_snapshots'),
    }
    # It would be good to save/load ATC_MODEL_TYPE with the snapshots.
    # For now, prediction function assumes ATC or needs modification for MLP.
    return loaded_data

def predict_atc_from_loaded_snapshots(
    loaded_snapshots_data: dict, 
    doy_for_prediction_numpy: np.ndarray, 
    era5_for_prediction_numpy: np.ndarray, # This is the full era5_stack (time, height, width)
    app_config: 'config',
    prediction_mask: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs ATC prediction using loaded snapshots for all pixels.

    Args:
        loaded_snapshots_data (dict): Data loaded by load_atc_snapshots.
        doy_for_prediction_numpy (np.ndarray): DOY for the prediction timeline (1D array, time_pred).
        era5_for_prediction_numpy (np.ndarray): ERA5 data for the prediction timeline (time_pred, height, width).
        app_config: Configuration object.
        prediction_mask (np.ndarray, optional): A boolean mask (height, width) specifying which pixels to predict.
                                                If None, all pixels are predicted.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - atc_predictions_mean (np.ndarray): Mean ATC predictions (time_pred, height, width).
            - atc_predictions_variance (np.ndarray): Ensemble variance of ATC predictions (time_pred, height, width).
    """
    C_snaps = loaded_snapshots_data['C_snapshots'] # (num_snaps, height, width)
    A_snaps = loaded_snapshots_data['A_snapshots']
    phi_snaps = loaded_snapshots_data['phi_snapshots']
    A_2_snaps = loaded_snapshots_data.get('A_2_snapshots') # Use .get for backward compatibility
    phi_2_snaps = loaded_snapshots_data.get('phi_2_snapshots')
    b_snaps = loaded_snapshots_data['b_snapshots']

    num_snapshots, height, width = C_snaps.shape
    num_times_pred = len(doy_for_prediction_numpy)

    print(f"Starting ATC prediction from loaded snapshots. Image: {height}x{width}, Snapshots: {num_snapshots}, Pred Timesteps: {num_times_pred}")

    # Output arrays
    atc_predictions_mean = np.full((num_times_pred, height, width), np.nan, dtype=np.float32)
    atc_predictions_variance = np.full((num_times_pred, height, width), np.nan, dtype=np.float32)

    # Determine device for prediction models (can be global if predictions are sequential per pixel)
    # For parallelizing prediction itself (not done here), device management would be per-worker.
    device = torch.device(app_config.ATC_DEVICE if torch.cuda.is_available() else "cpu")
    
    # Convert DOY for prediction to a tensor once
    doy_for_prediction_tensor = torch.from_numpy(doy_for_prediction_numpy).float().to(device) # (time_pred)

    # Create a single model instance to reuse
    # Initialize with dummy params, will be overwritten by loaded snapshots
    temp_model_initial_params = {'C':0.0,'A':0.0,'phi':0.0,'A_2':0.0,'phi_2':0.0,'b':0.0}
    temp_model = EnhancedATCModel(initial_params=temp_model_initial_params).to(device)
    temp_model.eval() # Set to evaluation mode

    # Determine which pixels to iterate over
    if prediction_mask is not None and prediction_mask.shape == (height, width):
        print(f"Using prediction mask. Predicting for {np.sum(prediction_mask)} pixels.")
        rows, cols = np.where(prediction_mask)
        pixel_indices_to_predict = zip(rows, cols)
        num_pixels_to_predict = len(rows)
    else:
        if prediction_mask is not None:
            print("Warning: prediction_mask shape mismatch or invalid. Predicting for all pixels.")
        # Fallback to iterating over all pixels if no mask is provided
        rows, cols = np.indices((height, width))
        pixel_indices_to_predict = zip(rows.flatten(), cols.flatten())
        num_pixels_to_predict = height * width


    for r, c in tqdm(pixel_indices_to_predict, desc="Predicting ATC", total=num_pixels_to_predict):
        # Per-pixel ERA5 for prediction timeline (single band)
        pixel_era5_pred = era5_for_prediction_numpy[:, r, c]  # (time_pred)
        pixel_era5_pred_tensor = torch.from_numpy(pixel_era5_pred).float().to(device)

        # Skip if no model snapshots or ERA5 parameters are all NaN
        if np.isnan(C_snaps[:, r, c]).all():
            continue
        if torch.isnan(pixel_era5_pred_tensor).all():
            continue

        pixel_ensemble_predictions_list = [] # List to hold (time_pred) arrays for mean predictions

        for snap_idx in range(num_snapshots):
            # Load parameters for the current snapshot and pixel
            current_C = C_snaps[snap_idx, r, c]
            current_A = A_snaps[snap_idx, r, c]
            current_phi = phi_snaps[snap_idx, r, c]
            current_b = b_snaps[snap_idx, r, c]
            # Handle new params with backward compatibility
            current_A_2 = A_2_snaps[snap_idx, r, c] if A_2_snaps is not None else 0.0
            current_phi_2 = phi_2_snaps[snap_idx, r, c] if phi_2_snaps is not None else 0.0

            current_params = {
                'C': torch.tensor(current_C, device=device),
                'A': torch.tensor(current_A, device=device),
                'phi': torch.tensor(current_phi, device=device),
                'A_2': torch.tensor(current_A_2, device=device),
                'phi_2': torch.tensor(current_phi_2, device=device),
                'b': torch.tensor(current_b, device=device),
            }
            
            # Check if any parameter for this specific snapshot is NaN. If so, this snapshot can't predict.
            if any(torch.isnan(p.data) for p in current_params.values()): # Check .data for Parameter objects
                # This snapshot's prediction will be all NaNs for this pixel
                nan_preds_for_snapshot = np.full(num_times_pred, np.nan, dtype=np.float32)
                pixel_ensemble_predictions_list.append(nan_preds_for_snapshot)
                continue

            temp_model.load_state_dict(current_params)
            
            with torch.no_grad():
                # Model forward now only returns the mean prediction
                preds_tensor = temp_model(doy_for_prediction_tensor, pixel_era5_pred_tensor)
                pixel_ensemble_predictions_list.append(preds_tensor.cpu().numpy()) # Store (time_pred)
                

        if pixel_ensemble_predictions_list:
            # Stack along a new axis (axis 0: snapshots) -> (num_snapshots, time_pred)
            pixel_ensemble_preds_stack = np.stack(pixel_ensemble_predictions_list, axis=0)
            
            # Calculate mean of predictions and variance of predictions (ensemble uncertainty)
            mean_of_predictions = np.nanmean(pixel_ensemble_preds_stack, axis=0)
            variance_of_predictions = np.nanvar(pixel_ensemble_preds_stack, axis=0)

            atc_predictions_mean[:, r, c] = mean_of_predictions
            atc_predictions_variance[:, r, c] = variance_of_predictions
        # If list is empty (e.g., all params were NaN), outputs remain NaN

    print("Finished ATC prediction from loaded snapshots.")
    return atc_predictions_mean, atc_predictions_variance