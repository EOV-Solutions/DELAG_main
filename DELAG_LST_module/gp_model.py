"""
Gaussian Process (GP) model for residuals using GPyTorch.
"""
import torch
import gpytorch
import numpy as np
import pandas as pd # For feature naming consistency if needed
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import warnings # Import the standard warnings module
import os # Import os module for file operations
from typing import Optional, List

import config # Assuming your config.py is accessible

# Define the GP model
class ApproximateGPModelResiduals(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        """
        Approximate GP model for LST residuals.

        Args:
            inducing_points (torch.Tensor): Tensor of inducing point locations 
                                          (num_inducing_points, num_features).
        """
        # Ensure inducing_points is float32
        if inducing_points.dtype != torch.float32:
            inducing_points = inducing_points.float()
            
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        
        # Mean and Kernel (RBF Kernel)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Forward pass of the GP model.

        Args:
            x (torch.Tensor): Input features (batch_size, num_features).

        Returns:
            gpytorch.distributions.MultivariateNormal: Predictive distribution.
        """
        # Ensure x is float32
        if x.dtype != torch.float32:
            x = x.float()
            
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def _train_gp_model_internal( # Renamed to internal, takes inducing_points
    train_x: torch.Tensor, 
    train_y: torch.Tensor, 
    inducing_points: torch.Tensor, # Added inducing_points as direct argument
    app_config: 'config'
) -> tuple[ApproximateGPModelResiduals, gpytorch.likelihoods.GaussianLikelihood, list[float]]:
    """
    Internal function to train the Approximate GP model for residuals.

    Args:
        train_x (torch.Tensor): Training features (num_samples, num_features).
        train_y (torch.Tensor): Training targets (residuals) (num_samples).
        inducing_points (torch.Tensor): Pre-determined inducing points.
        app_config: Configuration object.

    Returns:
        tuple[ApproximateGPModelResiduals, gpytorch.likelihoods.GaussianLikelihood, list[float]]:
            - Trained GP model.
            - Trained likelihood.
            - List of mean losses for each logging interval.
    """
    device = torch.device(app_config.GP_DEVICE if torch.cuda.is_available() else "cpu")
    train_x, train_y = train_x.to(device), train_y.to(device)
    inducing_points = inducing_points.to(device) # Ensure inducing points are on the correct device

    model = ApproximateGPModelResiduals(inducing_points=inducing_points).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()}, 
        {'params': likelihood.parameters()}
    ], lr=app_config.GP_LEARNING_RATE_INITIAL)

    # Use MLL for SVGP
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    # Create DataLoader for mini-batch training
    dataset = TensorDataset(train_x, train_y)
    dataloader = DataLoader(dataset, batch_size=app_config.GP_MINI_BATCH_SIZE, shuffle=True)

    print(f"Starting GP model training with {app_config.GP_EPOCHS_INITIAL + app_config.GP_EPOCHS_FINAL} epochs...")
    
    # Loss logging setup
    total_epochs_gp = app_config.GP_EPOCHS_INITIAL + app_config.GP_EPOCHS_FINAL
    loss_logging_interval_gp = getattr(app_config, 'GP_LOSS_LOGGING_INTERVAL', 10)
    num_loss_intervals_gp = (total_epochs_gp + loss_logging_interval_gp -1) // loss_logging_interval_gp
    interval_losses_output_gp = [np.nan] * num_loss_intervals_gp
    current_interval_losses_gp = []
    current_interval_idx_gp = 0
    global_epoch_count = 0

    # Initial learning rate phase
    for i in range(app_config.GP_EPOCHS_INITIAL):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(dataloader.dataset)
        current_interval_losses_gp.append(epoch_loss) # Store average loss for this epoch
        global_epoch_count += 1

        if global_epoch_count % loss_logging_interval_gp == 0:
            if current_interval_losses_gp:
                interval_losses_output_gp[current_interval_idx_gp] = np.mean(current_interval_losses_gp)
            current_interval_losses_gp = []
            current_interval_idx_gp += 1

        if (i + 1) % 10 == 0: # Keep some print statements for feedback during long training
            print(f"GP Epoch (Initial LR) {i+1}/{app_config.GP_EPOCHS_INITIAL}, Avg Loss: {epoch_loss:.3f}, LR: {app_config.GP_LEARNING_RATE_INITIAL}")

    # Lower learning rate phase
    for param_group in optimizer.param_groups:
        param_group['lr'] = app_config.GP_LEARNING_RATE_FINAL
    
    for i in range(app_config.GP_EPOCHS_FINAL):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(dataloader.dataset)
        current_interval_losses_gp.append(epoch_loss) # Store average loss for this epoch
        global_epoch_count += 1

        if global_epoch_count % loss_logging_interval_gp == 0:
            if current_interval_losses_gp:
                interval_losses_output_gp[current_interval_idx_gp] = np.mean(current_interval_losses_gp)
            current_interval_losses_gp = []
            if current_interval_idx_gp < num_loss_intervals_gp: # Prevent index out of bounds if last interval is full
                 current_interval_idx_gp += 1

        if (i + 1) % 2 == 0:
             print(f"GP Epoch (Final LR) {i+1+app_config.GP_EPOCHS_INITIAL}/{total_epochs_gp}, Avg Loss: {epoch_loss:.3f}, LR: {app_config.GP_LEARNING_RATE_FINAL}")

    # Handle any remaining losses in the last partial interval
    if current_interval_losses_gp and current_interval_idx_gp < num_loss_intervals_gp:
        interval_losses_output_gp[current_interval_idx_gp] = np.mean(current_interval_losses_gp)

    print("GP model training finished.")
    return model, likelihood, interval_losses_output_gp

def save_gp_model(model: ApproximateGPModelResiduals, likelihood: gpytorch.likelihoods.GaussianLikelihood, inducing_points: torch.Tensor, filepath: str, interval_losses: list[float] = None):
    """
    Saves the GP model state_dict, likelihood state_dict, inducing points, and optionally interval losses.
    Args:
        model: Trained GP model.
        likelihood: Trained likelihood.
        inducing_points (torch.Tensor): Inducing points used for the model.
        filepath (str): Path to save the model components (.pth).
        interval_losses (list[float], optional): List of mean losses per interval. If provided, saved to a .npy file with similar name.
    """
    state = {
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'inducing_points': inducing_points.cpu() # Save inducing points on CPU
    }
    torch.save(state, filepath)
    print(f"GP model, likelihood, and inducing points saved to {filepath}")

    if interval_losses is not None:
        loss_filepath = filepath.replace('.pth', '_interval_losses.npy') # More descriptive filename
        try:
            np.save(loss_filepath, np.array(interval_losses))
            print(f"GP interval losses saved to {loss_filepath}")
        except Exception as e:
            print(f"Error saving GP interval losses to {loss_filepath}: {e}")

def load_gp_model(filepath: str, app_config: 'config') -> tuple[ApproximateGPModelResiduals, gpytorch.likelihoods.GaussianLikelihood]:
    """
    Loads the GP model state_dict, likelihood state_dict, and inducing points.
    Initializes a new model instance with these components.

    Args:
        filepath (str): Path to the saved model components.
        app_config: Configuration object.

    Returns:
        tuple[ApproximateGPModelResiduals, gpytorch.likelihoods.GaussianLikelihood]:
            - Loaded and initialized GP model.
            - Loaded and initialized likelihood.
    """
    device = torch.device(app_config.GP_DEVICE if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GP model file not found at {filepath}")
        
    state = torch.load(filepath, map_location=device)
    
    inducing_points = state['inducing_points'].to(device)
    
    model = ApproximateGPModelResiduals(inducing_points=inducing_points).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    
    model.load_state_dict(state['model_state_dict'])
    likelihood.load_state_dict(state['likelihood_state_dict'])
    
    print(f"GP model, likelihood, and inducing points loaded from {filepath}")
    return model, likelihood

def load_gp_interval_losses(model_filepath: str) -> Optional[List[float]]:
    """
    Loads GP interval losses if the corresponding _interval_losses.npy file exists.

    Args:
        model_filepath (str): Path to the primary GP model .pth file.

    Returns:
        Optional[List[float]]: List of losses, or None if file not found or error occurs.
    """
    loss_filepath = model_filepath.replace('.pth', '_interval_losses.npy')
    if os.path.exists(loss_filepath):
        try:
            losses = np.load(loss_filepath)
            print(f"GP interval losses loaded from {loss_filepath}")
            return list(losses)
        except Exception as e:
            print(f"Error loading GP interval losses from {loss_filepath}: {e}")
            return None
    else:
        # print(f"GP interval loss file not found: {loss_filepath}") # Optional: for debugging
        return None

def predict_gp_residuals(
    model: ApproximateGPModelResiduals, 
    likelihood: gpytorch.likelihoods.GaussianLikelihood, 
    features_tensor: torch.Tensor, # Changed from np.ndarray to torch.Tensor
    app_config: 'config'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predicts residuals for a given set of features using the trained GP model.

    Args:
        model: Trained GP model.
        likelihood: Trained likelihood.
        features_tensor (torch.Tensor): Features for prediction as a tensor.
        app_config: Configuration object.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Predicted mean of residuals (flattened).
            - Predicted variance of residuals (flattened).
    """
    device = torch.device(app_config.GP_DEVICE if torch.cuda.is_available() else "cpu")
    model.to(device)
    likelihood.to(device)
    model.eval()
    likelihood.eval()

    # The features are already a tensor, just move to the correct device
    features_tensor = features_tensor.to(device)
    
    # Batch prediction
    # Streaming over batches without accumulating distribution objects
    prediction_bs = getattr(app_config, 'GP_PREDICTION_BATCH_SIZE', app_config.GP_MINI_BATCH_SIZE)
    num_rows = features_tensor.size(0)
    mean_out = np.empty((num_rows,), dtype=np.float32)
    var_out = np.empty((num_rows,), dtype=np.float32)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for start in tqdm(range(0, num_rows, prediction_bs), desc="Predicting GP Batches"):
            end = min(start + prediction_bs, num_rows)
            batch_features = features_tensor[start:end]
            preds = likelihood(model(batch_features))
            mean_out[start:end] = preds.mean.detach().cpu().float().numpy()
            var_out[start:end] = preds.variance.detach().cpu().float().numpy()

    return mean_out, var_out


def predict_gp_residuals_streaming(
    model: ApproximateGPModelResiduals,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    preprocessed_data: dict,
    app_config: 'config',
    prediction_mask: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict residuals by streaming features in time and spatial chunks to bound memory.

    Args:
        model: Trained GP model
        likelihood: Trained likelihood
        preprocessed_data: Dict with keys 's2_reflectance_stack', 'doy_stack', 'lon_coords', 'lat_coords'
        app_config: Configuration
        prediction_mask: Optional HxW boolean array of pixels to predict; if None, predict all

    Returns:
        (mean, variance) residual maps of shape (T, H, W)
    """
    device = torch.device(app_config.GP_DEVICE if torch.cuda.is_available() else "cpu")
    model.to(device)
    likelihood.to(device)
    model.eval()
    likelihood.eval()

    # Unpack arrays
    s2_reflectance_stack = preprocessed_data['s2_reflectance_stack']  # (T, B, H, W)
    doy_stack_numpy = preprocessed_data['doy_stack']                   # (T,)
    lon_coords = preprocessed_data['lon_coords']                       # (H, W)
    lat_coords = preprocessed_data['lat_coords']                       # (H, W)

    T, num_bands, H, W = s2_reflectance_stack.shape

    # Prepare output arrays
    gp_mean_residuals_full = np.full((T, H, W), np.nan, dtype=np.float32)
    gp_variance_residuals_full = np.full((T, H, W), np.nan, dtype=np.float32)

    # Mask handling
    if prediction_mask is not None:
        pixel_indices_y, pixel_indices_x = np.where(prediction_mask)
    else:
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        pixel_indices_y = yy.flatten()
        pixel_indices_x = xx.flatten()

    num_masked_pixels = len(pixel_indices_y)
    if num_masked_pixels == 0:
        return gp_mean_residuals_full, gp_variance_residuals_full

    # Chunking parameters
    chunk_pixels = getattr(app_config, 'GP_PREDICTION_CHUNK_PIXELS', 10000)
    chunk_timesteps = getattr(app_config, 'GP_PREDICTION_CHUNK_TIMESTEPS', 8)
    prediction_bs = getattr(app_config, 'GP_PREDICTION_BATCH_SIZE', getattr(app_config, 'GP_MINI_BATCH_SIZE', 1024))
    compute_variance = getattr(app_config, 'GP_PREDICT_VARIANCE', True)

    # Iterate over time and spatial chunks
    for t_start in range(0, T, chunk_timesteps):
        t_end = min(T, t_start + chunk_timesteps)
        ct = t_end - t_start
        doy_slice = doy_stack_numpy[t_start:t_end]  # (ct,)

        for p_start in range(0, num_masked_pixels, chunk_pixels):
            p_end = min(num_masked_pixels, p_start + chunk_pixels)
            py = pixel_indices_y[p_start:p_end]
            px = pixel_indices_x[p_start:p_end]
            cp = len(py)
            if cp == 0:
                continue

            # Build features for (ct, cp)
            # Coordinates (cp,)
            lon_vals = lon_coords[py, px]
            lat_vals = lat_coords[py, px]
            # Expanded to (ct, cp) then flatten to (ct*cp,)
            lon_exp = np.tile(lon_vals, (ct, 1)).flatten()
            lat_exp = np.tile(lat_vals, (ct, 1)).flatten()
            doy_exp = np.tile(doy_slice[:, np.newaxis], (1, cp)).flatten()

            # S2 bands: (ct, B, cp) -> (ct, cp, B) -> (ct*cp, B)
            s2_chunk = s2_reflectance_stack[t_start:t_end, :, py, px]
            s2_flat = s2_chunk.transpose(0, 2, 1).reshape(-1, num_bands)

            # Stack features: [doy, lon, lat, s2_bands...]
            features_chunk = np.concatenate([
                doy_exp[:, None].astype(np.float32),
                lon_exp[:, None].astype(np.float32),
                lat_exp[:, None].astype(np.float32),
                s2_flat.astype(np.float32)
            ], axis=1)

            # Predict in batches and write directly to output slices
            num_rows = features_chunk.shape[0]
            mean_chunk = np.empty((num_rows,), dtype=np.float32)
            var_chunk = np.empty((num_rows,), dtype=np.float32) if compute_variance else None

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                for r_start in range(0, num_rows, prediction_bs):
                    r_end = min(num_rows, r_start + prediction_bs)
                    batch_features = torch.from_numpy(features_chunk[r_start:r_end]).to(device)
                    preds = likelihood(model(batch_features))
                    mean_chunk[r_start:r_end] = preds.mean.detach().cpu().float().numpy()
                    if compute_variance:
                        var_chunk[r_start:r_end] = preds.variance.detach().cpu().float().numpy()

            # Reshape back to (ct, cp)
            mean_reshaped = mean_chunk.reshape(ct, cp)
            if compute_variance:
                var_reshaped = var_chunk.reshape(ct, cp)
            else:
                var_reshaped = np.zeros_like(mean_reshaped, dtype=np.float32)

            # Write into full arrays
            gp_mean_residuals_full[t_start:t_end, py, px] = mean_reshaped
            gp_variance_residuals_full[t_start:t_end, py, px] = var_reshaped

    return gp_mean_residuals_full, gp_variance_residuals_full


def prepare_gp_training_data(
    preprocessed_data: dict, 
    atc_predictions: np.ndarray, 
    app_config: 'config', 
    is_prediction: bool = False, 
    prediction_mask: Optional[np.ndarray] = None
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, tuple[int, int, int], torch.Tensor]:
    """
    Prepares training or prediction data for the GP model.
    """
    print("Preparing GP data using vectorized operations...")
    # Extract data from dictionary
    lst_stack = preprocessed_data['lst_stack']
    s2_reflectance_stack = preprocessed_data['s2_reflectance_stack']
    doy_stack_numpy = preprocessed_data['doy_stack']
    # era5_stack = preprocessed_data.get('era5_stack')
    lon_coords = preprocessed_data['lon_coords']
    lat_coords = preprocessed_data['lat_coords']
    
    T, H, W = lst_stack.shape
    original_dims = (T, H, W)

    # Determine the mask to use for pixel selection
    if is_prediction and prediction_mask is not None:
        mask_to_use = prediction_mask
        print(f"Using provided prediction mask: {np.sum(mask_to_use)} pixels selected.")
    elif 'training_pixel_mask' in preprocessed_data:
        mask_to_use = preprocessed_data['training_pixel_mask']
        print(f"Using 'training_pixel_mask' from data: {np.sum(mask_to_use)} pixels selected.")
    else:
        mask_to_use = np.ones((H, W), dtype=bool)
        print("Warning: No spatial mask provided. Using all pixels.")

    # Calculate residuals (LST - ATC)
    residuals = lst_stack - atc_predictions
    residuals[np.isnan(lst_stack)] = np.nan

    # Get indices of pixels to process based on the mask
    pixel_indices_y, pixel_indices_x = np.where(mask_to_use)
    num_masked_pixels = len(pixel_indices_y)
    
    # --- Feature Engineering (Vectorized) ---
    features_list = []
    feature_names = []

    # Expand coordinates and time features for each masked pixel
    lon_expanded = np.tile(lon_coords[pixel_indices_y, pixel_indices_x], (T, 1)).flatten()
    lat_expanded = np.tile(lat_coords[pixel_indices_y, pixel_indices_x], (T, 1)).flatten()
    doy_expanded = np.tile(doy_stack_numpy[:, np.newaxis], (1, num_masked_pixels)).flatten()
    
    features_list.extend([doy_expanded, lon_expanded, lat_expanded])
    feature_names.extend(['doy', 'lon', 'lat'])
    
    # if era5_stack is not None:
    #     era5_masked = era5_stack[:, pixel_indices_y, pixel_indices_x]
    #     features_list.append(era5_masked.flatten())
    #     feature_names.append('era5')

    s2_masked = s2_reflectance_stack[:, :, pixel_indices_y, pixel_indices_x]
    s2_flattened = s2_masked.transpose(0, 2, 1).reshape(-1, s2_masked.shape[1])
    for i in range(s2_flattened.shape[1]):
        features_list.append(s2_flattened[:, i])
        feature_names.append(f's2_band_{i+1}')
    
    # Final feature matrix for all time steps for the selected pixels
    features_all_pixel_time_flat = np.vstack(features_list).T
    print(f"Constructed feature matrix with shape: {features_all_pixel_time_flat.shape}")

    if is_prediction:
        # For prediction, we don't have a `y` target, and inducing points are loaded from the model
        # We also don't filter out NaN features here, as predictions might be desired even with partial data,
        # though they will likely result in NaN outputs from the model.
        train_y_tensor = torch.empty(0)
        inducing_points_tensor = torch.empty(0)
        train_x_tensor = torch.from_numpy(features_all_pixel_time_flat).float()
    else:
        # For training, we must filter to only clear-sky observations with valid features
        residuals_masked = residuals[:, pixel_indices_y, pixel_indices_x]
        
        # 1. Mask for valid residuals (clear-sky)
        clear_obs_mask_flat = ~np.isnan(residuals_masked.flatten())
        
        # 2. Mask for valid features (no NaNs in any feature for a given observation)
        valid_features_mask = ~np.isnan(features_all_pixel_time_flat).any(axis=1)
        
        # 3. Combine masks to get final training set
        combined_training_mask = clear_obs_mask_flat & valid_features_mask
        
        train_x_flat = features_all_pixel_time_flat[combined_training_mask]
        train_y_flat = residuals_masked.flatten()[combined_training_mask]

        print(f"Filtered to {train_x_flat.shape[0]} valid training points (clear-sky and valid features).")

        train_x_tensor = torch.from_numpy(train_x_flat).float()
        train_y_tensor = torch.from_numpy(train_y_flat).float()
        
        # Inducing Points Initialization
        num_inducing = min(app_config.GP_NUM_INDUCING_POINTS, train_x_tensor.size(0))
        if num_inducing > 0:
            indices = torch.randperm(train_x_tensor.size(0))[:num_inducing]
            inducing_points_tensor = train_x_tensor[indices, :]
        else:
            inducing_points_tensor = torch.empty(0, train_x_tensor.shape[1], dtype=torch.float32)
            print("Warning: No training data available to select inducing points.")

    return train_x_tensor, train_y_tensor, features_all_pixel_time_flat, original_dims, inducing_points_tensor

def train_and_save_gp_model(preprocessed_data: dict, atc_predictions:np.ndarray, app_config: 'config'):
    """
    Main function to orchestrate the GP model training and saving process.
    """
    # 1. Prepare data (calculate residuals, extract features, apply mask)
    train_x, train_y, _, _, inducing_points_initial = prepare_gp_training_data(
        preprocessed_data, atc_predictions, app_config, is_prediction=False
    )
    
    if train_x.shape[0] == 0:
        print("Skipping GP model training as no clear-sky training data is available.")
        return

    # 2. Train the GP model
    model, likelihood, interval_losses = _train_gp_model_internal(
        train_x, train_y, inducing_points_initial, app_config
    )

    # 3. Save the trained model and losses
    model_save_path = os.path.join(app_config.MODEL_WEIGHTS_PATH, app_config.GP_MODEL_WEIGHT_FILENAME)
    learned_inducing_points = model.variational_strategy.inducing_points.data.clone().detach()
    save_gp_model(model, likelihood, learned_inducing_points, model_save_path, interval_losses)

def load_and_predict_gp_residuals(
    preprocessed_data: dict, 
    atc_predictions: np.ndarray, 
    app_config: 'config',
    prediction_mask: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a trained GP model and predicts residuals for the given data.
    Only predicts for pixels specified in the prediction_mask if provided.
    """
    print("Loading GP model for residual prediction...")
    model_load_path = os.path.join(app_config.MODEL_WEIGHTS_PATH, app_config.GP_MODEL_WEIGHT_FILENAME)
    model, likelihood = load_gp_model(model_load_path, app_config)

    # Streamed prediction to bound memory and CPU spikes
    gp_mean_residuals_full, gp_variance_residuals_full = predict_gp_residuals_streaming(
        model=model,
        likelihood=likelihood,
        preprocessed_data=preprocessed_data,
        app_config=app_config,
        prediction_mask=prediction_mask
    )

    return gp_mean_residuals_full, gp_variance_residuals_full

# # --- Test block for gp_model.py ---
# if __name__ == '__main__':
#     # Create a dummy config (can inherit from your actual config if it has defaults)
#     class DummyConfig(config.Config): # Inherit from actual config to get some defaults
#         def __init__(self):
#             super().__init__() # Ensure base class init is called if it does anything
#             # Override specific settings for the test if needed
#             self.DEVICE = "cpu" # General device setting
#             self.ATC_DEVICE = "cpu" # Specific for ATC (not used directly in GP tests)
#             self.GP_DEVICE = "cpu"  # Specific for GP
#             self.GP_NUM_INDUCING_POINTS = 32 # Smaller for faster test
#             self.GP_EPOCHS_INITIAL = 3 # Minimal epochs for test
#             self.GP_EPOCHS_FINAL = 2   # Minimal epochs for test
#             self.GP_MINI_BATCH_SIZE = 1024
#             self.GP_LEARNING_RATE_INITIAL = 0.01
#             self.GP_LEARNING_RATE_FINAL = 0.001
#             # Define GP_RESIDUAL_FEATURES to match dummy data: 4 S2 bands + 2 coords
#             # This GP_RESIDUAL_FEATURES in config might become misleading as features are now dynamic.
#             # The actual number of features is determined in prepare_gp_training_data.
#             self.GP_RESIDUAL_FEATURES_CONFIG_IGNORED = ['s2_b1', 's2_b2', 's2_b3', 's2_b4', 'lon', 'lat']
#             self.GP_USE_TEMPORAL_MEAN_S2_FEATURES = False # Test default behavior first
#             self.GP_USE_NDVI_FEATURE = False # Test default behavior
#             self.S2_RED_INDEX = 2 # Example for dummy S2 [B,G,R,N]
#             self.S2_NIR_INDEX = 3 # Example for dummy S2 [B,G,R,N]
#             self.GP_LOSS_LOGGING_INTERVAL = 2 # For testing, log more frequently

#     test_app_config_default = DummyConfig()
    
#     # Test with GP_USE_TEMPORAL_MEAN_S2_FEATURES = True
#     class DummyConfigTemporalMeanS2(DummyConfig):
#         def __init__(self):
#             super().__init__()
#             self.GP_DEVICE = "cpu" # Ensure GP_DEVICE is set, can be overridden by specific tests
#             self.ATC_DEVICE = "cpu" # Ensure ATC_DEVICE is set
#             self.GP_USE_TEMPORAL_MEAN_S2_FEATURES = True
#             self.GP_USE_NDVI_FEATURE = False
#             self.GP_LOSS_LOGGING_INTERVAL = 2 # Override for test
            
#     test_app_config_temporal_mean_s2 = DummyConfigTemporalMeanS2()

#     # Test with GP_USE_NDVI_FEATURE = True
#     class DummyConfigNDVI(DummyConfig):
#         def __init__(self):
#             super().__init__()
#             self.GP_DEVICE = "cpu" # Ensure GP_DEVICE is set
#             self.ATC_DEVICE = "cpu" # Ensure ATC_DEVICE is set
#             self.GP_USE_TEMPORAL_MEAN_S2_FEATURES = False # Ensure this is False if NDVI is primary
#             self.GP_USE_NDVI_FEATURE = True
#             self.GP_LOSS_LOGGING_INTERVAL = 2 # Override for test

#     test_app_config_ndvi = DummyConfigNDVI()

#     # Dummy preprocessed_data
#     num_times_test, height_test, width_test = 3, 5, 5
#     num_s2_bands_test = 4 # Matching the change

#     # LST stack with some NaNs
#     dummy_lst_stack = np.random.rand(num_times_test, height_test, width_test).astype(np.float32) * 10 + 290
#     dummy_lst_stack[0, 0, 0] = np.nan # Simulate a cloudy pixel
#     dummy_lst_stack[1, 1, 1:3] = np.nan

#     # S2 stack (time, bands, height, width)
#     dummy_s2_stack = np.random.rand(num_times_test, num_s2_bands_test, height_test, width_test).astype(np.float32)
#     dummy_s2_stack[0, :, 0, 0] = np.nan # S2 can also have NaNs if source was NaN

#     # Dummy NDVI stack (time, height, width) - will be populated by data_preprocessing if active
#     # For direct testing of gp_model.py, we can simulate its presence if GP_USE_NDVI_FEATURE is true in test config
#     dummy_ndvi_stack = np.random.rand(num_times_test, height_test, width_test).astype(np.float32)
#     dummy_ndvi_stack[0, 0, 1] = np.nan # Simulate some NaN in NDVI

#     # Coordinates
#     dummy_lon_coords = np.linspace(0, 1, width_test).reshape(1, -1).repeat(height_test, axis=0)
#     dummy_lat_coords = np.linspace(0, 1, height_test).reshape(-1, 1).repeat(width_test, axis=1)

#     # Dummy ATC predictions (all clear for simplicity here)
#     dummy_atc_predictions = np.random.rand(num_times_test, height_test, width_test).astype(np.float32) * 10 + 288

#     preprocessed_data_test_base = {
#         "lst_stack": dummy_lst_stack,
#         "s2_reflectance_stack": dummy_s2_stack, 
#         "lon_coords": dummy_lon_coords,
#         "lat_coords": dummy_lat_coords,
#         # "cloud_mask_stack": dummy_cloud_mask # No longer used
#     }

#     print("--- Testing GP Model module --- ")
    
#     # Test scenarios
#     test_configs_to_run = {
#         "Instantaneous_S2": test_app_config_default,
#         "Temporal_Mean_S2": test_app_config_temporal_mean_s2,
#         "NDVI_Feature": test_app_config_ndvi
#     }

#     for test_name, current_test_config in test_configs_to_run.items():
#         print(f"\\n--- Running GP Model Test Scenario: {test_name} ---")
        
#         # Add dummy NDVI to preprocessed_data if this test scenario uses NDVI
#         current_preprocessed_data_test = preprocessed_data_test_base.copy()
#         if getattr(current_test_config, 'GP_USE_NDVI_FEATURE', False):
#             current_preprocessed_data_test['ndvi_stack'] = dummy_ndvi_stack
#             print("  (Added dummy NDVI stack for this test scenario)")
#         else:
#             # Ensure ndvi_stack is not present if not testing NDVI, or is None
#             current_preprocessed_data_test['ndvi_stack'] = None 

#         try:
#             # Test data preparation
#             train_x_out, train_y_out, features_pred_flat_out, original_dims_out, initial_inducing_points_out = prepare_gp_training_data(
#                 current_preprocessed_data_test, dummy_atc_predictions, current_test_config
#             )
#             print(f"prepare_gp_training_data output shapes for {test_name}:")
#             print(f"  train_x: {train_x_out.shape}")
#             print(f"  train_y: {train_y_out.shape}")
#             print(f"  features_pred_flat: {features_pred_flat_out.shape}")
#             print(f"  original_dims: {original_dims_out}")
#             print(f"  initial_inducing_points: {initial_inducing_points_out.shape}")

#             expected_total_obs = num_times_test * height_test * width_test
#             # Determine expected number of features based on the current test config
#             if getattr(current_test_config, 'GP_USE_NDVI_FEATURE', False):
#                 expected_num_features_for_test = 1 + 2 # NDVI + lon + lat
#             else:
#                 expected_num_features_for_test = num_s2_bands_test + 2 # S2 bands + lon + lat
            
#             assert features_pred_flat_out.shape == (expected_total_obs, expected_num_features_for_test), \
#                 f"Shape mismatch for prediction features in {test_name}. Expected ({expected_total_obs}, {expected_num_features_for_test}), Got {features_pred_flat_out.shape}"
#             assert original_dims_out == (num_times_test, height_test, width_test), f"Original dimensions mismatch in {test_name}"
#             assert train_x_out.shape[1] == expected_num_features_for_test, \
#                 f"Number of features in train_x is incorrect for {test_name}. Expected {expected_num_features_for_test}, Got {train_x_out.shape[1]}"

#             # Basic check on training data points (should be less than total if there are NaNs in LST or features)
#             # This check becomes more complex with conditional features, so focusing on shapes primarily.
#             print(f"Number of training points for {test_name}: {train_x_out.shape[0]}")

#             # Test full pipeline: train_and_predict_all_gp_residuals
#             # This old function is now split. We test train_and_save then load_and_predict
            
#             # Path for dummy model saving
#             dummy_gp_model_filename = f"dummy_gp_model_{test_name}.pth"
#             current_test_config.GP_MODEL_WEIGHT_FILENAME = dummy_gp_model_filename # Override for test
#             dummy_model_weights_dir = "dummy_gp_weights_output"
#             os.makedirs(dummy_model_weights_dir, exist_ok=True)
#             current_test_config.MODEL_WEIGHTS_PATH = dummy_model_weights_dir

#             if train_x_out.shape[0] > 0: # Only run full train/predict if there is training data
#                 print(f"Testing train_and_save_gp_model for {test_name}...")
#                 train_and_save_gp_model(
#                     current_preprocessed_data_test, dummy_atc_predictions, current_test_config
#                 )
#                 dummy_gp_model_path = os.path.join(current_test_config.MODEL_WEIGHTS_PATH, current_test_config.GP_MODEL_WEIGHT_FILENAME)
#                 assert os.path.exists(dummy_gp_model_path), f"GP model file was not saved for {test_name} at {dummy_gp_model_path}"
                
#                 # Test loading interval losses
#                 loaded_losses = load_gp_interval_losses(dummy_gp_model_path)
#                 assert loaded_losses is not None, f"GP interval losses could not be loaded for {test_name}"
#                 expected_num_gp_loss_intervals = (current_test_config.GP_EPOCHS_INITIAL + current_test_config.GP_EPOCHS_FINAL + current_test_config.GP_LOSS_LOGGING_INTERVAL -1) // current_test_config.GP_LOSS_LOGGING_INTERVAL
#                 assert len(loaded_losses) == expected_num_gp_loss_intervals, f"Loaded GP losses have incorrect length for {test_name}. Expected {expected_num_gp_loss_intervals}, got {len(loaded_losses)}"
#                 print(f"Loaded GP losses for {test_name}: {loaded_losses}")

#                 print(f"Model saved for {test_name}. Now testing load_and_predict_gp_residuals...")

#                 gp_mean_map, gp_var_map = load_and_predict_gp_residuals(
#                     current_preprocessed_data_test, dummy_atc_predictions, current_test_config
#                 )
                
#                 print(f"load_and_predict_gp_residuals output shapes for {test_name}:")
#                 print(f"  gp_mean_map: {gp_mean_map.shape}")
#                 print(f"  gp_var_map: {gp_var_map.shape}")
#                 assert gp_mean_map.shape == (num_times_test, height_test, width_test)
#                 assert gp_var_map.shape == (num_times_test, height_test, width_test)
#                 print(f"GP module test completed successfully for {test_name}.")
#             else:
#                 print("Skipping full GP train/predict test as no training data was generated (e.g., all LST was NaN).")
#                 # Test the case where no training data is available
#                 dummy_lst_all_nan = np.full_like(dummy_lst_stack, np.nan)
#                 preprocessed_data_all_nan_lst = current_preprocessed_data_test.copy()
#                 preprocessed_data_all_nan_lst["lst_stack"] = dummy_lst_all_nan
                
#                 print(f"Testing train_and_save_gp_model with NO training data for {test_name}...")
#                 # This should not save a file if training is skipped.
#                 train_and_save_gp_model(
#                     preprocessed_data_all_nan_lst, dummy_atc_predictions, current_test_config
#                 )
#                 # Ensure file for this specific "no data" scenario does not exist if train_and_save skips saving
#                 dummy_no_data_model_file = os.path.join(current_test_config.MODEL_WEIGHTS_PATH, current_test_config.GP_MODEL_WEIGHT_FILENAME)
#                 assert not os.path.exists(dummy_no_data_model_file.replace('.pth', '_interval_losses.npy')), \
#                     f"GP loss file for NO DATA scenario ({test_name}) should not exist if training was skipped."

#                 print(f"Testing load_and_predict_gp_residuals with NO training data (expecting NaNs due to no model or prior) for {test_name}...")
#                 gp_mean_map_no_train, gp_var_map_no_train = load_and_predict_gp_residuals(
#                     preprocessed_data_all_nan_lst, dummy_atc_predictions, current_test_config
#                 )

#                 assert np.all(np.isnan(gp_mean_map_no_train)), f"Mean map should be all NaN if no training data ({test_name})"
#                 assert np.all(np.isnan(gp_var_map_no_train)), f"Variance map should be all NaN if no training data ({test_name})"

#         except Exception as e:
#             print(f"Error during GP module test ({test_name}): {e}")
#             import traceback
#             traceback.print_exc()
    
#     # Clean up dummy model weights directory
#     import shutil
#     if os.path.exists(dummy_model_weights_dir):
#         shutil.rmtree(dummy_model_weights_dir)
#         print(f"Cleaned up dummy GP weights directory: {dummy_model_weights_dir}")

#     print("Finished GP Model module main execution.") 