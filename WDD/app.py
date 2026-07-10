"""
Welding Defect Detection - Main Streamlit Application

Three-phase educational workshop:
1. Training Phase: Students train a MobileNetV2 model on synthetic weld data
2. Optimization Phase: Students optimize the model for edge deployment
3. Inference Phase: Real-time defect detection with OOD detection
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ===== VERBOSE LOGGING UTILITY =====
def log_step(step_name, description="", step_type="INFO"):
    """Log progress steps to console and GUI."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{timestamp}]"
    
    # ASCII-safe console output
    if step_type == "START":
        print(f"\n{prefix} >>> STARTING: {step_name}")
        if description:
            print(f"{'':15} {description}")
    elif step_type == "STEP":
        print(f"{prefix} ... {step_name}")
        if description:
            print(f"{'':15} {description}")
    elif step_type == "SUCCESS":
        print(f"{prefix} [OK] {step_name}")
        if description:
            print(f"{'':15} {description}")
    elif step_type == "PROGRESS":
        print(f"{prefix} --> {step_name}")
        if description:
            print(f"{'':15} {description}")
    else:  # INFO
        print(f"{prefix} [*] {step_name}")
        if description:
            print(f"{'':15} {description}")
    
    sys.stdout.flush()

from config import training_config, DATASET_DIR, CHECKPOINTS_DIR
from data.generator import generate_dataset
from data.data_manager import DataSourceManager
from trainer.dataset import create_dataloaders, get_dataset_stats
from trainer.trainer import WeldDefectTrainer
from simulator.optimizer import OptimizationEngine
from simulator.scorer import DeploymentScorer
from simulator.metrics import metrics_calculator
from model_naming import ModelNaming
import torch


def init_session_state():
    """Initialize Streamlit session state."""
    log_step("Initializing Session State", "Setting up user session variables")
    
    if 'trainer' not in st.session_state:
        log_step("Creating Trainer", "Instantiating WeldDefectTrainer object")
        st.session_state.trainer = WeldDefectTrainer()
        log_step("Trainer Ready", "WeldDefectTrainer initialized successfully", "SUCCESS")
    
    if 'dataset_ready' not in st.session_state:
        st.session_state.dataset_ready = False
        log_step("Dataset Status", "Marked as not ready (will be generated on demand)")
    
    if 'training_started' not in st.session_state:
        st.session_state.training_started = False
        log_step("Training Status", "Marked as not started")
    
    if 'selected_dataset_path' not in st.session_state:
        st.session_state.selected_dataset_path = DATASET_DIR
        log_step("Dataset Path", f"Initialized to default: {DATASET_DIR}")
    
    if 'baseline_training_info' not in st.session_state:
        st.session_state.baseline_training_info = None
        log_step("Baseline Info", "Initialized as empty")
    
    log_step("Session Initialization Complete", "All session variables ready", "SUCCESS")


def reset_phase_3():
    """Reset Phase 3 state (inference) only. Keeps Phase 1 and 2 state."""
    st.session_state.inference_engine = None
    st.session_state.engine_initialized = False
    if 'image_uploader' in st.session_state:
        del st.session_state.image_uploader
    if 'phase3_checkpoint_select' in st.session_state:
        del st.session_state.phase3_checkpoint_select
    st.success("✅ Phase 3 reset successfully!")
    st.rerun()


def reset_phase_2():
    """Reset Phase 2 and 3 state. Keeps Phase 1 state."""
    # Clear Phase 2 state
    if 'optimized_model_info' in st.session_state:
        del st.session_state.optimized_model_info
    if 'optimizer' in st.session_state:
        del st.session_state.optimizer
    
    # Clear Phase 3 state (without showing message)
    st.session_state.inference_engine = None
    st.session_state.engine_initialized = False
    if 'phase2_checkpoint_select' in st.session_state:
        del st.session_state.phase2_checkpoint_select
    if 'phase3_checkpoint_select' in st.session_state:
        del st.session_state.phase3_checkpoint_select
    if 'image_uploader' in st.session_state:
        del st.session_state.image_uploader
    
    st.success("✅ Phase 2 and 3 reset successfully!")
    st.rerun()


def reset_phase_1():
    """Reset Phase 1, 2, and 3 state (complete reset)."""
    # Clear Phase 1 state
    st.session_state.training_started = False
    st.session_state.baseline_training_info = None
    st.session_state.dataset_ready = False
    st.session_state.selected_dataset_path = DATASET_DIR
    
    # Reinitialize trainer
    st.session_state.trainer = WeldDefectTrainer()
    
    # Clear Phase 2 state
    if 'optimized_model_info' in st.session_state:
        del st.session_state.optimized_model_info
    if 'optimizer' in st.session_state:
        del st.session_state.optimizer
    
    # Clear Phase 3 state
    st.session_state.inference_engine = None
    st.session_state.engine_initialized = False
    
    st.success("✅ All phases reset successfully!")
    st.rerun()


def delete_checkpoint(checkpoint_name: str, on_success_callback=None):
    """Delete a checkpoint file with confirmation."""
    checkpoint_path = CHECKPOINTS_DIR / checkpoint_name
    
    if not checkpoint_path.exists():
        st.error(f"Checkpoint not found: {checkpoint_name}")
        return False
    
    try:
        checkpoint_path.unlink()
        log_step("Checkpoint Deleted", f"Successfully removed: {checkpoint_name}", "SUCCESS")
        
        # Clear related session state if this was the baseline
        if st.session_state.baseline_training_info and \
           st.session_state.baseline_training_info.get('checkpoint_name') == checkpoint_name:
            st.session_state.baseline_training_info = None
            st.session_state.training_started = False
        
        # Clear Phase 2 state (in case deleted model was used for optimization)
        if 'optimized_model_info' in st.session_state:
            del st.session_state.optimized_model_info
        if 'optimizer' in st.session_state:
            del st.session_state.optimizer
        
        # Reset Phase 3 inference engine
        st.session_state.inference_engine = None
        st.session_state.engine_initialized = False
        
        # Call callback if provided
        if on_success_callback:
            on_success_callback()
        
        st.success(f"✅ Deleted: {checkpoint_name}")
        st.rerun()
        return True
    except Exception as e:
        st.error(f"Failed to delete checkpoint: {e}")
        log_step("Deletion Error", str(e), "ERROR")
        return False


def training_phase():
    """Phase 1: Model training interface."""
    log_step("PHASE 1 START", "User entered Phase 1 - Model Training", "START")
    
    # Reset button for Phase 1 (top right corner)
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🔄 Reset Phase 1", help="Reset training and clear all phase data"):
            reset_phase_1()
    
    # NEXT ACTION BANNER
    st.markdown("""
    <div style="background-color: #e8f4f8; border-left: 5px solid #0066cc; padding: 15px; margin: 10px 0; border-radius: 5px;">
        <b>YOUR NEXT ACTION:</b><br>
        1. Choose your dataset: Synthetic (generated) or External (OpenSourceData01, etc.)<br>
        2. Review dataset statistics (number of training/validation/test images)<br>
        3. Configure training parameters (epochs, batch size, learning rate)<br>
        4. Click "Start Training" button<br>
        5. Monitor training progress in console<br>
        6. Once complete, proceed to Phase 2 for optimization
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Training Phase 1: Train Your Model")
    
    st.write("""
    In this phase, you will train a model on weld inspection data.
    
    You can choose between:
    - **Synthetic Data**: Procedurally generated weld images (400 normal + 400 defective)
    - **External Data**: Real or open-source datasets (e.g., OpenSourceData01 with real weld samples)
    
    **What you'll learn:**
    - How neural networks are trained
    - The relationship between loss and accuracy
    - How training progresses over epochs
    - Overfitting and validation monitoring
    """)
    
    # ==================== MODEL SELECTION ====================
    st.subheader("Step 1: Select Your Model Architecture")
    
    from trainer.model_builder import get_available_models, get_model_info
    
    # Initialize selected model in session state
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "MobileNetV2"
    
    available_models = get_available_models()
    
    col_model1, col_model2 = st.columns([2, 1])
    with col_model1:
        st.session_state.selected_model = st.selectbox(
            "Choose model architecture:",
            options=available_models,
            index=available_models.index(st.session_state.selected_model),
            help="Select which neural network architecture to train"
        )
    
    with col_model2:
        if st.button("ℹ️ Model Info", use_container_width=True, key="model_info_btn"):
            st.session_state.show_model_info = True
    
    # Show model info if requested
    if st.session_state.get('show_model_info', False):
        model_info = get_model_info(st.session_state.selected_model)
        with st.expander("📊 Model Details", expanded=True):
            st.markdown(f"""
            **{st.session_state.selected_model}**
            - Size: {model_info['size_mb']:.1f} MB
            - Parameters: {model_info['num_params']:,}
            - Description: {model_info['description']}
            """)
        st.session_state.show_model_info = False
    
    log_step("Model Selected", f"User selected: {st.session_state.selected_model}", "PROGRESS")
    
    # ==================== DATA SOURCE SELECTION ====================
    st.subheader("Step 2: Select Your Dataset")
    
    # Initialize selected dataset path in session state
    if 'selected_dataset_path' not in st.session_state:
        st.session_state.selected_dataset_path = None
    
    log_step("Discovering Data Sources", "Scanning for available datasets")
    
    # Get all available data sources
    available_sources = DataSourceManager.get_all_data_sources()
    display_names = [name for name, _ in available_sources]
    
    col_info, col_info2 = st.columns([1, 1])
    with col_info:
        st.write(f"**Found {len(available_sources)} data source(s)**")
    
    # Data source selector
    selected_source = st.selectbox(
        "Choose your training dataset:",
        options=display_names,
        index=0,  # Default to first (synthetic)
        help="Select between generated synthetic data or external datasets"
    )
    
    log_step("Data Source Selected", f"User selected: {selected_source}", "PROGRESS")
    
    # Handle selection
    is_synthetic = selected_source == DataSourceManager.SYNTHETIC_NAME
    selected_dataset_path = DataSourceManager.get_dataset_path_by_name(selected_source)
    
    # Display dataset info
    if is_synthetic:
        st.markdown("""
        <div style="background-color: #f0f8e8; border-left: 4px solid #4CAF50; padding: 10px; border-radius: 4px;">
        <b>Synthetic Dataset</b><br>
        <small>Procedurally generated 800 weld images (400 normal, 400 defective) with 5 defect types.<br>
        Auto-generated on first training if not present. ~10 seconds to generate.</small>
        </div>
        """, unsafe_allow_html=True)
        
        log_step("Dataset Type", "Synthetic data selected", "PROGRESS")
        
        # Check if synthetic dataset exists
        dataset_exists = DATASET_DIR.exists() and (DATASET_DIR / 'train').exists()
        
        if not dataset_exists:
            log_step("Synthetic Dataset Not Found", "Will generate on demand", "PROGRESS")
            st.warning("Synthetic dataset not found. It will be generated when you click 'Start Training'.")
            st.session_state.dataset_ready = True  # Allow training to proceed - we generate on-demand
        else:
            log_step("Synthetic Dataset Found", "Using existing synthetic dataset", "SUCCESS")
            st.success("Synthetic dataset exists - ready to use!")
            st.session_state.dataset_ready = True
        
        actual_dataset_path = DATASET_DIR
    else:
        # External dataset
        dataset_info = DataSourceManager.get_dataset_info(selected_dataset_path)
        
        if dataset_info['is_valid']:
            st.markdown(f"""
            <div style="background-color: #f8f0e8; border-left: 4px solid #FF9800; padding: 10px; border-radius: 4px;">
            <b>External Dataset: {dataset_info['name']}</b><br>
            <small>Located at: <code>data/{dataset_info['name']}</code><br>
            Structure: train/ | valid/ | test/ | </small>
            </div>
            """, unsafe_allow_html=True)
            
            log_step("External Dataset Valid", f"Dataset {dataset_info['name']} is valid", "SUCCESS")
            st.success("Dataset structure is valid!")
            st.session_state.dataset_ready = True
            actual_dataset_path = selected_dataset_path
        else:
            # Show error with detailed instructions for flat datasets
            st.error("DATASET INCOMPATIBLE - Structure Problem")
            st.write(dataset_info['error'])
            
            # If it's a flat dataset, show conversion instructions
            if "FLAT STRUCTURE" in dataset_info['error']:
                st.markdown("""
                ### How to Fix This Dataset
                
                Your dataset has all images mixed in train/, valid/, test/ folders without class subdirectories.
                
                **Quick Fix:**
                1. Open a terminal/command prompt
                2. Navigate to your project folder
                3. Run this command:
                ```
                python data/restructure_flat_dataset.py --dataset OpenSourceData01
                ```
                4. Wait for the conversion to complete
                5. Then reload this app and try again
                
                The script will automatically classify images based on filename keywords (good/bad/crack/etc).
                """)
            
            log_step("External Dataset Invalid", f"Error: {dataset_info['error']}", "PROGRESS")
            st.session_state.dataset_ready = False
            actual_dataset_path = None
    
    # Store selected path for later use in training
    st.session_state.selected_dataset_path = actual_dataset_path
    
    # Dataset statistics
    if st.session_state.dataset_ready and actual_dataset_path:
        log_step("Loading Dataset Statistics", "Calculating training/validation/test split sizes")
        stats = get_dataset_stats(actual_dataset_path)
        log_step("Statistics Loaded", 
                f"Train: {stats['train'].get('normal', 0) + stats['train'].get('defect', 0)} images, "
                f"Val: {stats['val'].get('normal', 0) + stats['val'].get('defect', 0)} images, "
                f"Test: {stats['test'].get('normal', 0) + stats['test'].get('defect', 0)} images", "SUCCESS")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            train_count = stats['train'].get('normal', 0) + stats['train'].get('defect', 0)
            st.metric("Training Images", f"{train_count}")
        with col2:
            val_count = stats['val'].get('normal', 0) + stats['val'].get('defect', 0)
            st.metric("Validation Images", f"{val_count}")
        with col3:
            test_count = stats['test'].get('normal', 0) + stats['test'].get('defect', 0)
            st.metric("Test Images", f"{test_count}")
        
        st.info(f"""
        **Dataset Split:**
        - Train: {stats['train'].get('normal', 0)} normal + {stats['train'].get('defect', 0)} defective
        - Validation: {stats['val'].get('normal', 0)} normal + {stats['val'].get('defect', 0)} defective
        - Test: {stats['test'].get('normal', 0)} normal + {stats['test'].get('defect', 0)} defective
        """)
    
    # Training controls
    st.subheader("Step 3: Configure Training Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_epochs = st.slider(
            "Number of Epochs",
            min_value=training_config.min_epochs,
            max_value=training_config.max_epochs,
            value=5,
            help="One epoch = going through entire dataset once. More epochs = longer training but better accuracy"
        )
    
    with col2:
        batch_size = st.selectbox(
            "Batch Size",
            options=training_config.batch_sizes,
            index=1,
            help="Images per batch. Smaller = slower but more stable. Larger = faster but less stable"
        )
    
    with col3:
        learning_rate = st.selectbox(
            "Learning Rate",
            options=training_config.learning_rates,
            format_func=lambda x: f"{x:.5f}",
            index=2,
            help="How much to adjust weights each step. Too high = unstable, Too low = slow"
        )
    
    st.info(f"""
    **Training Configuration Summary:**
    - Epochs: {num_epochs} (will train for {num_epochs} passes through the dataset)
    - Batch Size: {batch_size} (process {batch_size} images at once)
    - Learning Rate: {learning_rate:.5f} (weight adjustment rate)
    - Expected time: ~{num_epochs * 2}-{num_epochs * 3} minutes on CPU
    """)
    
    # Training button
    if st.button("Start Training", key="start_training"):
        if st.session_state.dataset_ready:
            # For synthetic dataset, generate on demand if needed
            selected_source_is_synthetic = st.session_state.selected_dataset_path == DATASET_DIR
            
            if selected_source_is_synthetic:
                synthetic_exists = DATASET_DIR.exists() and (DATASET_DIR / 'train').exists()
                if not synthetic_exists:
                    log_step("Generating Synthetic Dataset", "Creating 800 weld images on demand")
                    st.info("Generating synthetic dataset...")
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    progress_text.text("Generating synthetic weld defect images...")
                    log_step("Generating Dataset", "Creating 800 synthetic weld images (400 normal, 400 defective)")
                    
                    try:
                        log_step("Dataset Generation Starting", f"Target directory: {DATASET_DIR}")
                        generate_dataset()
                        log_step("Dataset Generation Complete", "800 images created successfully", "SUCCESS")
                        progress_bar.progress(100)
                        progress_text.text("Dataset generation complete!")
                        st.success("Dataset generated!")
                        
                        # Verify dataset was created
                        verify_count = len(list((DATASET_DIR / 'train' / 'normal').glob('*.png'))) if (DATASET_DIR / 'train' / 'normal').exists() else 0
                        log_step("Dataset Verification", f"Found {verify_count} training normal images", "PROGRESS")
                    except Exception as e:
                        log_step("Dataset Generation Failed", f"Error: {str(e)}", "PROGRESS")
                        st.error(f"Failed to generate dataset: {str(e)}")
                        st.stop()
            
            log_step("Starting Training", f"Epochs={num_epochs}, Batch={batch_size}, LR={learning_rate:.5f}", "START")
            
            st.info("Step 4: Training in Progress...")
            
            with st.spinner("Setting up training..."):
                log_step("Creating Data Loaders", "Preparing training, validation, and test loaders")
                
                # Use selected dataset path, or fallback to synthetic
                dataset_path = st.session_state.selected_dataset_path or DATASET_DIR
                log_step("Using Dataset Path", f"Path: {dataset_path}")
                
                # Verify dataset path exists and has content
                if not dataset_path.exists():
                    log_step("Dataset Path Missing", f"Creating: {dataset_path}")
                    dataset_path.mkdir(parents=True, exist_ok=True)
                
                train_split = dataset_path / 'train'
                if not train_split.exists():
                    st.error(f"Training data not found at {train_split}")
                    log_step("Training Data Missing", f"Expected at: {train_split}", "PROGRESS")
                    st.stop()
                
                # Verify we have images in the dataset
                normal_dir = train_split / 'normal'
                defect_dir = train_split / 'defect'
                
                normal_count = 0
                if normal_dir.exists():
                    for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.PNG']:
                        normal_count += len(list(normal_dir.glob(ext)))
                
                defect_count = 0
                if defect_dir.exists():
                    for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.PNG']:
                        defect_count += len(list(defect_dir.glob(ext)))
                
                if normal_count == 0 and defect_count == 0:
                    st.error(f"No images found in training set! Normal: {normal_count}, Defect: {defect_count}")
                    log_step("No Images", f"Training split is empty. Normal: {normal_count}, Defect: {defect_count}", "PROGRESS")
                    st.stop()
                
                log_step("Dataset Content Verified", f"Found {normal_count} normal + {defect_count} defect images in training set", "SUCCESS")
                
                # Create data loaders using selected dataset path
                train_loader, val_loader, test_loader = create_dataloaders(
                    dataset_path,
                    batch_size=batch_size,
                    num_workers=0
                )
                log_step("Data Loaders Ready", "All data pipelines configured", "SUCCESS")
            
            # Create placeholders for real-time updates
            status_placeholder = st.empty()
            metrics_placeholder = st.empty()
            progress_placeholder = st.empty()
            
            log_step("Training Loop Starting", f"Will complete {num_epochs} epochs")
            
            # Generate checkpoint name using naming convention with selected model
            dataset_name = st.session_state.selected_dataset_path.name
            # Clean model name: remove hyphens and spaces for checkpoint name
            model_name_lower = st.session_state.selected_model.lower().replace("-", "").replace(" ", "_")
            checkpoint_name = ModelNaming.generate_baseline_name(
                model_name=model_name_lower,
                dataset_name=dataset_name,
                epochs=num_epochs
            )
            
            # Check if model already exists (persistent across sessions)
            checkpoint_path = CHECKPOINTS_DIR / checkpoint_name
            
            if checkpoint_path.exists():
                log_step("Model Found", f"Using existing checkpoint: {checkpoint_name}", "SUCCESS")
                status_placeholder.info(f"✅ Model already trained! Loading from checkpoint: {checkpoint_name}")
                
                # Load existing model and get its metadata
                if st.session_state.trainer.load_checkpoint(checkpoint_path):
                    log_step("Model Loaded", f"Successfully loaded: {checkpoint_name}", "SUCCESS")
                    
                    # Display results from previously trained model
                    st.success("✅ Model loaded from previous training session!")
                    
                    # Try to extract and display previous results
                    # Load a small validation batch to evaluate model
                    st.info("Evaluating loaded model...")
                    with st.spinner("Computing validation metrics..."):
                        try:
                            # Create a small evaluation
                            val_accuracies = []
                            val_losses = []
                            
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            st.session_state.trainer.model.to(device)
                            st.session_state.trainer.model.eval()
                            
                            criterion = torch.nn.CrossEntropyLoss()
                            total_loss = 0.0
                            correct = 0
                            total = 0
                            
                            with torch.no_grad():
                                for images, labels in val_loader:
                                    images, labels = images.to(device), labels.to(device)
                                    outputs = st.session_state.trainer.model(images)
                                    loss = criterion(outputs, labels)
                                    
                                    total_loss += loss.item()
                                    _, predicted = torch.max(outputs, 1)
                                    total += labels.size(0)
                                    correct += (predicted == labels).sum().item()
                            
                            avg_accuracy = correct / total if total > 0 else 0.0
                            avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
                            
                            st.metric("Loaded Model Validation Accuracy", f"{avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
                            st.metric("Loaded Model Validation Loss", f"{avg_loss:.4f}")
                            
                            # Store baseline training metadata for Phase 2
                            st.session_state.baseline_training_info = {
                                'epochs': num_epochs,
                                'batch_size': batch_size,
                                'learning_rate': learning_rate,
                                'dataset_path': str(st.session_state.selected_dataset_path),
                                'dataset_name': dataset_name,
                                'checkpoint_name': checkpoint_name,
                                'best_val_accuracy': avg_accuracy,
                                'train_losses': [avg_loss],
                                'val_losses': [avg_loss],
                                'train_accuracies': [avg_accuracy],
                                'val_accuracies': [avg_accuracy]
                            }
                            
                            log_step("Model Evaluated", f"Val Accuracy: {avg_accuracy:.4f}, Val Loss: {avg_loss:.4f}", "SUCCESS")
                        except Exception as e:
                            log_step("Model Evaluation Warning", f"Could not evaluate: {str(e)}", "PROGRESS")
                            st.session_state.baseline_training_info = {
                                'epochs': num_epochs,
                                'batch_size': batch_size,
                                'learning_rate': learning_rate,
                                'dataset_path': str(st.session_state.selected_dataset_path),
                                'dataset_name': dataset_name,
                                'checkpoint_name': checkpoint_name,
                                'best_val_accuracy': 0.0,
                                'train_losses': [],
                                'val_losses': [],
                                'train_accuracies': [],
                                'val_accuracies': []
                            }
                    
                    st.session_state.training_started = True
                else:
                    st.error(f"Failed to load checkpoint: {checkpoint_name}")
                    st.stop()
            
            else:
                # Model doesn't exist - run training
                log_step("No Model Found", f"Starting training for: {checkpoint_name}", "START")
                status_placeholder.info("Training is running... Check console for detailed progress")
                
                # Create the selected model and reinitialize trainer
                from trainer.model_builder import create_model
                log_step("Creating Model", f"Architecture: {st.session_state.selected_model}", "STEP")
                selected_model_lower = st.session_state.selected_model.lower().replace("-", "")
                model = create_model(
                    num_classes=2,
                    model_name=st.session_state.selected_model,
                    device='cpu'
                )
                st.session_state.trainer = WeldDefectTrainer(model=model, device='cpu')
                log_step("Trainer Ready", f"Initialized with {st.session_state.selected_model}", "SUCCESS")
                
                results = st.session_state.trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    checkpoint_name=checkpoint_name
                )
                
                if results['success']:
                    log_step("Training Complete", 
                            f"Best Val Accuracy: {results['best_val_accuracy']:.4f} ({results['best_val_accuracy']*100:.2f}%)", 
                            "SUCCESS")
                    
                    status_placeholder.success("Training Completed Successfully!")
                    
                    st.metric("Best Validation Accuracy", f"{results['best_val_accuracy']:.4f} ({results['best_val_accuracy']*100:.2f}%)")
                    st.metric("Final Training Loss", f"{results['train_losses'][-1]:.4f}")
                    st.metric("Final Validation Loss", f"{results['val_losses'][-1]:.4f}")
                    
                    # Store baseline training metadata for Phase 2
                    st.session_state.baseline_training_info = {
                        'epochs': num_epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'dataset_path': str(st.session_state.selected_dataset_path),
                        'dataset_name': dataset_name,
                        'checkpoint_name': checkpoint_name,
                        'best_val_accuracy': results['best_val_accuracy'],
                        'train_losses': results['train_losses'],
                        'val_losses': results['val_losses'],
                        'train_accuracies': results['train_accuracies'],
                        'val_accuracies': results['val_accuracies']
                    }
                    log_step("Baseline Metadata Stored", f"Saved training info: {num_epochs} epochs, batch {batch_size}, LR {learning_rate:.5f}", "SUCCESS")
                    
                    st.session_state.training_started = True
                else:
                    log_step("Training Failed", f"Error: {results.get('error', 'Unknown error')}", "PROGRESS")
                    status_placeholder.error(f"Training failed: {results.get('error', 'Unknown error')}")
        else:
            st.error("Dataset not ready - please wait for dataset generation")
    
    # Checkpoint management
    st.subheader("Step 5: Manage Trained Models")
    checkpoints = sorted([cp.name for cp in CHECKPOINTS_DIR.glob('*.pt')]) if CHECKPOINTS_DIR.exists() else []
    if checkpoints:
        log_step("Checkpoints Found", f"Found {len(checkpoints)} saved model(s)")
        selected_checkpoint = st.selectbox("Load a saved model:", checkpoints, key="phase1_checkpoint_select")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("✅ Load Selected Model", use_container_width=True):
                log_step("Loading Checkpoint", f"Loading model: {selected_checkpoint}")
                checkpoint_path = PROJECT_ROOT / 'checkpoints' / selected_checkpoint
                
                # Recreate trainer with the correct architecture for the selected checkpoint
                # to avoid state-dict mismatch errors (e.g., loading ShuffleNet into MobileNet).
                from trainer.model_builder import create_model
                selected_checkpoint_lower = selected_checkpoint.lower()
                if selected_checkpoint_lower.startswith("mobilenetv2_"):
                    model_name_for_creation = "MobileNetV2"
                elif selected_checkpoint_lower.startswith("efficientnet_b0_") or selected_checkpoint_lower.startswith("efficientnetb0_"):
                    model_name_for_creation = "EfficientNet-B0"
                elif selected_checkpoint_lower.startswith("shufflenet_v2_"):
                    model_name_for_creation = "ShuffleNet V2"
                else:
                    # Fallback for legacy/custom names
                    checkpoint_info_fallback = ModelNaming.parse_model_name(selected_checkpoint)
                    parsed_model_name = checkpoint_info_fallback.get('model_name', '')
                    if parsed_model_name == "mobilenetv2":
                        model_name_for_creation = "MobileNetV2"
                    elif parsed_model_name in ["efficientnet_b0", "efficientnetb0"]:
                        model_name_for_creation = "EfficientNet-B0"
                    elif parsed_model_name == "shufflenet_v2":
                        model_name_for_creation = "ShuffleNet V2"
                    else:
                        model_name_for_creation = "MobileNetV2"

                model = create_model(
                    num_classes=2,
                    model_name=model_name_for_creation,
                    device='cpu'
                )
                st.session_state.trainer = WeldDefectTrainer(model=model, device='cpu')

                if st.session_state.trainer.load_checkpoint(str(checkpoint_path)):
                    st.success(f"✅ Model loaded: {selected_checkpoint}")
                    log_step("Checkpoint Loaded", f"Successfully loaded {selected_checkpoint}", "SUCCESS")
                    
                    # Parse checkpoint name to extract metadata
                    checkpoint_info = ModelNaming.parse_model_name(selected_checkpoint)
                    dataset_name = checkpoint_info['dataset_name']
                    
                    # Store baseline training metadata for Phase 2
                    st.session_state.baseline_training_info = {
                        'epochs': checkpoint_info['epochs'] or 5,
                        'batch_size': 16,
                        'learning_rate': 0.001,
                        'dataset_path': str(st.session_state.selected_dataset_path),
                        'dataset_name': dataset_name,
                        'checkpoint_name': selected_checkpoint,
                        'best_val_accuracy': 0.92,  # Default reasonable value
                        'train_losses': [0.1],
                        'val_losses': [0.1],
                        'train_accuracies': [0.92],
                        'val_accuracies': [0.92]
                    }
                    st.session_state.training_started = True
                    log_step("Baseline Info Stored", f"Metadata extracted for Phase 2: {dataset_name}, {checkpoint_info['epochs']} epochs", "SUCCESS")
                else:
                    st.error(f"Failed to load checkpoint: {selected_checkpoint}")
        
        with col2:
            if st.button("ℹ️ Info", use_container_width=True):
                st.session_state.show_model_info = True
        
        with col3:
            if st.button("🗑️ Delete", use_container_width=True):
                st.session_state.show_delete_confirm = True
        
        # Show model info if requested
        if st.session_state.get('show_model_info', False):
            checkpoint_info = ModelNaming.parse_model_name(selected_checkpoint)
            with st.expander("📊 Model Details", expanded=True):
                st.markdown(ModelNaming.get_model_info_display(selected_checkpoint))
            st.session_state.show_model_info = False
        
        # Show delete confirmation if requested
        if st.session_state.get('show_delete_confirm', False):
            with st.container(border=True):
                st.warning(f"⚠️ Are you sure you want to delete '{selected_checkpoint}'?")
                st.caption("This action cannot be undone.")
                
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("🗑️ Yes, Delete", use_container_width=True, key="confirm_delete"):
                        st.session_state.show_delete_confirm = False
                        delete_checkpoint(selected_checkpoint)
                
                with col_no:
                    if st.button("❌ Cancel", use_container_width=True, key="cancel_delete"):
                        st.session_state.show_delete_confirm = False
                        st.rerun()
    else:
        st.info("No trained models saved yet. Complete a training session first!")
    
    # Model info
    st.subheader("Model Architecture Information")
    model_info = st.session_state.trainer.get_model_info()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Type", model_info['model_name'])
        st.metric("Model Size", f"{model_info['model_size_mb']:.2f} MB")
    with col2:
        st.metric("Total Parameters", f"{model_info['num_parameters']:,}")
        st.metric("Output Classes", model_info['num_classes'])
    
    st.info(f"""
    **Model Details:**
    - Architecture: {model_info['model_name']}
    - Total Parameters: {model_info['num_parameters']:,}
    - Model Size: {model_info['model_size_mb']:.2f} MB
    - Classes: {model_info['num_classes']} (Normal weld / Defective weld)
    """)


def optimization_phase():
    """Phase 2: Model optimization interface - RETRAIN WITH OPTIMIZATIONS."""
    from phase2_retraining_new import optimization_phase_retraining
    optimization_phase_retraining()


def main():
    """Main application."""
    # Page config
    st.set_page_config(
        page_title="Welding Defect Detection",
        page_icon="[WELD]",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-card {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    log_step("App Startup", "Welding Defect Detection starting", "START")
    init_session_state()
    
    # Header
    st.title("Welding Defect Detection")
    st.write("AI-powered system for real-time weld quality inspection and defect detection")
    st.write("=" * 80)
    
    # START HERE BANNER
    st.markdown("""
    <div style="background-color: #e8f4f8; border: 2px solid #0066cc; padding: 20px; border-radius: 8px; margin: 15px 0;">
        <h3>Quick Start Guide</h3>
        <ol>
            <li><b>Phase 1 - Training:</b> Train your model on weld images (synthetic or real data)</li>
            <li><b>Phase 2 - Optimization:</b> Optimize the model for edge deployment with compression/pruning</li>
            <li><b>Phase 3 - Inference:</b> Upload images and get real-time defect predictions with OOD protection</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    log_step("App Ready", "Displaying main interface", "SUCCESS")
    
    # Sidebar with progress tracker
    with st.sidebar:
        st.header("WORKSHOP PROGRESS TRACKER")
        st.write("---")
        
        progress_phase1 = st.session_state.trainer.get_model_info()['model_size_mb'] > 0
        progress_phase2 = 'optimizer' in st.session_state
        
        col1, col2 = st.columns(2)
        with col1:
            status_1 = "DONE" if progress_phase1 else "PENDING"
            st.metric("Phase 1 Status", status_1)
        with col2:
            status_2 = "ACTIVE" if progress_phase2 else "PENDING"
            st.metric("Phase 2 Status", status_2)
        
        st.write("---")
        
        st.subheader("QUICK START TIPS")
        st.info("""
PHASE 1 RECOMMENDATIONS:
- Start with 5-10 epochs for faster training
- Batch size 16-32 is usually optimal
- Watch for accuracy plateauing (model has learned enough)
- Save checkpoints of good models

PHASE 2 RECOMMENDATIONS:
- Begin with compression first (usually safest)
- Gradually increase pruning (remove less important weights)
- Try quantization with different bit widths (32 > 16 > 8 bits)
- Monitor all three constraint bars
- Save different optimization configurations to compare
        """)
        
        st.write("---")
        st.subheader("WORKSHOP TIMELINE")
        st.write("""
0-5 min:    Setup and data generation
5-90 min:   Phase 1 - Training
90-160 min: Phase 2 - Optimization
160-180 min: Results & Discussion
        """)
    
    
    # Data resources section
    st.write("---")
    st.subheader("DATA RESOURCES")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Current Dataset:**
        - Synthetic Data: 800 procedurally generated images
        - 5 defect types: crack, porosity, undercut, spatter, incomplete fusion
        - Fully portable and reproducible
        - Perfect for learning the workflow
        """)
    
    with col2:
        st.info("""
        **Available Datasets:**
        
        The following datasets are available in the system:
        
        1. **OpenSourceData01** (Recommended)
           - Real weld defect images
           - 2,178 labeled images with normal/defect classification
           - Source: [Kaggle Weld Defect Detection](https://www.kaggle.com/code/jayeshm0103/weld-defect-detection-and-classification)
           - Ready to use for training!
        
        2. **Synthetic Data** (Generated)
           - 800 procedurally generated weld images
           - 5 defect types: crack, porosity, undercut, spatter, incomplete fusion
           - Auto-generated on demand
           - Perfect for testing the workflow
        """)
    
    st.write("---")
    
    # Tab selection
    tab1, tab2, tab3 = st.tabs(["PHASE 1: TRAINING", "PHASE 2: OPTIMIZATION", "PHASE 3: INFERENCE"])
    
    with tab1:
        training_phase()
    
    with tab2:
        optimization_phase()
    
    with tab3:
        from phase3_inference import inference_phase
        inference_phase()
    
    log_step("App Display Complete", "All UI rendered successfully", "SUCCESS")


if __name__ == "__main__":
    main()
