"""
Phase 3: Real-time Inference & Defect Detection

Upload weld images and get real-time predictions:
- Is it a weld image? (OOD detection)
- Does it have defects? (Classification)
- Confidence scores and uncertainty quantification
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CHECKPOINTS_DIR, training_config
from trainer.model_builder import create_model
from inference import InferenceEngine, InferencePrediction
from model_naming import ModelNaming


def log_step_local(step_name, description="", step_type="INFO"):
    """Log progress steps to console and GUI."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{timestamp}]"
    
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


def initialize_inference_engine(ood_threshold: float = 0.5):
    """Initialize the inference engine with loaded model.
    
    Args:
        ood_threshold: Threshold for OOD detection (0.0-1.0). Higher = stricter OOD detection.
    """
    if 'inference_engine' not in st.session_state or st.session_state.inference_engine is None:
        try:
            # Create model
            model = create_model(num_classes=2, device='cpu')
            
            # Initialize inference engine with configured threshold
            engine = InferenceEngine(model=model, device='cpu', ood_threshold=ood_threshold)
            
            st.session_state.inference_engine = engine
            st.session_state.engine_initialized = True
            return True
        except Exception as e:
            st.error(f"Failed to initialize inference engine: {e}")
            return False
    return True


def ood_threshold_from_sensitivity_pct(sensitivity_pct: float) -> float:
    """Convert weld sensitivity percentage to engine OOD threshold.

    sensitivity_pct means required weld-likeness confidence.
    Engine uses OOD score where higher means more OOD, so:
    OOD threshold = 1 - sensitivity.
    """
    bounded_pct = max(0.0, min(100.0, float(sensitivity_pct)))
    return 1.0 - (bounded_pct / 100.0)


def inference_phase():
    """Phase 3: Real-time inference with OOD detection."""
    log_step_local("PHASE 3 START", "User entered Phase 3 - Real-time Inference", "START")
    
    # Initialize session state thresholds (percent-based)
    if 'ood_sensitivity_pct' not in st.session_state:
        if 'ood_threshold' in st.session_state:
            old_ood = float(st.session_state.ood_threshold)
            st.session_state.ood_sensitivity_pct = old_ood * 100.0 if old_ood <= 1.0 else old_ood
        else:
            st.session_state.ood_sensitivity_pct = 50.0
    if 'defect_conf_threshold_pct' not in st.session_state:
        if 'defect_conf_threshold' in st.session_state:
            old_defect = float(st.session_state.defect_conf_threshold)
            st.session_state.defect_conf_threshold_pct = old_defect * 100.0 if old_defect <= 1.0 else old_defect
        else:
            st.session_state.defect_conf_threshold_pct = 50.0
    if 'processing_time_threshold_ms' not in st.session_state:
        st.session_state.processing_time_threshold_ms = 50.0
    
    # Reset button for Phase 3 (top right corner)
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🔄 Reset Phase 3", help="Reset inference state (Phase 1 & 2 state is retained)"):
            from app import reset_phase_3
            reset_phase_3()
    
    # Initialize engine with configured threshold
    initial_ood_threshold = ood_threshold_from_sensitivity_pct(st.session_state.ood_sensitivity_pct)
    if not initialize_inference_engine(ood_threshold=initial_ood_threshold):
        st.stop()
    
    # Check if Phase 1 training has been completed
    if not st.session_state.baseline_training_info:
        st.warning("⚠️ Tip: Complete Phase 1 training first to use your trained model in Phase 3")
    
    # NEXT ACTION BANNER
    st.markdown("""
    <div style="background-color: #f0f8e8; border-left: 5px solid #00cc00; padding: 15px; margin: 10px 0; border-radius: 5px;">
        <b>YOUR NEXT ACTION:</b><br>
        1. Load a trained model checkpoint<br>
        2. Upload one or more weld images<br>
        3. Get instant predictions with OOD detection<br>
        4. View detailed analysis: defects, confidence, OOD indicators<br>
        5. Process batch images for production use
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Phase 3: Real-time Inference & Defect Detection")
    
    st.write("""
    **Smart Defect Detection with Out-of-Distribution Protection**
    
    This phase uses your trained model to:
    1. **Detect Weld Images** - Identify if image is actually a weld (OOD detection)
    2. **Classify Defects** - Predict if weld has defects or is normal
    3. **Quantify Confidence** - Show certainty of predictions
    4. **Explain Decisions** - Why OOD was triggered if applicable
    
    **What is OOD (Out-of-Distribution) Detection?**
    - Even if you upload a picture of a cat, shoes, or random object
    - The model analyzes if it looks like a weld image
    - If not, it alerts you: "This doesn't appear to be a weld!"
    - Uses multiple heuristics: edge detection, texture, color, brightness
    """)
    
    # ==================== CHECKPOINT SELECTION ====================
    st.subheader("Step 1: Load Your Model Checkpoint")
    
    # Get available checkpoints
    checkpoints = []
    if CHECKPOINTS_DIR.exists():
        checkpoints = sorted([f.name for f in CHECKPOINTS_DIR.glob("*.pt")])
    
    if not checkpoints:
        st.warning("No checkpoints found. Complete Phase 1 training first.")
        # Use a default untrained model for demo
        st.info("Demo mode: Using untrained model for demonstration only")
        checkpoint_choice = "demo_mode"
    else:
        # Create display names using ModelNaming
        display_to_filename = {}
        display_names = []
        for checkpoint in checkpoints:
            display_name = ModelNaming.get_friendly_name(checkpoint)
            display_to_filename[display_name] = checkpoint
            display_names.append(display_name)
        
        # Selectbox with friendly names
        selected_display = st.selectbox(
            "Select a checkpoint to load:",
            options=display_names,
            index=len(display_names) - 1,  # Latest checkpoint
            key="phase3_checkpoint_select",
            help="Choose which trained model to use for inference"
        )
        
        checkpoint_choice = display_to_filename[selected_display]
        
        # Auto-load the selected checkpoint
        checkpoint_path = CHECKPOINTS_DIR / checkpoint_choice
        
        # Check if checkpoint file exists
        if not checkpoint_path.exists():
            st.error(f"❌ Checkpoint file not found: {checkpoint_path}")
            st.info(f"Expected location: {checkpoint_path}")
            st.stop()
        
        # Determine architecture from checkpoint filename and create correct model
        try:
            checkpoint_choice_lower = checkpoint_choice.lower()

            # Use robust filename-prefix detection first.
            # This avoids model-name parsing ambiguity when dataset names contain underscores.
            if checkpoint_choice_lower.startswith("mobilenetv2_"):
                model_name_for_creation = "MobileNetV2"
            elif checkpoint_choice_lower.startswith("efficientnet_b0_") or checkpoint_choice_lower.startswith("efficientnetb0_"):
                model_name_for_creation = "EfficientNet-B0"
            elif checkpoint_choice_lower.startswith("shufflenet_v2_"):
                model_name_for_creation = "ShuffleNet V2"
            else:
                # Fallback for legacy/custom filenames
                parsed_info = ModelNaming.parse_model_name(checkpoint_choice)
                model_name_from_checkpoint = parsed_info['model_name']
                if model_name_from_checkpoint == "mobilenetv2":
                    model_name_for_creation = "MobileNetV2"
                elif model_name_from_checkpoint in ["efficientnet_b0", "efficientnetb0"]:
                    model_name_for_creation = "EfficientNet-B0"
                elif model_name_from_checkpoint == "shufflenet_v2":
                    model_name_for_creation = "ShuffleNet V2"
                else:
                    model_name_for_creation = "MobileNetV2"  # Default fallback
            
            # Create model with correct architecture
            correct_model = create_model(
                num_classes=2,
                model_name=model_name_for_creation,
                device='cpu'
            )
            
            # Create new engine with correct model and configured threshold
            st.session_state.inference_engine = InferenceEngine(
                model=correct_model,
                device='cpu',
                ood_threshold=ood_threshold_from_sensitivity_pct(st.session_state.ood_sensitivity_pct)
            )
            
            # Load checkpoint into new engine
            checkpoint_loaded = st.session_state.inference_engine.load_checkpoint(checkpoint_path)
            
            if checkpoint_loaded:
                log_step_local("Model Loaded", f"Checkpoint: {checkpoint_choice} ({model_name_for_creation})", "SUCCESS")
            else:
                st.error(f"❌ Failed to load checkpoint: {checkpoint_choice}")
                st.info(f"File path: {checkpoint_path}")
                st.info(f"File exists: {checkpoint_path.exists()}")
                if checkpoint_path.exists():
                    st.info(f"File size: {checkpoint_path.stat().st_size} bytes")
                st.info("📋 Check the console/terminal for detailed error messages")
                st.stop()
        except Exception as e:
            st.error(f"❌ Error loading checkpoint: {str(e)}")
            import traceback
            st.error("📋 Exception Details:")
            st.code(traceback.format_exc())
            st.stop()
    
    if checkpoint_choice == "demo_mode":
        st.info("Using demo (untrained) model for visualization")
    
    # ==================== INFERENCE SETTINGS ====================
    st.divider()
    st.subheader("Step 2: Configure Detection Sensitivity")

    col1, col2, col3 = st.columns(3)

    with col1:
        new_ood_sensitivity_pct = st.slider(
            "OOD Detection Sensitivity",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.ood_sensitivity_pct),
            step=1.0,
            format="%.0f%%",
            help="""Required weld-likeness confidence to accept an image as weld.
            - 100%: strictest (images below this are marked OOD)
            - Lower values: more permissive OOD filtering
            """
        )

    with col2:
        new_defect_conf_threshold_pct = st.slider(
            "Defect Classification",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.defect_conf_threshold_pct),
            step=1.0,
            format="%.0f%%",
            help="""Confidence threshold for labeling an image as DEFECT.
            - 100%: strictest defect decision threshold
            - Lower values: more sensitive to possible defects
            """
        )

    with col3:
        new_processing_time_threshold_ms = st.slider(
            "Processing Time",
            min_value=10.0,
            max_value=2000.0,
            value=float(st.session_state.processing_time_threshold_ms),
            step=10.0,
            help="""Alert threshold for inference latency in milliseconds.
            - Lower values: Stricter latency target
            - Higher values: More tolerant latency target
            """
        )

    # Update OOD threshold on the active engine (without replacing loaded checkpoint model)
    if new_ood_sensitivity_pct != st.session_state.ood_sensitivity_pct:
        st.session_state.ood_sensitivity_pct = new_ood_sensitivity_pct
        mapped_ood_threshold = ood_threshold_from_sensitivity_pct(new_ood_sensitivity_pct)
        if st.session_state.inference_engine is not None:
            st.session_state.inference_engine.ood_detector.threshold = mapped_ood_threshold

    # Update UI-level thresholds used for interpretation/alerts
    st.session_state.defect_conf_threshold_pct = new_defect_conf_threshold_pct
    st.session_state.processing_time_threshold_ms = new_processing_time_threshold_ms

    st.caption(
        f"Active thresholds: OOD={st.session_state.ood_sensitivity_pct:.0f}%, "
        f"Defect={st.session_state.defect_conf_threshold_pct:.0f}%, "
        f"Latency={st.session_state.processing_time_threshold_ms:.0f} ms"
    )
    
    # ==================== IMAGE UPLOAD ====================
    st.divider()
    st.subheader("Step 3: Upload Weld Images")
    
    upload_type = st.radio(
        "Choose upload method:",
        ["Single Image", "Multiple Images (Batch)"],
        horizontal=True
    )
    
    uploaded_files = st.file_uploader(
        "Upload weld images" if upload_type == "Single Image" else "Upload multiple weld images",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=(upload_type == "Multiple Images (Batch)"),
        key="image_uploader"
    )
    
    if not uploaded_files:
        st.info("📤 Upload images to get predictions")
        return
    
    # Convert to list for consistent handling
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
    
    log_step_local("Images Uploaded", f"Processing {len(uploaded_files)} image(s)", "PROGRESS")
    st.divider()
    st.subheader("Step 4: Run Inference")
    
    if st.button("🔍 Analyze Images", use_container_width=True, key="analyze_button"):
        log_step_local("Inference Starting", f"Processing {len(uploaded_files)} images", "START")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        predictions = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            # Save temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            # Predict
            status_text.text(f"Analyzing image {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
            pred = st.session_state.inference_engine.predict_single(tmp_path)
            predictions.append(pred)
            
            # Update progress
            progress = (idx + 1) / len(uploaded_files)
            progress_bar.progress(progress)
        
        progress_bar.empty()
        status_text.empty()
        
        log_step_local("Inference Complete", f"Processed {len(predictions)} images", "SUCCESS")
        
        # ==================== RESULTS DISPLAY ====================
        st.divider()
        st.subheader("Step 5: Prediction Results")

        ood_sensitivity_pct = st.session_state.get('ood_sensitivity_pct', 50.0)
        defect_threshold_pct = st.session_state.get('defect_conf_threshold_pct', 50.0)
        st.caption(
            f"Rule: Accept as weld if weld confidence >= {ood_sensitivity_pct:.0f}%. "
            f"If accepted as weld, classify DEFECT when defect probability >= {defect_threshold_pct:.0f}%."
        )
        
        # Single image results
        if len(predictions) == 1:
            pred = predictions[0]
            display_single_prediction(pred, uploaded_files[0])
        
        # Multiple image results
        else:
            display_batch_predictions(predictions, uploaded_files)


def display_single_prediction(pred: InferencePrediction, uploaded_file):
    """Display detailed results for a single image."""
    defect_threshold_pct = st.session_state.get('defect_conf_threshold_pct', 50.0)
    processing_time_threshold = st.session_state.get('processing_time_threshold_ms', 250.0)
    ood_sensitivity_pct = float(st.session_state.get('ood_sensitivity_pct', 50.0))
    weld_confidence_pct = round((1.0 - pred.ood_score) * 100.0, 2)
    defect_probability_pct = round(pred.defect_prob * 100.0, 2)
    is_weld_by_threshold = weld_confidence_pct >= float(ood_sensitivity_pct)
    is_defect_by_threshold = defect_probability_pct >= float(defect_threshold_pct)
    ood_score_display_pct = (100.0 - weld_confidence_pct) if is_weld_by_threshold else weld_confidence_pct
    
    # Display image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
    
    with col2:
        st.write("### Prediction Results")
        
        # OOD Status (most important)
        if is_weld_by_threshold:
            st.success(f"✅ **Weld Image Detected**")
            st.write(
                f"Weld confidence: {weld_confidence_pct:.2f}% "
                f"(threshold: {ood_sensitivity_pct:.0f}%)"
            )
        else:
            st.error(f"❌ **Not a Weld Image (OOD)**")
            st.write(f"OOD Score: {ood_score_display_pct:.2f}%")
            st.write(
                f"Weld confidence: {weld_confidence_pct:.2f}% "
                f"(below threshold: {ood_sensitivity_pct:.0f}%)"
            )
            st.write(f"**Reason:** {pred.ood_reason}")
        
        st.divider()
        
        # Classification (only if it's a weld)
        if is_weld_by_threshold:
            st.write("### Defect Classification")
            
            if is_defect_by_threshold:
                st.error(f"🚨 **DEFECT DETECTED**")
                st.metric("Classification", "DEFECT")
            else:
                st.success(f"✅ **NORMAL (No Defect)**")
                st.metric("Classification", "NORMAL")

            st.write(
                f"Defect probability: {defect_probability_pct:.2f}% "
                f"(threshold: {defect_threshold_pct:.0f}%)"
            )
        else:
            st.warning("⚠️ Cannot classify - image does not appear to be a weld")

        st.divider()
        st.write("### Processing Time Assessment")

        processing_time_ms = pred.processing_time_ms
        threshold_ms = float(processing_time_threshold)

        if processing_time_ms <= threshold_ms:
            st.success(
                f"✅ Status: NORMAL | Processing time: {processing_time_ms:.1f}ms "
                f"(threshold: {threshold_ms:.1f}ms)"
            )
        else:
            breach_pct = ((processing_time_ms - threshold_ms) / threshold_ms) * 100.0

            if breach_pct <= 10.0:
                st.warning(
                    f"⚠️ Status: WARNING | Processing time: {processing_time_ms:.1f}ms "
                    f"(threshold: {threshold_ms:.1f}ms, breach: +{breach_pct:.1f}%)"
                )
            elif breach_pct <= 25.0:
                st.error(
                    f"🚨 Status: CRITICAL | Processing time: {processing_time_ms:.1f}ms "
                    f"(threshold: {threshold_ms:.1f}ms, breach: +{breach_pct:.1f}%)"
                )
            else:
                st.error(
                    f"❌ Status: FAILED | Processing time: {processing_time_ms:.1f}ms "
                    f"(threshold: {threshold_ms:.1f}ms, breach: +{breach_pct:.1f}%)"
                )
    
def display_batch_predictions(predictions: list, uploaded_files):
    """Display results for batch of images."""
    defect_threshold_pct = st.session_state.get('defect_conf_threshold_pct', 50.0)
    processing_time_threshold = st.session_state.get('processing_time_threshold_ms', 250.0)
    ood_sensitivity_pct = float(st.session_state.get('ood_sensitivity_pct', 50.0))
    
    st.write(f"### Batch Analysis: {len(predictions)} Images")

    per_image_results = []
    for pred, uploaded_file in zip(predictions, uploaded_files):
        weld_confidence_pct = round((1.0 - pred.ood_score) * 100.0, 2)
        defect_probability_pct = round(pred.defect_prob * 100.0, 2)
        is_weld_by_threshold = weld_confidence_pct >= float(ood_sensitivity_pct)
        is_defect_by_threshold = defect_probability_pct >= float(defect_threshold_pct)

        if not is_weld_by_threshold:
            classification_label = 'N/A'
        else:
            classification_label = 'DEFECT' if is_defect_by_threshold else 'NORMAL'

        per_image_results.append({
            'uploaded_file': uploaded_file,
            'pred': pred,
            'weld_confidence_pct': weld_confidence_pct,
            'defect_probability_pct': defect_probability_pct,
            'is_weld': is_weld_by_threshold,
            'classification': classification_label,
        })
    
    # Summary statistics
    defect_count = sum(1 for r in per_image_results if r['is_weld'] and r['classification'] == 'DEFECT')
    normal_count = sum(1 for r in per_image_results if r['is_weld'] and r['classification'] == 'NORMAL')
    ood_count = sum(1 for r in per_image_results if not r['is_weld'])
    slow_count = sum(1 for r in per_image_results if r['pred'].processing_time_ms > processing_time_threshold)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Images", len(predictions))
    with col2:
        st.metric("Defects Found", defect_count)
    with col3:
        st.metric("Normal Welds", normal_count)
    with col4:
        st.metric("Non-Weld Images", ood_count)
    with col5:
        st.metric("Slow Images", slow_count)
    
    # Detailed results table
    st.divider()
    st.write("### Detailed Results")

    weld_col = f"Weld Confidence (threshold: {ood_sensitivity_pct:.0f}%)"
    defect_col = f"Defect Probability (threshold: {defect_threshold_pct:.0f}%)"
    latency_col = f"Latency (threshold: {processing_time_threshold:.1f}ms)"

    def _latency_status_with_time(processing_time_ms: float, threshold_ms: float) -> str:
        if processing_time_ms <= threshold_ms:
            return f"Normal ({processing_time_ms:.1f}ms)"
        breach_pct = ((processing_time_ms - threshold_ms) / threshold_ms) * 100.0
        if breach_pct <= 10.0:
            return f"Warning ({processing_time_ms:.1f}ms)"
        return f"Critical Failure ({processing_time_ms:.1f}ms)"
    
    results_data = []
    for row in per_image_results:
        pred = row['pred']
        uploaded_file = row['uploaded_file']
        weld_confidence_pct = row['weld_confidence_pct']
        defect_probability_pct = row['defect_probability_pct']
        classification_label = row['classification']
        is_weld_by_threshold = row['is_weld']
        latency_status = _latency_status_with_time(pred.processing_time_ms, float(processing_time_threshold))
        results_data.append({
            'Image': uploaded_file.name,
            'Is Weld?': '✅ Yes' if is_weld_by_threshold else '❌ No',
            weld_col: f"{weld_confidence_pct:.2f}%",
            'Classification': classification_label,
            defect_col: f"{defect_probability_pct:.2f}%" if is_weld_by_threshold else 'N/A',
            latency_col: latency_status,
        })

    df = pd.DataFrame(results_data)

    def _style_result_columns(row):
        styles = [''] * len(row)

        if row.get('Is Weld?') == '✅ Yes':
            styles[df.columns.get_loc('Is Weld?')] = 'background-color: #d6f5d6; color: #0f5132; font-weight: 600;'
        else:
            styles[df.columns.get_loc('Is Weld?')] = 'background-color: #f8d7da; color: #842029; font-weight: 600;'

        classification_value = row.get('Classification')
        if classification_value == 'NORMAL':
            styles[df.columns.get_loc('Classification')] = 'background-color: #d6f5d6; color: #0f5132; font-weight: 600;'
        elif classification_value == 'DEFECT':
            styles[df.columns.get_loc('Classification')] = 'background-color: #f8d7da; color: #842029; font-weight: 600;'
        else:
            styles[df.columns.get_loc('Classification')] = 'background-color: #e9ecef; color: #495057;'

        latency_value = str(row.get(latency_col, ''))
        if latency_value.startswith('Normal'):
            styles[df.columns.get_loc(latency_col)] = 'background-color: #d6f5d6; color: #0f5132; font-weight: 600;'
        elif latency_value.startswith('Warning'):
            styles[df.columns.get_loc(latency_col)] = 'background-color: #fff3cd; color: #664d03; font-weight: 600;'
        else:
            styles[df.columns.get_loc(latency_col)] = 'background-color: #f8d7da; color: #842029; font-weight: 600;'

        return styles

    styled_df = df.style.apply(_style_result_columns, axis=1)

    st.markdown(
        """
        <style>
        div[data-testid="stDataFrame"] th {
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            line-height: 1.2 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(styled_df, use_container_width=True)
    
    # Show individual predictions
    st.divider()
    st.write("### Individual Image Analysis")
    
    for i, row in enumerate(per_image_results):
        pred = row['pred']
        uploaded_file = row['uploaded_file']
        is_weld_by_threshold = row['is_weld']
        classification_label = row['classification']
        defect_probability_pct = row['defect_probability_pct']
        with st.expander(f"📊 Image {i+1}: {uploaded_file.name}", expanded=(i==0)):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image)
            
            with col2:
                if is_weld_by_threshold:
                    st.success(f"✅ Weld Image")
                    if classification_label == 'DEFECT':
                        st.error(f"🚨 **DEFECT** ({defect_probability_pct:.2f}% defect probability)")
                    else:
                        st.success(f"✅ **NORMAL** ({defect_probability_pct:.2f}% defect probability)")
                else:
                    st.error(f"❌ Not a Weld Image")
                    st.write(f"Reason: {pred.ood_reason}")

                if pred.processing_time_ms > processing_time_threshold:
                    st.warning(
                        f"⚠️ Slow inference: {pred.processing_time_ms:.1f}ms "
                        f"(target: {processing_time_threshold:.1f}ms)"
                    )


if __name__ == "__main__":
    # Initialize session state
    if 'baseline_training_info' not in st.session_state:
        st.session_state.baseline_training_info = None
    if 'inference_engine' not in st.session_state:
        st.session_state.inference_engine = None
    if 'engine_initialized' not in st.session_state:
        st.session_state.engine_initialized = False
    
    inference_phase()
