"""
Dashboard components for optimization phase.

Displays KPI cards, constraint progress, and deployment status.
"""

import streamlit as st
from simulator.metrics import ModelMetrics
from simulator.scorer import DeploymentScore
from simulator.profiles import DeploymentProfile


def display_kpi_cards(metrics: ModelMetrics, baseline_metrics: ModelMetrics):
    """
    Display KPI cards for model metrics.
    
    Args:
        metrics: Current model metrics
        baseline_metrics: Baseline (unoptimized) metrics
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy_pct = metrics.accuracy * 100
        delta = (metrics.accuracy - baseline_metrics.accuracy) * 100
        st.metric(
            "Accuracy",
            f"{accuracy_pct:.1f}%",
            delta=f"{delta:.1f}%",
            delta_color="off" if delta <= 0 else "normal"
        )
    
    with col2:
        size_reduction = (1 - metrics.model_size_mb / baseline_metrics.model_size_mb) * 100
        st.metric(
            "Model Size",
            f"{metrics.model_size_mb:.1f} MB",
            delta=f"-{size_reduction:.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        speedup = baseline_metrics.latency_ms / metrics.latency_ms
        st.metric(
            "Latency",
            f"{metrics.latency_ms:.1f} ms",
            delta=f"{speedup:.1f}x faster",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Throughput",
            f"{metrics.throughput_fps:.1f} FPS"
        )


def display_sparsity_metrics(metrics: ModelMetrics):
    """
    Display sparsity and compression metrics.
    
    Args:
        metrics: Current model metrics
    """
    col1, col2 = st.columns(2)
    
    with col1:
        sparsity_pct = metrics.sparsity * 100
        st.metric("Sparsity", f"{sparsity_pct:.1f}%", help="Fraction of zero weights")
    
    with col2:
        st.metric(
            "Compression Ratio",
            f"{metrics.compression_ratio:.2f}x",
            help="Original size / Compressed size"
        )


def display_constraint_status(score: DeploymentScore, profile: DeploymentProfile):
    """
    Display constraint compliance status.
    
    Args:
        score: Deployment score
        profile: Deployment profile
    """
    st.subheader("📋 Constraint Status")
    
    if score.is_deployable:
        st.success("✅ **All constraints met - Ready for deployment!**", icon="✅")
    else:
        st.error("❌ **Some constraints not met - Optimization needed**", icon="❌")
    
    # Display violations
    if score.violations:
        st.warning("**Issues to resolve:**")
        for violation in score.violations:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{violation.constraint_name}**")
                st.caption(violation.recommendation)
            with col2:
                st.metric("Current", f"{violation.current_value:.1f} {violation.unit}")
            with col3:
                st.metric("Required", f"{violation.required_value:.1f} {violation.unit}")
    
    # Display warnings
    if score.warnings:
        st.info("**Soft constraint warnings:**")
        for warning in score.warnings:
            st.write(f"• {warning}")


def display_deployment_gauge(score: DeploymentScore):
    """
    Display overall deployment score gauge.
    
    Args:
        score: Deployment score
    """
    st.subheader("🎯 Deployment Readiness Score")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Color based on score
        if score.total_score >= 80:
            color = "🟢"
            status = "Excellent"
        elif score.total_score >= 60:
            color = "🟡"
            status = "Good"
        elif score.total_score >= 40:
            color = "🟠"
            status = "Fair"
        else:
            color = "🔴"
            status = "Poor"
        
        st.write(f"## {color} {score.total_score:.1f}/100 - {status}")
    
    with col2:
        st.metric("Accuracy", f"{score.accuracy_score:.0f}/100")
    
    with col3:
        st.metric("Size", f"{score.size_score:.0f}/100")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Latency", f"{score.latency_score:.0f}/100")
    
    with col2:
        st.metric("Throughput", f"{score.throughput_score:.0f}/100")
    
    # Display recommendations
    if score.recommendations:
        st.info("**Recommendations:**")
        for rec in score.recommendations:
            st.write(f"• {rec}")


def display_constraint_progress(progress: dict):
    """
    Display progress bars for each constraint.
    
    Args:
        progress: Progress dictionary from scorer
    """
    st.subheader("📈 Constraint Progress")
    
    col1, col2, col3 = st.columns(3)
    
    # Accuracy
    with col1:
        accuracy = progress['accuracy']
        st.write("**Accuracy**")
        
        color = "🟢" if accuracy['met'] else "🔴"
        st.write(f"{color} {accuracy['current']:.1%} / {accuracy['required']:.1%}")
        
        st.progress(min(100, accuracy['progress_pct']) / 100)
    
    # Size
    with col2:
        size = progress['size']
        st.write("**Model Size**")
        
        color = "🟢" if size['met'] else "🔴"
        st.write(f"{color} {size['current']:.1f}MB / {size['required']:.1f}MB")
        
        st.progress(size['progress_pct'] / 100)
    
    # Latency
    with col3:
        latency = progress['latency']
        st.write("**Latency**")
        
        color = "🟢" if latency['met'] else "🔴"
        st.write(f"{color} {latency['current']:.0f}ms / {latency['required']:.0f}ms")
        
        st.progress(latency['progress_pct'] / 100)


def display_deployment_profile_summary(profile: DeploymentProfile):
    """
    Display deployment profile information.
    
    Args:
        profile: Deployment profile
    """
    st.subheader(profile.name)
    st.write(profile.description)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Use Case**")
        st.caption(profile.use_case)
        
        st.write("**Hardware**")
        st.caption(profile.hardware_description)
    
    with col2:
        st.write("**Hard Constraints**")
        st.caption(f"Size: ≤ {profile.max_model_size_mb:.0f}MB")
        st.caption(f"Latency: ≤ {profile.max_latency_ms:.0f}ms")
        st.caption(f"Accuracy: ≥ {profile.min_accuracy:.1%}")
        
        st.write("**Preferred**")
        st.caption(f"Accuracy: {profile.preferred_accuracy:.1%}")
        st.caption(f"Throughput: {profile.preferred_throughput_fps:.1f} FPS")


def display_tradeoff_scenarios():
    """
    Display guided trade-off scenarios to help students understand accuracy vs latency/compression trade-offs.
    Shows example configurations that achieve different accuracy levels with optimized parameters.
    """
    st.subheader("Step 2A: Understand Accuracy vs Performance Trade-offs")
    
    st.write("""
    Before manually adjusting parameters, explore how accuracy compromises affect model performance.
    Click on a scenario below to see what optimization parameters achieve that balance.
    """)
    
    # Define trade-off scenarios: (accuracy, compression, pruning, quantization_bits, description)
    scenarios = [
        (1.0, 0.0, 0.0, 32, "[REFERENCE] Baseline Model", "color: #4CAF50"),
        (0.95, 0.15, 0.05, 32, "[BALANCED] 95% Accuracy", "color: #2196F3"),
        (0.90, 0.35, 0.15, 16, "[OPTIMIZED] 90% Accuracy", "color: #FF9800"),
        (0.85, 0.50, 0.30, 16, "[AGGRESSIVE] 85% Accuracy", "color: #F44336"),
        (0.80, 0.60, 0.40, 8, "[EXTREME] 80% Accuracy", "color: #9C27B0"),
    ]
    
    from simulator.metrics import metrics_calculator
    
    cols = st.columns(len(scenarios))
    for idx, (accuracy, compression, pruning, quantization, label, color_style) in enumerate(scenarios):
        with cols[idx]:
            # Calculate metrics for this scenario
            metrics = metrics_calculator.calculate_metrics(
                compression=compression,
                pruning=pruning,
                quantization_bits=quantization,
                target_accuracy=accuracy
            )
            
            baseline = metrics_calculator.baseline
            speedup = baseline.latency_ms / metrics.latency_ms
            size_reduction = (1 - metrics.model_size_mb / baseline.model_size_mb) * 100
            
            st.markdown(f"<div style='border: 2px solid; border-radius: 8px; padding: 12px; {color_style}; opacity: 0.9'>", unsafe_allow_html=True)
            st.markdown(f"**{label}**", unsafe_allow_html=True)
            st.caption(f"Acc: {accuracy*100:.0f}% | Size: -{size_reduction:.0f}% | Speed: {speedup:.1f}x")
            st.caption(f"Comp: {compression*100:.0f}% | Prune: {pruning*100:.0f}% | Quant: {quantization}b")
            
            if st.button(f"Load Scenario {idx+1}", key=f"scenario_{idx}", use_container_width=True):
                st.session_state.preset_accuracy = accuracy
                st.session_state.preset_compression = compression
                st.session_state.preset_pruning = pruning
                st.session_state.preset_quantization = quantization
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    if any(key in st.session_state for key in ['preset_accuracy', 'preset_compression', 'preset_pruning', 'preset_quantization']):
        st.success("Scenario loaded! Use the manual controls below to fine-tune.", icon="✅")


def display_optimization_controls() -> dict:
    """
    Display optimization parameter controls.
    Supports preset values loaded from scenario selection.
    
    Returns:
        Dictionary with compression, pruning, quantization, and target_accuracy values
    """
    st.subheader("Step 2B: Manual Fine-tuning")
    
    # Show if preset is loaded
    is_preset_loaded = any(key in st.session_state for key in ['preset_accuracy', 'preset_compression', 'preset_pruning', 'preset_quantization'])
    
    if is_preset_loaded:
        col_info, col_clear = st.columns([0.8, 0.2])
        with col_info:
            st.info("Scenario loaded! Adjust parameters below to fine-tune.", icon="✅")
        with col_clear:
            if st.button("Start Fresh", key="clear_preset", use_container_width=True, help="Clear scenario and start from baseline"):
                for key in ['preset_accuracy', 'preset_compression', 'preset_pruning', 'preset_quantization']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    else:
        st.write("Adjust parameters below. All changes update metrics in real-time.")
    
    # Get preset values if loaded from scenario
    default_compression = st.session_state.get('preset_compression', 0.0)
    default_pruning = st.session_state.get('preset_pruning', 0.0)
    default_quantization = st.session_state.get('preset_quantization', 32)
    default_accuracy = st.session_state.get('preset_accuracy', 0.95)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        compression = st.slider(
            "Compression",
            min_value=0.0,
            max_value=1.0,
            value=default_compression,
            step=0.05,
            help="Reduce model complexity (0-100%)"
        )
        st.caption(f"{compression*100:.0f}% compression")
    
    with col2:
        pruning = st.slider(
            "Pruning",
            min_value=0.0,
            max_value=0.95,
            value=default_pruning,
            step=0.05,
            help="Remove unnecessary connections (0-95%)"
        )
        st.caption(f"{pruning*100:.0f}% pruning")
    
    with col3:
        quantization_bits = st.selectbox(
            "Quantization",
            options=[32, 16, 8, 4],
            format_func=lambda x: f"{x}-bit",
            index=[32, 16, 8, 4].index(int(default_quantization)),
            help="Reduce bits per weight"
        )
        size_reduction = (1 - quantization_bits / 32) * 100
        st.caption(f"~{size_reduction:.0f}% size reduction")
    
    with col4:
        target_accuracy = st.slider(
            "Target Accuracy",
            min_value=0.50,
            max_value=1.0,
            value=default_accuracy,
            step=0.01,
            format="%.2f",
            help="Minimum accuracy to maintain (50% - 100%)"
        )
        st.caption(f"{target_accuracy*100:.1f}% target")
    
    return {
        'compression': compression,
        'pruning': pruning,
        'quantization_bits': quantization_bits,
        'target_accuracy': target_accuracy
    }


def format_model_name_with_params(compression: float, pruning: float, quantization_bits: int, target_accuracy: float) -> str:
    """
    Format a model checkpoint name that includes optimization parameters.
    
    Example: model_acc95_comp30_prune15_q16.pt
    
    Args:
        compression: Compression ratio (0-1)
        pruning: Pruning ratio (0-1)
        quantization_bits: Quantization bits (4, 8, 16, 32)
        target_accuracy: Target accuracy (0.5-1.0)
    
    Returns:
        Formatted model checkpoint filename
    """
    acc = int(target_accuracy * 100)
    comp = int(compression * 100)
    prune = int(pruning * 100)
    
    return f"model_acc{acc}_comp{comp}_prune{prune}_q{quantization_bits}.pt"


def display_optimization_summary(summary: dict):
    """
    Display summary of optimizations applied.
    
    Args:
        summary: Summary dictionary from optimizer
    """
    st.subheader("📊 Optimization Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Accuracy Loss",
            f"{summary['accuracy_loss_pct']:.2f}%",
            help="Percentage points lost compared to baseline"
        )
    
    with col2:
        st.metric(
            "Size Reduction",
            f"{summary['size_reduction_pct']:.1f}%",
            help="Percentage reduction in model size"
        )
    
    with col3:
        st.metric(
            "Speedup",
            f"{summary['speedup']:.1f}x",
            help="Inference speedup compared to baseline"
        )
