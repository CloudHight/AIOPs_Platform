#!/usr/bin/env python3
"""
ML Workflow Orchestrator
Executes the complete ML pipeline: create → train → validate → test → deploy → monitor
"""

from 01_create import create_dataset
from 02_train import train_model
from 03_validate import validate_model
from 04_test import test_model
from 05_deploy import deploy_model, cleanup_endpoint
from 06_monitor import analyze_anomalies, monitor_performance

def run_ml_workflow():
    """Execute the complete ML workflow."""
    print("Starting ML Workflow Pipeline...")
    
    # Stage 1: Create dataset
    df = create_dataset()
    
    # Stage 2: Train model
    rcf = train_model(df)
    if rcf is None:
        print("Training failed. Exiting workflow.")
        return
    
    # Stage 5: Deploy model (before testing)
    predictor = deploy_model(rcf)
    
    # Stage 4: Test model
    rcf_scores = test_model(df, predictor)
    
    # Stage 3: Validate model
    validation_results = validate_model(df, rcf_scores)
    
    # Stage 6: Monitor and analyze
    df_analyzed = analyze_anomalies(df, rcf_scores)
    monitor_performance(df_analyzed)
    
    # Cleanup
    cleanup_endpoint(predictor)
    
    print("ML Workflow completed successfully!")
    return df_analyzed, validation_results

if __name__ == "__main__":
    run_ml_workflow()