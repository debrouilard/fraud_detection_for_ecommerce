#!/usr/bin/env python3
"""
Command-line interface for fraud detection system.
"""
import click
import pandas as pd
from pathlib import Path
import json

from src.data_loader import DataLoader
from src.feature_engineer import prepare_features
from src.preprocessor import preprocess_pipeline
from src.model_trainer import train_all_models, ModelTrainer
from src.model_evaluator import evaluate_model, ModelEvaluator
from src.predictor import load_predictor
from src.logger import setup_logger
from src.config import data_config, model_config

logger = setup_logger(__name__, "cli.log")


@click.group()
def cli():
    """Fraud Detection CLI - Train, evaluate, and predict fraud."""
    pass


@cli.command()
@click.option(
    '--dataset',
    type=click.Choice(['fraud', 'creditcard', 'both']),
    default='both',
    help='Dataset to process'
)
@click.option(
    '--force',
    is_flag=True,
    help='Force reprocessing even if processed data exists'
)
def preprocess(dataset, force):
    """Preprocess raw data and engineer features."""
    click.echo(f"🔄 Preprocessing {dataset} dataset(s)...")
    
    loader = DataLoader()
    datasets_to_process = ['fraud', 'creditcard'] if dataset == 'both' else [dataset]
    
    for ds_type in datasets_to_process:
        try:
            click.echo(f"\n{'='*60}")
            click.echo(f"Processing {ds_type.upper()} dataset")
            click.echo(f"{'='*60}")
            
            # Load raw data
            if ds_type == 'fraud':
                df = loader.load_fraud_data()
                drop_cols = data_config.fraud_drop_columns
            else:
                df = loader.load_creditcard_data()
                drop_cols = data_config.creditcard_drop_columns
            
            # Engineer features
            click.echo("Engineering features...")
            df_processed = prepare_features(
                df,
                dataset_type=ds_type,
                drop_columns=drop_cols + [data_config.target_column],
                fit=True
            )
            
            # Add target back
            df_processed[data_config.target_column] = df[data_config.target_column].values
            
            # Save processed data
            loader.save_processed_data(df_processed, ds_type)
            
            click.echo(f"✅ {ds_type.upper()} preprocessing complete!")
            click.echo(f"   Samples: {len(df_processed)}")
            click.echo(f"   Features: {len(df_processed.columns) - 1}")
            
        except Exception as e:
            click.echo(f"❌ Error processing {ds_type}: {str(e)}", err=True)
            logger.error(f"Preprocessing error for {ds_type}: {str(e)}")


@cli.command()
@click.option(
    '--dataset',
    type=click.Choice(['fraud', 'creditcard', 'both']),
    default='both',
    help='Dataset to train on'
)
@click.option(
    '--models',
    type=click.Choice(['logistic_regression', 'random_forest', 'xgboost', 'all']),
    default='all',
    help='Model types to train'
)
@click.option(
    '--no-smote',
    is_flag=True,
    help='Disable SMOTE oversampling'
)
def train(dataset, models, no_smote):
    """Train fraud detection models."""
    click.echo(f"🚀 Training models on {dataset} dataset(s)...")
    
    # Determine models to train
    if models == 'all':
        model_types = model_config.model_types
    else:
        model_types = [models]
    
    datasets_to_train = ['fraud', 'creditcard'] if dataset == 'both' else [dataset]
    
    for ds_type in datasets_to_train:
        try:
            click.echo(f"\n{'='*60}")
            click.echo(f"Training on {ds_type.upper()} dataset")
            click.echo(f"{'='*60}")
            
            # Load processed data
            from src.data_loader import load_train_test_split
            X_train, X_test, y_train, y_test = load_train_test_split(ds_type)
            
            click.echo(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            # Preprocess data
            click.echo("Preprocessing data...")
            X_train_proc, y_train_proc, X_test_proc = preprocess_pipeline(
                X_train, y_train, X_test,
                use_smote=not no_smote
            )
            
            # Train models
            click.echo(f"\nTraining {len(model_types)} model(s)...")
            trained_models = train_all_models(
                X_train_proc,
                y_train_proc,
                dataset_name=ds_type,
                model_types=model_types
            )
            
            # Evaluate each model
            click.echo("\nEvaluating models...")
            results = {}
            for model_type, trainer in trained_models.items():
                metrics = evaluate_model(trainer, X_test_proc, y_test)
                results[model_type] = metrics
                
                click.echo(f"\n{model_type.upper()}:")
                click.echo(f"  Accuracy:  {metrics['accuracy']:.4f}")
                click.echo(f"  Precision: {metrics['precision']:.4f}")
                click.echo(f"  Recall:    {metrics['recall']:.4f}")
                click.echo(f"  F1 Score:  {metrics['f1_score']:.4f}")
                click.echo(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
            
            # Save results
            results_file = model_config.model_save_dir / f"{ds_type}_results.json"
            with open(results_file, 'w') as f:
                # Convert metrics to serializable format
                serializable_results = {
                    model: {k: v for k, v in metrics.items() 
                           if k not in ['classification_report', 'confusion_matrix']}
                    for model, metrics in results.items()
                }
                json.dump(serializable_results, f, indent=2)
            
            click.echo(f"\n✅ Training complete for {ds_type}!")
            click.echo(f"   Results saved to: {results_file}")
            
        except Exception as e:
            click.echo(f"❌ Error training {ds_type}: {str(e)}", err=True)
            logger.error(f"Training error for {ds_type}: {str(e)}")


@cli.command()
@click.option(
    '--model-type',
    type=click.Choice(['logistic_regression', 'random_forest', 'xgboost']),
    required=True,
    help='Type of model to use'
)
@click.option(
    '--dataset',
    type=click.Choice(['fraud', 'creditcard']),
    required=True,
    help='Dataset type the model was trained on'
)
@click.option(
    '--input-file',
    type=click.Path(exists=True),
    required=True,
    help='CSV file with transactions to predict'
)
@click.option(
    '--output-file',
    type=click.Path(),
    help='Output file for predictions (default: input_predictions.csv)'
)
@click.option(
    '--threshold',
    type=float,
    default=0.5,
    help='Classification threshold'
)
def predict(model_type, dataset, input_file, output_file, threshold):
    """Make predictions on new data."""
    click.echo(f"🔮 Making predictions with {model_type} on {dataset} data...")
    
    try:
        # Load predictor
        predictor = load_predictor(model_type, dataset)
        click.echo(f"✓ Model loaded")
        
        # Load input data
        df = pd.read_csv(input_file)
        click.echo(f"✓ Loaded {len(df)} transactions")
        
        # Make predictions
        results = predictor.predict_batch(df, threshold=threshold, include_details=True)
        
        # Combine with original data
        output_df = pd.concat([df, results], axis=1)
        
        # Save results
        if output_file is None:
            output_file = Path(input_file).stem + '_predictions.csv'
        
        output_df.to_csv(output_file, index=False)
        
        # Summary
        fraud_count = results['is_fraud'].sum()
        fraud_pct = (fraud_count / len(results)) * 100
        
        click.echo(f"\n✅ Predictions complete!")
        click.echo(f"   Total transactions: {len(results)}")
        click.echo(f"   Fraud detected: {fraud_count} ({fraud_pct:.2f}%)")
        click.echo(f"   Results saved to: {output_file}")
        
        # Risk level summary
        if 'risk_level' in results.columns:
            risk_summary = results['risk_level'].value_counts()
            click.echo(f"\n   Risk Level Summary:")
            for level, count in risk_summary.items():
                click.echo(f"     {level}: {count}")
        
    except Exception as e:
        click.echo(f"❌ Error making predictions: {str(e)}", err=True)
        logger.error(f"Prediction error: {str(e)}")


@cli.command()
@click.option(
    '--dataset',
    type=click.Choice(['fraud', 'creditcard', 'both']),
    default='both',
    help='Dataset to evaluate'
)
def evaluate(dataset):
    """Evaluate trained models on test data."""
    click.echo(f"📊 Evaluating models on {dataset} dataset(s)...")
    
    datasets_to_eval = ['fraud', 'creditcard'] if dataset == 'both' else [dataset]
    
    for ds_type in datasets_to_eval:
        try:
            click.echo(f"\n{'='*60}")
            click.echo(f"Evaluating {ds_type.upper()} models")
            click.echo(f"{'='*60}")
            
            # Load data
            from src.data_loader import load_train_test_split
            _, X_test, _, y_test = load_train_test_split(ds_type)
            
            # Load and evaluate each model
            for model_type in model_config.model_types:
                try:
                    predictor = load_predictor(model_type, ds_type)
                    results = predictor.predict_batch(X_test, include_details=True)
                    
                    # Calculate metrics
                    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
                    
                    accuracy = accuracy_score(y_test, results['is_fraud'])
                    f1 = f1_score(y_test, results['is_fraud'])
                    roc_auc = roc_auc_score(y_test, results['fraud_probability'])
                    
                    click.echo(f"\n{model_type.upper()}:")
                    click.echo(f"  Accuracy: {accuracy:.4f}")
                    click.echo(f"  F1 Score: {f1:.4f}")
                    click.echo(f"  ROC AUC:  {roc_auc:.4f}")
                    
                except FileNotFoundError:
                    click.echo(f"  Model not found: {model_type}")
                    continue
            
        except Exception as e:
            click.echo(f"❌ Error evaluating {ds_type}: {str(e)}", err=True)
            logger.error(f"Evaluation error for {ds_type}: {str(e)}")


@cli.command()
def info():
    """Display system information and configuration."""
    click.echo("🛡️  Fraud Detection System Information\n")
    
    click.echo("Configuration:")
    click.echo(f"  Data directory: {data_config.fraud_data_path.parent}")
    click.echo(f"  Models directory: {model_config.model_save_dir}")
    click.echo(f"  Test split: {data_config.test_size}")
    click.echo(f"  Random seed: {data_config.random_state}")
    
    click.echo(f"\nAvailable models:")
    for model_type in model_config.model_types:
        click.echo(f"  - {model_type}")
    
    click.echo(f"\nSMOTE configuration:")
    click.echo(f"  Enabled: {model_config.use_smote}")
    click.echo(f"  Sampling strategy: {model_config.smote_sampling_strategy}")
    
    # Check for existing models
    click.echo(f"\nTrained models:")
    model_dir = model_config.model_save_dir
    if model_dir.exists():
        models = list(model_dir.glob("*.joblib"))
        if models:
            for model in models:
                click.echo(f"  ✓ {model.name}")
        else:
            click.echo("  No models found")
    else:
        click.echo("  Models directory not found")


if __name__ == '__main__':
    cli()
