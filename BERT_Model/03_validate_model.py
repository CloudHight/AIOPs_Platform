import boto3
import time
from sagemaker.tuner import HyperparameterTuner

def monitor_tuning_job(tuning_job_name):
    sm_client = boto3.client('sagemaker')
    print(f"[INFO] Monitoring tuning job: {tuning_job_name}")

    while True:
        response = sm_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)
        status = response['HyperParameterTuningJobStatus']
        counters = response['TrainingJobStatusCounters']
        print(f"[TUNER STATUS] {status} | Completed: {counters.get('Completed',0)}, InProgress: {counters.get('InProgress',0)}, Failed: {counters.get('Failed',0)}")

        best = response.get('BestTrainingJob', {})
        if best.get('TrainingJobName'):
            print(f"  [INFO] Current best: {best['TrainingJobName']}")
            metric = best.get('FinalHyperParameterTuningJobObjectiveMetric', {})
            if metric:
                print(f"      Best {metric.get('MetricName','metric')}: {metric.get('Value','?')}")
            print(f"      Tuned hyperparameters: {best.get('TunedHyperParameters',{})}")

        if status in ('Completed', 'Failed', 'Stopped'):
            print(f"[TUNER DONE] Final status: {status}")
            break

        time.sleep(60)

    if status == "Completed":
        print(f"[INFO] Best training job: {best.get('TrainingJobName')}")
        return best
    else:
        print("[ERROR] Tuning failed or was stopped.")
        return None

def get_best_model_artifacts(best_training_job_name):
    sm_client = boto3.client('sagemaker')
    response = sm_client.describe_training_job(TrainingJobName=best_training_job_name)
    model_artifacts = response['ModelArtifacts']['S3ModelArtifacts']
    print(f"[INFO] Best model artifacts: {model_artifacts}")
    return model_artifacts

if __name__ == "__main__":
    # Replace with your actual tuning job name from 02_train_model.py output
    TUNING_JOB_NAME = "YOUR_TUNING_JOB_NAME"
    
    best_job = monitor_tuning_job(TUNING_JOB_NAME)
    if best_job:
        model_artifacts = get_best_model_artifacts(best_job['TrainingJobName'])
        print(f"[INFO] Validation complete. Best model ready for testing.")