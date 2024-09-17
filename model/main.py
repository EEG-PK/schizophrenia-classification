import optuna
from optuna.trial import TrialState
import tensorflow as tf

from mockup_data import mockup
from params import DATASET, DATASETS_DIR
from training import objective, test_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("CUDA Version: ", tf.sysconfig.get_build_info()["cuda_version"])
print("cuDNN Version: ", tf.sysconfig.get_build_info()["cudnn_version"])

# TODO: Replace to proper data load
# Data mock-up for now
data_dir = 'data'
schizophrenia_files = ['33w1.eea', '088w1.eea', '103w.eea']
health_files = ['088w1.eea', '103w.eea']

mockup(data_dir, schizophrenia_files, health_files, dataset_dir=f"{DATASETS_DIR}/{DATASET}")


def show_result(study: optuna.Study) -> None:
    """
    Display the results of the Optuna study, including statistics about the trials and the best trial.

    :param study: The Optuna study containing the results of the optimization.
    """
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def main():
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=2)
    )
    study.optimize(objective, n_trials=25, timeout=600, gc_after_trial=True)
    show_result(study)

    # acc_avg = test_model()
    # print(acc_avg)


if __name__ == "__main__":
    main()
