import optuna
from optuna.trial import TrialState
import tensorflow as tf

from training import objective


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
    # DEBUG
    # tf.config.run_functions_eagerly(True)
    # tf.data.experimental.enable_debug_mode()

    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("CUDA Version: ", tf.sysconfig.get_build_info()["cuda_version"])
    print("cuDNN Version: ", tf.sysconfig.get_build_info()["cudnn_version"])

    study = optuna.create_study(
        study_name="cnn_lstm_v1", storage="sqlite:///schizo_model.db", direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    study.optimize(objective, n_trials=25, timeout=600, gc_after_trial=True)
    show_result(study)
    show_result(study)


if __name__ == "__main__":
    main()
