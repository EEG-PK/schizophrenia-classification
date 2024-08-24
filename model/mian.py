import optuna
from optuna.trial import TrialState
from mockup_data import mockup
from training import objective

# Taki mockup danych na ten moment
data_dir = 'model/data'
schizophrenia_files = ['33w1.eea', '088w1.eea', '103w.eea']
health_files = ['088w1.eea', '103w.eea']

mockup(data_dir, schizophrenia_files, health_files)


def show_result(study):
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


if __name__ == "__main__":
    main()

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)
# print(f"Best trial: {study.best_trial.value}")
# print("Best hyperparameters: {}".format(study.best_trial.params))
