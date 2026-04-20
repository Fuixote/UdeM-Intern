from stage1_training_regression import main


if __name__ == "__main__":
    main(
        default_model_family="lr",
        default_feature_mode="utility_cpra",
        description="Stage-1 linear regression training (utility + recipient cPRA)",
    )
