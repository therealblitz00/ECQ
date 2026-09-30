"""Example pre-processing idea, used as the template everyone copies."""
import numpy as np

from testing_pipeline import run_experiment


def johns_preprocessing_idea(X_train, Y_train, X_val):
    """
    Hypothesis: Removing a highly noisy CRM column (index 10) and dropping
    the first 1000 rows (suspected outdated legacy data) improves the model.
    """
    # Create copies to avoid mutating the original arrays
    X_tr_new = np.copy(X_train)
    Y_tr_new = np.copy(Y_train)
    X_va_new = np.copy(X_val)

    # Feature Selection / Engineering:
    # Set the noisy feature at index 10 to 0.
    # RULE: If you drop/mask a feature in Training, you MUST do it in Validation too!
    X_tr_new[:, 10] = 0
    X_va_new[:, 10] = 0

    # Row Filtering (Outlier Removal):
    # Drop the first 1000 rows from the training set.
    # RULE: NEVER drop rows from the validation set!
    X_tr_new = X_tr_new[1000:]
    Y_tr_new = Y_tr_new[1000:]

    return X_tr_new, Y_tr_new, X_va_new


if __name__ == "__main__":
    run_experiment(
        author="John",
        idea_name="Drop_Feature10_and_Legacy_Rows",
        preprocess_func=johns_preprocessing_idea,
    )
