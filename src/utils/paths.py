import os.path


def get_root_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def get_data_path():
    return os.path.join(get_root_path(), "data")

def get_results_path():
    return os.path.join(get_root_path(), "results")

def get_figures_path():
    return os.path.join(get_root_path(), "figures")


if __name__ == "__main__":

    print(get_root_path())
    print(get_data_path())
    print(get_results_path())
    print(get_figures_path())