import os
from multiprocessing.dummy import Pool

# TODO: Write parallel type of algorithm

def convert_files(dir, in_ext, action, recursive=True, exclude_ext=None):
    """ Traverse directory recursively to convert files.
        If recursive==False, only files in the directory dir are converted.
    """
    files = sorted(os.listdir(dir))
    for file in files:
        path = os.path.join(dir, file)
        if os.path.isdir(path):
            if recursive:
                convert_files(path, in_ext, action, recursive)
                pass
            pass
        elif path.endswith(in_ext):
            if exclude_ext is None or not path.endswith(exclude_ext):
                action(path)
                pass
            pass
        pass
    pass


def convert_files_parallel(dir, in_ext, action, recursive=True, exclude_ext=None, num_process=10):
    """ Traverse directory recursively to convert files.
        If recursive==False, only files in the directory dir are converted.
    """
    files = sorted(os.listdir(dir))
    path_to_perform_action_list = []
    for file in files:
        path = os.path.join(dir, file)
        if os.path.isdir(path):
            if recursive:
                convert_files(path, in_ext, action, recursive)
        elif path.endswith(in_ext):
            if exclude_ext is None or not path.endswith(exclude_ext):
                path_to_perform_action_list.append(path)
    with Pool(num_process) as p:
        p.map(action, path_to_perform_action_list)

