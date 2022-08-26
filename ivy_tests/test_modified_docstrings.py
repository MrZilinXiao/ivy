# global
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import pytest
import re
import os
import subprocess
from typing import List

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers

def get_changed_func_name(py_path: str) -> List[str]:
    ret = []
    with open(py_path) as f:
        code_lines = f.readlines()
        code_str = ''.join(code_lines)
    # purified lines for later function names retrieval
    code_lines = [code_line.strip() for code_line in code_lines]

    # get changed line number
    diff_command = f'git --no-pager diff "HEAD^..HEAD" --no-color -- {py_path}'
    diff_ret = subprocess.check_output(diff_command, shell=True).decode('utf-8')

    # parse diff_ret into changed line number from strings like `@@ -0,0 **+1**,2 @@`
    line_change_pattern = re.compile(r'^@@ -([0-9]+),([0-9]+) [+]([0-9]+),([0-9]+) @@', re.MULTILINE)
    changed_line_nums = [int(t[2]) for t in line_change_pattern.findall(diff_ret)]

    # find all possible docstring spans
    docstrings_pattern = re.compile(r'"""[\w\W]*?"""')
    new_line_pattern = re.compile(r'\n')
    func_name_pattern = re.compile(r'def (.*)\(')
    for m in docstrings_pattern.finditer(code_str):
        start_line = len(new_line_pattern.findall(code_str, 0, m.start(0)))+1
        end_line = len(new_line_pattern.findall(code_str, 0, m.end(0)))+1
        if any(start_line <= changed_line_num <= end_line for changed_line_num in changed_line_nums):
            # retrieve the function name corresponding to the start line
            def_line = code_lines[start_line - 1]  # `def asin(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:`
            func_name_lst = list(func_name_pattern.findall(def_line))
            assert len(func_name_lst) == 1, "Error Matching Function Names..."
            ret.extend(func_name_lst)
    return ret


@pytest.mark.parametrize("backend", ["jax", "numpy", "tensorflow", "torch"])
def test_docstrings(backend):
    ivy.set_default_device("cpu")
    ivy.set_backend(backend)
    failures = list()
    success = True

    """
        Functions skipped as their output dependent on outside factors:
            random_normal, random_uniform, shuffle, num_gpus, current_backend,
            get_backend

    """
    to_skip = [
        "random_normal",
        "random_uniform",
        "randint",
        "shuffle",
        "num_gpus",
        "current_backend",
        "get_backend",
        "namedtuple",
        "invalid_dtype",
        "DType",
        "Dtype",
        "multinomial",
        "num_cpu_cores",
        "get_all_ivy_arrays_on_dev",
        "num_ivy_arrays_on_dev",
        "total_mem_on_dev",
        "used_mem_on_dev",
        "percent_used_mem_on_dev",
        "function_unsupported_dtypes",
        "randint",
        "unique_counts",
        "unique_all",
        "total_mem_on_dev",
    ]
    # the temp skip list consists of functions which have an issue with their
    # implementation
    skip_list_temp = [
        "outer",
        "argmax",
        "split",
        "det",
        "cumprod",
        "where",
    ]

    # skip list for array and container docstrings
    skip_arr_cont = ["layer_norm"]
    # currently_being_worked_on = ["layer_norm"]

    # comment out the line below in future to check for the functions in temp skip list
    to_skip += skip_list_temp  # + currently_being_worked_on

    # read git diff filelist
    with open("ivy/name-changed") as f:
        changed_filepaths = f.readlines()

    for changed_filepath in [changed_filepath for changed_filepath in changed_filepaths if changed_filepath.endswith('.py')]:  # filtering py-code only files
        path_strs_lst = changed_filepath.split(os.path.pathsep)  # 'array', 'container' or others
        print(path_strs_lst)
        if path_strs_lst[0] != 'ivy' or len(path_strs_lst) <= 2:  # skip all non-ivy changes
            continue
        test_type = path_strs_lst[1]
        from_array = test_type == 'array'
        from_container = test_type == 'container'
        # for each changed diff file, decide what functions to be tested
        test_func_names = get_changed_func_name(changed_filepath)  # served as original dir(v)
        if from_array:
            for method_name in test_func_names:
                method = getattr(ivy.Array, method_name)
                if hasattr(ivy.functional, method_name):
                    grad_incomp_handle = getattr(ivy.functional, method_name)
                else:
                    grad_incomp_handle = method
                if (
                        method_name in skip_arr_cont
                        or helpers.gradient_incompatible_function(
                            fn=grad_incomp_handle
                        )
                        or helpers.docstring_examples_run(fn=method, from_array=True)
                    ):
                        continue
                success = False
                failures.append("Array." + method_name)

        elif from_container:
            for method_name in test_func_names:
                method = getattr(ivy.Container, method_name)
                if hasattr(ivy.functional, method_name):
                    grad_incomp_handle = getattr(ivy.functional, method_name)
                else:
                    grad_incomp_handle = method
                if (
                        method_name in skip_arr_cont
                        or helpers.gradient_incompatible_function(
                            fn=grad_incomp_handle
                        )
                        or helpers.docstring_examples_run(fn=method, from_container=True)
                    ):
                        continue
                success = False
                failures.append("Container." + method_name)

        else:  # ignore other changes first; figure it out later;
            raise NotImplementedError("Under Constrution...")
            # for method_name in test_func_names:
            #     if (
            #         k in to_skip
            #         or helpers.gradient_incompatible_function(fn=v)
            #         or helpers.docstring_examples_run(fn=v)
            #     ):
            #         continue
            #     success = False
            #     failures.append(k)

    if not success:
        assert (
            success
        ), "\nThe following methods had failing docstrings:\n\n{}\n".format(
            "\n".join(failures)
        )

    ivy.unset_backend()
