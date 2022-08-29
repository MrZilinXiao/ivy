import json
import subprocess

output = {}

with open('name-changed') as f:
    for diff_filename in f.readlines():
        diff_filename = diff_filename.strip()
        diff_command = 'git --no-pager diff "HEAD^..HEAD" --no-color -- {diff_filename}'
        try:
            diff_ret = subprocess.check_output(diff_command, shell=True, stderr=subprocess.STDOUT).decode('utf-8')
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

        output[diff_filename] = diff_ret

json.dump(output, open('name-changed.json', 'w'))