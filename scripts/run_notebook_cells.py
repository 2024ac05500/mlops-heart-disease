import sys
import json
import io
from contextlib import redirect_stdout, redirect_stderr


def execute_notebook_cells(input_nb_path: str, output_nb_path: str):
    with open(input_nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # execution environment
    env = {}

    exec_count = 1
    for cell in nb.get('cells', []):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "")

        stdout = io.StringIO()
        stderr = io.StringIO()
        outputs = []
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(compile(source, '<string>', 'exec'), env)
        except Exception as e:
            # capture exception text
            stderr.write(str(e))

        std_out_val = stdout.getvalue()
        std_err_val = stderr.getvalue()

        if std_out_val:
            outputs.append({"output_type": "stream", "name": "stdout", "text": std_out_val})
        if std_err_val:
            outputs.append({"output_type": "stream", "name": "stderr", "text": std_err_val})

        # notebooks saved as JSON: ensure outputs and execution_count fields exist
        cell['outputs'] = outputs
        cell['execution_count'] = exec_count
        exec_count += 1

    with open(output_nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Executed notebook saved to {output_nb_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_notebook_cells.py input.ipynb output.ipynb")
        sys.exit(1)
    execute_notebook_cells(sys.argv[1], sys.argv[2])
