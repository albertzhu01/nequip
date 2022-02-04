import subprocess

params = {"train-dir": "/n/home10/axzhu/nequip/results/bpa/minimal/",
          "output": "bpa-min-sub.xyz"}

tmpdir = "/n/home10/axzhu/nequip/results/bpa/minimal/"
retcode = subprocess.run(
            ["nequip-evaluate"]
            + sum(
                (["--" + k, str(v)] for k, v in params.items() if v is not None),
                [],
            ),
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
retcode.check_returncode()

# Check the output
metrics = dict(
    [
        tuple(e.strip() for e in line.split("=", 1))
        for line in retcode.stdout.decode().splitlines()
    ]
)
metrics = {k: float(v) for k, v in metrics.items()}
