import subprocess

params = {"train-dir": "/n/home10/axzhu/nequip/results/bpa/minimal/",
          "model": "/n/home10/axzhu/nequip/results/bpa/minimal_sub-no-train/last_model.pth",
          "output": "bpasub.xyz"}

tmpdir = "/n/home10/axzhu/nequip/"
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

# import subprocess
#
# tmpdir = "/n/home10/axzhu/nequip/"
# retcode = subprocess.run(
#             ["nequip-train", "/n/home10/axzhu/nequip/configs/minimal_bpa.yaml"],
#             cwd=tmpdir,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#         )
# retcode.check_returncode()
