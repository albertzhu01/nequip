import subprocess

params = {"train-dir": "/n/home10/axzhu/nequip/results/bpa/minimal_sub-no-train/",
          "model": "/n/home10/axzhu/nequip/results/bpa/minimal_sub-no-train/last_model.pth",
          "dataset-config": "/n/home10/axzhu/nequip/configs/bpa_600K.yaml",
          "output": "bpasub-no-train-600K.xyz",
          "output-fields": "node_features,atomic_energy",
          "log": "bpasub-no-train-600K"}

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

# nequip-evaluate --train-dir /n/home10/axzhu/nequip/results/bpa/minimal_sub-no-train/ --model /n/home10/axzhu/nequip/results/bpa/minimal_sub-no-train/last_model.pth --dataset-config /n/home10/axzhu/nequip/configs/bpa_600K.yaml --output bpasub-no-train-600K.xyz --output-fields node_features --log bpasub-no-train-600K
# nequip-evaluate --train-dir /n/home10/axzhu/nequip/results/bpa/minimal/ --dataset-config /n/home10/axzhu/nequip/configs/bpa_600K.yaml --output bpasub-600K.xyz --output-fields node_features --log bpasub-600K
